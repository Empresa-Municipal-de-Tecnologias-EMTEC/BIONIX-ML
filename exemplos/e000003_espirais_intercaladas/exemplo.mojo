import src.camadas.mlp as mlp_pkg
import src.dados as dados_pkg
import src.graficos as graficos_pkg
import src.conjuntos as conjuntos_pkg
import src.computacao as computacao_pkg
import src.uteis as uteis


fn _garantir_dataset_bmp(var caminho_bmp: String, var caminho_ok: String):
    var marcador = uteis.ler_texto_seguro(caminho_ok).strip()
    if marcador == "ok" and dados_pkg.diagnosticar_bmp(caminho_bmp):
        print("Dataset BMP já existe; reutilizando:", caminho_bmp)
        return

    print("Gerando dataset BMP de espirais intercaladas...")
    var bytes_bmp = graficos_pkg.gerar_bmp_espirais_intercaladas_bytes(192, 192)
    var ok = dados_pkg.gravar_arquivo_binario(caminho_bmp, bytes_bmp)
    if not ok:
        print("Falha ao gravar BMP em:", caminho_bmp)
        return

    _ = uteis.gravar_texto_seguro(caminho_ok, "ok")
    print("Dataset gerado:", caminho_bmp)


def executar_exemplo():
    print("--- Exemplo e000003: espirais intercaladas (BMP + autograd + ativações + MLP) ---")

    var tipo_computacao = computacao_pkg.backend_nome_de_id(computacao_pkg.backend_cpu_id())
    var caminho_bmp = "exemplos/e000003_espirais_intercaladas/dataset_espirais.bmp"
    var caminho_ok = "exemplos/e000003_espirais_intercaladas/dataset_espirais.ok"

    # 1) Geração condicional do dataset em BMP
    _garantir_dataset_bmp(caminho_bmp, caminho_ok)

    # 2) Carregamento do dataset supervisionado a partir de bitmap(s)
    var conjunto = conjuntos_pkg.carregar_bitmap_supervisionado(caminho_bmp, tipo_computacao, 2, 0.05, 0.6)
    var entradas = conjunto.entradas.copy()
    var alvos = conjunto.alvos.copy()

    if len(entradas.dados) == 0:
        print("Falha ao carregar dataset a partir do BMP.")
        return

    print("Amostras:", entradas.formato[0], "| Features:", entradas.formato[1])

    var prep_lotes = conjuntos_pkg.preparar_treino_validacao_em_lotes(conjunto, 128, 12, 0.2)
    print("Lotes de treino (epocas x lotes):", len(prep_lotes.treino_por_epoca), "| Lotes de validação:", len(prep_lotes.validacao))

    # 3) Treino do bloco MLP por lotes (autograd + funções de ativação)
    var mlp = mlp_pkg.BlocoMLP(2, 16, tipo_computacao)
    var loss_final = mlp_pkg.treinar_por_lotes(mlp, prep_lotes.treino_por_epoca, prep_lotes.validacao, 0.03, 1)
    print("Loss final:", loss_final)

    # 4) Métrica simples de acurácia
    var pred = mlp_pkg.inferir(mlp, entradas)
    var acertos = 0
    for i in range(entradas.formato[0]):
        var p = 1.0 if pred.dados[i] >= 0.5 else Float32(0.0)
        if p == alvos.dados[i]:
            acertos = acertos + 1
    var acc = Float32(acertos) / Float32(entradas.formato[0])
    print("Acurácia aproximada:", acc)

    print("--- Fim do exemplo e000003 ---")
