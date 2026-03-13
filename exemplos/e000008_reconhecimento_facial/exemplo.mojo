import src.camadas.cnn as cnn_pkg
import src.conjuntos as conjuntos_pkg
import src.computacao as computacao_pkg
import src.nucleo.Tensor as tensor_defs


fn _fatia_linha(entradas: tensor_defs.Tensor, var idx: Int) -> tensor_defs.Tensor:
    var formato = List[Int]()
    formato.append(1)
    formato.append(entradas.formato[1])
    var out = tensor_defs.Tensor(formato^, entradas.tipo_computacao)
    var base = idx * entradas.formato[1]
    for i in range(entradas.formato[1]):
        out.dados[i] = entradas.dados[base + i]
    return out^


fn _classe_binaria(var prob: Float32) -> String:
    if prob >= 0.5:
        return "rosto"
    return "nao_rosto"


fn _acuracia(pred: tensor_defs.Tensor, alvos: tensor_defs.Tensor) -> Float32:
    if len(pred.dados) == 0:
        return 0.0
    var acertos = 0
    for i in range(len(pred.dados)):
        var p = 1.0 if pred.dados[i] >= 0.5 else Float32(0.0)
        if p == alvos.dados[i]:
            acertos = acertos + 1
    return Float32(acertos) / Float32(len(pred.dados))


def executar_exemplo():
    print("--- Exemplo e000008: reconhecimento_facial com bloco CNN ---")

    var caminho_dataset = "exemplos/e000008_reconhecimento_facial/dataset.txt"
    var tipo = computacao_pkg.backend_nome_de_id(computacao_pkg.backend_cpu_id())
    var conjunto = conjuntos_pkg.carregar_txt_supervisionado(caminho_dataset, 8, 8, tipo)

    if conjunto.entradas.formato[0] == 0:
        print("Dataset vazio ou invalido:", caminho_dataset)
        return

    print("Amostras:", conjunto.entradas.formato[0], "| Features:", conjunto.entradas.formato[1])

    var bloco = cnn_pkg.BlocoCNN(8, 8, 3, 3, 3, tipo)
    var loss = cnn_pkg.treinar(bloco, conjunto.entradas, conjunto.alvos, 0.08, 240, 40)
    print("Loss final:", loss)

    var pred = cnn_pkg.inferir(bloco, conjunto.entradas)
    var acc = _acuracia(pred, conjunto.alvos)
    print("Acuracia treino:", acc)

    print("\n--- Inferencia (3 amostras) ---")
    var max_demo = 3
    if conjunto.entradas.formato[0] < max_demo:
        max_demo = conjunto.entradas.formato[0]
    for i in range(max_demo):
        var amostra = _fatia_linha(conjunto.entradas, i)
        var p = cnn_pkg.inferir(bloco, amostra)
        var prob = p.dados[0]
        var real = _classe_binaria(conjunto.alvos.dados[i])
        var prev = _classe_binaria(prob)
        print("Amostra", i, "| real:", real, "| prev:", prev, "| prob:", prob)

    print("--- Fim do exemplo e000008 ---")
