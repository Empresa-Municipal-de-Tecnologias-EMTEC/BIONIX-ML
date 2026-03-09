import src.camadas.mlp as mlp_pkg
import src.autograd as autograd
import src.computacao.dispatcher_gradiente as dispatcher_gradiente
import src.dados as dados_pkg
import src.nucleo.Tensor as tensor_defs
import os


fn _lista_arquivos_bmp(var dir_raiz: String) -> List[String]:
    var out = List[String]()
    var classes = os.listdir(dir_raiz)
    for c in classes:
        var dir_classe = os.path.join(dir_raiz, c)
        if not os.path.isdir(dir_classe):
            continue
        var arquivos = os.listdir(dir_classe)
        for nome in arquivos:
            if nome.endswith(".bmp"):
                out.append(os.path.join(dir_classe, nome))
    return out^


fn _parse_label_de_caminho(var caminho: String) -> Int:
    var normalizado = caminho.replace("\\", "/")
    var partes = normalizado.split("/")
    var ultima_pasta = partes[len(partes) - 2]
    try:
        return Int(ultima_pasta)
    except Exception:
        return 0


fn _carregar_dataset_digitos_de_arquivos(caminhos: List[String], var tipo_computacao: String) -> List[tensor_defs.Tensor]:
    if len(caminhos) == 0:
        var fx = List[Int]()
        fx.append(0)
        fx.append(0)
        var fy = List[Int]()
        fy.append(0)
        fy.append(10)
        var vazio_x = tensor_defs.Tensor(fx^, tipo_computacao)
        var vazio_y = tensor_defs.Tensor(fy^, tipo_computacao)
        var out_vazio = List[tensor_defs.Tensor]()
        out_vazio.append(vazio_x.copy())
        out_vazio.append(vazio_y.copy())
        return out_vazio^

    var primeira = List[List[Float32]]()
    try:
        var caminho0 = caminhos[0].copy()
        primeira = dados_pkg.carregar_bmp_grayscale_matriz(caminho0)
    except Exception:
        var fx_erro = List[Int]()
        fx_erro.append(0)
        fx_erro.append(0)
        var fy_erro = List[Int]()
        fy_erro.append(0)
        fy_erro.append(10)
        var vazio_x_erro = tensor_defs.Tensor(fx_erro^, tipo_computacao)
        var vazio_y_erro = tensor_defs.Tensor(fy_erro^, tipo_computacao)
        var out_vazio_erro = List[tensor_defs.Tensor]()
        out_vazio_erro.append(vazio_x_erro.copy())
        out_vazio_erro.append(vazio_y_erro.copy())
        return out_vazio_erro^

    var altura = len(primeira)
    var largura = len(primeira[0]) if altura > 0 else 0
    var features = largura * altura
    var amostras = len(caminhos)
    print("Carregando", amostras, "imagens...")

    var formato_x = List[Int]()
    formato_x.append(amostras)
    formato_x.append(features)
    var formato_y = List[Int]()
    formato_y.append(amostras)
    formato_y.append(10)

    var x_t = tensor_defs.Tensor(formato_x^, tipo_computacao)
    var y_t = tensor_defs.Tensor(formato_y^, tipo_computacao)

    for i in range(amostras):
        var m = List[List[Float32]]()
        try:
            var caminho_i = caminhos[i].copy()
            m = dados_pkg.carregar_bmp_grayscale_matriz(caminho_i)
        except Exception:
            m = primeira.copy()

        var label = _parse_label_de_caminho(caminhos[i].copy())

        var k = 0
        for yy in range(altura):
            for xx in range(largura):
                x_t.dados[i * features + k] = m[yy][xx]
                k = k + 1

        for c in range(10):
            y_t.dados[i * 10 + c] = 1.0 if c == label else Float32(0.0)

        if i > 0 and i % 500 == 0:
            print("  progresso:", i, "/", amostras)

    var out = List[tensor_defs.Tensor]()
    out.append(x_t.copy())
    out.append(y_t.copy())
    return out^


fn _dataset_tem_classes_0_a_9(var dir_dataset: String) -> Bool:
    for classe in range(10):
        var dir_classe = os.path.join(dir_dataset, String(classe))
        if not os.path.isdir(dir_classe):
            return False
    return True


fn _dividir_arquivos_treino_valid_teste(var dir_dataset: String) -> List[List[String]]:
    var treino = List[String]()
    var valid = List[String]()
    var teste = List[String]()

    # Split estratificado determinístico por classe: 70% treino, 15% validação, 15% teste
    for classe in range(10):
        var dir_classe = os.path.join(dir_dataset, String(classe))
        if not os.path.isdir(dir_classe):
            continue

        var arquivos = List[String]()
        try:
            arquivos = os.listdir(dir_classe)
        except Exception:
            continue
        var idx = 0
        var usados_classe = 0
        var max_por_classe = 320
        for nome in arquivos:
            if not nome.endswith(".bmp"):
                continue
            if max_por_classe > 0 and usados_classe >= max_por_classe:
                break

            var caminho = os.path.join(dir_classe, nome)
            var bucket = idx % 20
            if bucket < 14:
                treino.append(caminho)
            elif bucket < 17:
                valid.append(caminho)
            else:
                teste.append(caminho)
            idx = idx + 1
            usados_classe = usados_classe + 1

    var out = List[List[String]]()
    out.append(treino^)
    out.append(valid^)
    out.append(teste^)
    return out^


fn _fatiar_2d(t: tensor_defs.Tensor, var inicio: Int, var fim: Int) -> tensor_defs.Tensor:
    var linhas = t.formato[0]
    var colunas = t.formato[1]
    if inicio < 0:
        inicio = 0
    if fim > linhas:
        fim = linhas
    if fim < inicio:
        fim = inicio

    var n = fim - inicio
    var f = List[Int]()
    f.append(n)
    f.append(colunas)
    var out = tensor_defs.Tensor(f^, t.tipo_computacao)

    for i in range(n):
        var src_i = inicio + i
        for j in range(colunas):
            out.dados[i * colunas + j] = t.dados[src_i * colunas + j]

    return out^


fn _argmax_linha(t: tensor_defs.Tensor, var linha: Int) -> Int:
    var colunas = t.formato[1]
    var melhor = 0
    var melhor_v = t.dados[linha * colunas]
    for c in range(1, colunas):
        var v = t.dados[linha * colunas + c]
        if v > melhor_v:
            melhor_v = v
            melhor = c
    return melhor


fn _acuracia_multiclasse(pred: tensor_defs.Tensor, alvos_one_hot: tensor_defs.Tensor) -> Float32:
    var n = pred.formato[0]
    if n <= 0:
        return 0.0
    var acertos = 0
    for i in range(n):
        var p = _argmax_linha(pred, i)
        var y = _argmax_linha(alvos_one_hot, i)
        if p == y:
            acertos = acertos + 1
    return Float32(acertos) / Float32(n)


fn _treinar_por_lotes_multiclasse(
    mut bloco: mlp_pkg.BlocoMLP,
    x_treino: tensor_defs.Tensor,
    y_treino: tensor_defs.Tensor,
    x_valid: tensor_defs.Tensor,
    y_valid: tensor_defs.Tensor,
    var epocas: Int,
    var tamanho_lote: Int,
    var taxa_aprendizado: Float32,
):
    var total = x_treino.formato[0]
    if total <= 0:
        return

    if tamanho_lote <= 0:
        tamanho_lote = total

    for epoca in range(epocas):
        var soma_loss: Float32 = 0.0
        var lotes = 0

        var inicio = 0
        while inicio < total:
            var fim = inicio + tamanho_lote
            if fim > total:
                fim = total

            var xb = _fatiar_2d(x_treino, inicio, fim)
            var yb = _fatiar_2d(y_treino, inicio, fim)

            var ctx = autograd.construir_contexto_mlp(xb, yb, bloco.pesos, bloco.biases)
            var grads = dispatcher_gradiente.calcular_gradientes_mlp(ctx, bloco.pesos, True)
            soma_loss = soma_loss + grads.loss
            lotes = lotes + 1

            for camada in range(len(bloco.pesos)):
                for i in range(len(bloco.pesos[camada].dados)):
                    bloco.pesos[camada].dados[i] = bloco.pesos[camada].dados[i] - taxa_aprendizado * grads.grad_ws[camada].dados[i]
                for j in range(len(bloco.biases[camada].dados)):
                    bloco.biases[camada].dados[j] = bloco.biases[camada].dados[j] - taxa_aprendizado * grads.grad_bs[camada].dados[j]

            inicio = fim

        var pred_val = x_valid.copy()
        try:
            pred_val = mlp_pkg.inferir(bloco, x_valid)
        except Exception:
            print("Falha na inferência de validação durante o treino.")
            return
        var loss_val = tensor_defs.erro_quadratico_medio_escalar(pred_val, y_valid)
        var acc_val = _acuracia_multiclasse(pred_val, y_valid)
        var loss_treino_medio = soma_loss / Float32(lotes) if lotes > 0 else 0.0
        print("Época", epoca, "| Loss treino médio:", loss_treino_medio, "| Loss validação:", loss_val, "| Acc validação:", acc_val)


def executar_exemplo():
    print("--- Exemplo e000004: reconhecimento de dígitos (0-9) com MLP ---")

    var tipo_computacao = "cpu"
    var dir_dataset = "exemplos/e000004_reconhecimento_digitos/dataset"

    if not _dataset_tem_classes_0_a_9(dir_dataset):
        print("Dataset inválido: esperado", dir_dataset, "com subpastas 0..9 contendo BMPs.")
        return

    var arquivos_split = _dividir_arquivos_treino_valid_teste(dir_dataset)
    var arquivos_treino = arquivos_split[0].copy()
    var arquivos_valid = arquivos_split[1].copy()
    var arquivos_teste = arquivos_split[2].copy()
    print("Arquivos split | treino:", len(arquivos_treino), "| valid:", len(arquivos_valid), "| teste:", len(arquivos_teste))
    var treino = _carregar_dataset_digitos_de_arquivos(arquivos_treino, tipo_computacao)
    var valid = _carregar_dataset_digitos_de_arquivos(arquivos_valid, tipo_computacao)
    var teste = _carregar_dataset_digitos_de_arquivos(arquivos_teste, tipo_computacao)

    var x_treino = treino[0].copy()
    var y_treino = treino[1].copy()
    var x_valid = valid[0].copy()
    var y_valid = valid[1].copy()
    var x_teste = teste[0].copy()
    var y_teste = teste[1].copy()

    if x_treino.formato[0] == 0 or x_valid.formato[0] == 0 or x_teste.formato[0] == 0:
        print("Falha ao carregar dataset de dígitos.")
        return

    print("Amostras treino:", x_treino.formato[0], "| Features:", x_treino.formato[1])
    print("Amostras validação:", x_valid.formato[0], "| Amostras teste:", x_teste.formato[0], "| Classes:", y_treino.formato[1])

    var topologia = List[Int]()
    topologia.append(x_treino.formato[1])
    topologia.append(128)
    topologia.append(64)
    topologia.append(10)
    var mlp = mlp_pkg.BlocoMLP(topologia^, tipo_computacao)

    # Imagens 128x128 possuem alta dimensionalidade; configuração mais estável para esse cenário
    var epocas = 50
    var tamanho_lote = 32
    var taxa_aprendizado: Float32 = 0.01
    print("Epocas:", epocas, "| Lote:", tamanho_lote, "| LR:", taxa_aprendizado)

    _treinar_por_lotes_multiclasse(mlp, x_treino, y_treino, x_valid, y_valid, epocas, tamanho_lote, taxa_aprendizado)

    var pred_treino = mlp_pkg.inferir(mlp, x_treino)
    var pred_valid = mlp_pkg.inferir(mlp, x_valid)
    var pred_teste = mlp_pkg.inferir(mlp, x_teste)
    var acc_treino = _acuracia_multiclasse(pred_treino, y_treino)
    var acc_valid = _acuracia_multiclasse(pred_valid, y_valid)
    var acc_teste = _acuracia_multiclasse(pred_teste, y_teste)

    print("Acurácia treino (0-9):", acc_treino)
    print("Acurácia validação (0-9):", acc_valid)
    print("Acurácia teste (0-9):", acc_teste)
    print("--- Fim do exemplo e000004 ---")
