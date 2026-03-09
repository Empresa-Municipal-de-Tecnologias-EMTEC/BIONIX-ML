import src.camadas.mlp as mlp_pkg
import src.autograd as autograd
import src.computacao.dispatcher_gradiente as dispatcher_gradiente
import src.dados as dados_pkg
import src.graficos as graficos_pkg
import src.nucleo.Tensor as tensor_defs
import src.uteis as uteis
import os


fn _desenhar_retangulo_cheio(
    mut imagem: List[Int],
    var largura: Int,
    var altura: Int,
    var x0: Int,
    var y0: Int,
    var x1: Int,
    var y1: Int,
    var valor: Int,
):
    var xa = x0
    var xb = x1
    var ya = y0
    var yb = y1
    if xa > xb:
        var tmp = xa
        xa = xb
        xb = tmp
    if ya > yb:
        var tmp2 = ya
        ya = yb
        yb = tmp2

    if xa < 0:
        xa = 0
    if ya < 0:
        ya = 0
    if xb >= largura:
        xb = largura - 1
    if yb >= altura:
        yb = altura - 1

    if xa > xb or ya > yb:
        return

    for y in range(ya, yb + 1):
        for x in range(xa, xb + 1):
            imagem[y * largura + x] = valor


fn _desenhar_digito_sete_segmentos(
    mut imagem: List[Int],
    var largura: Int,
    var altura: Int,
    var digito: Int,
    var espessura: Int,
    var desloc_x: Int,
    var desloc_y: Int,
    var valor: Int,
):
    var seg = List[Bool]()
    for _ in range(7):
        seg.append(False)

    # a, b, c, d, e, f, g
    if digito == 0:
        seg[0] = True; seg[1] = True; seg[2] = True; seg[3] = True; seg[4] = True; seg[5] = True
    elif digito == 1:
        seg[1] = True; seg[2] = True
    elif digito == 2:
        seg[0] = True; seg[1] = True; seg[6] = True; seg[4] = True; seg[3] = True
    elif digito == 3:
        seg[0] = True; seg[1] = True; seg[6] = True; seg[2] = True; seg[3] = True
    elif digito == 4:
        seg[5] = True; seg[6] = True; seg[1] = True; seg[2] = True
    elif digito == 5:
        seg[0] = True; seg[5] = True; seg[6] = True; seg[2] = True; seg[3] = True
    elif digito == 6:
        seg[0] = True; seg[5] = True; seg[6] = True; seg[4] = True; seg[2] = True; seg[3] = True
    elif digito == 7:
        seg[0] = True; seg[1] = True; seg[2] = True
    elif digito == 8:
        for i in range(7):
            seg[i] = True
    else:  # 9
        seg[0] = True; seg[1] = True; seg[2] = True; seg[3] = True; seg[5] = True; seg[6] = True

    var margem_x = 3 + desloc_x
    var margem_y = 3 + desloc_y
    var xL = margem_x
    var xR = largura - margem_x - 1
    var yT = margem_y
    var yM = altura // 2
    var yB = altura - margem_y - 1
    var t = espessura

    # Segmentos horizontais: a, g, d
    if seg[0]:
        _desenhar_retangulo_cheio(imagem, largura, altura, xL + t, yT, xR - t, yT + t, valor)
    if seg[6]:
        _desenhar_retangulo_cheio(imagem, largura, altura, xL + t, yM - t // 2, xR - t, yM + t // 2, valor)
    if seg[3]:
        _desenhar_retangulo_cheio(imagem, largura, altura, xL + t, yB - t, xR - t, yB, valor)

    # Segmentos verticais: f, b, e, c
    if seg[5]:
        _desenhar_retangulo_cheio(imagem, largura, altura, xL, yT + t, xL + t, yM - t // 2, valor)
    if seg[1]:
        _desenhar_retangulo_cheio(imagem, largura, altura, xR - t, yT + t, xR, yM - t // 2, valor)
    if seg[4]:
        _desenhar_retangulo_cheio(imagem, largura, altura, xL, yM + t // 2, xL + t, yB - t, valor)
    if seg[2]:
        _desenhar_retangulo_cheio(imagem, largura, altura, xR - t, yM + t // 2, xR, yB - t, valor)


fn _gerar_imagem_digito(var largura: Int, var altura: Int, var digito: Int, var variante: Int) -> List[Int]:
    var img = graficos_pkg.criar_imagem_grayscale(largura, altura, 0)

    var espessura = 2 + (variante % 3)
    var dx = (variante % 5) - 2
    var dy = ((variante // 3) % 5) - 2
    var intensidade = 170 + ((variante * 17) % 70)
    if intensidade > 255:
        intensidade = 255

    _desenhar_digito_sete_segmentos(img, largura, altura, digito, espessura, dx, dy, intensidade)

    # Ruído determinístico leve
    var total = largura * altura
    for i in range(0, total, 37):
        var y = i // largura
        var x = i % largura
        var delta = (variante + x + y) % 15
        var idx = y * largura + x
        var v = img[idx] + delta
        if v > 255:
            v = 255
        img[idx] = v

    return img^


fn _garantir_dataset(var dir_dataset: String, var caminho_ok: String):
    var largura = 16
    var altura = 24

    var dir_treino = os.path.join(dir_dataset, "treino")
    var dir_teste = os.path.join(dir_dataset, "teste")
    try:
        os.makedirs(dir_treino, exist_ok=True)
        os.makedirs(dir_teste, exist_ok=True)
    except Exception:
        pass

    var marcador = uteis.ler_texto_seguro(caminho_ok).strip()
    if marcador == "ok":
        print("Dataset de dígitos já existe; reutilizando:", dir_dataset)
        return

    print("Gerando dataset de dígitos 0-9 (BMP)...")
    var amostras_treino_por_classe = 60
    var amostras_teste_por_classe = 20

    for classe in range(10):
        var dir_classe_treino = os.path.join(dir_treino, String(classe))
        var dir_classe_teste = os.path.join(dir_teste, String(classe))
        try:
            os.makedirs(dir_classe_treino, exist_ok=True)
            os.makedirs(dir_classe_teste, exist_ok=True)
        except Exception:
            pass

        for i in range(amostras_treino_por_classe):
            var img = _gerar_imagem_digito(largura, altura, classe, i)
            var bmp = graficos_pkg.gerar_bmp_24bits_de_grayscale(img, largura, altura)
            var caminho = os.path.join(dir_classe_treino, "d_" + String(classe) + "_" + String(i) + ".bmp")
            _ = dados_pkg.gravar_arquivo_binario(caminho, bmp^)

        for i in range(amostras_teste_por_classe):
            var img2 = _gerar_imagem_digito(largura, altura, classe, 1000 + i)
            var bmp2 = graficos_pkg.gerar_bmp_24bits_de_grayscale(img2, largura, altura)
            var caminho2 = os.path.join(dir_classe_teste, "d_" + String(classe) + "_" + String(i) + ".bmp")
            _ = dados_pkg.gravar_arquivo_binario(caminho2, bmp2^)

    _ = uteis.gravar_texto_seguro(caminho_ok, "ok")
    print("Dataset gerado em:", dir_dataset)


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


fn _carregar_dataset_digitos(var dir_raiz: String, var tipo_computacao: String) -> List[tensor_defs.Tensor]:
    var caminhos = _lista_arquivos_bmp(dir_raiz)
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
        out_vazio.append(vazio_x)
        out_vazio.append(vazio_y)
        return out_vazio^

    var primeira = dados_pkg.carregar_bmp_grayscale_matriz(caminhos[0])
    var altura = len(primeira)
    var largura = len(primeira[0]) if altura > 0 else 0
    var features = largura * altura
    var amostras = len(caminhos)

    var formato_x = List[Int]()
    formato_x.append(amostras)
    formato_x.append(features)
    var formato_y = List[Int]()
    formato_y.append(amostras)
    formato_y.append(10)

    var x_t = tensor_defs.Tensor(formato_x^, tipo_computacao)
    var y_t = tensor_defs.Tensor(formato_y^, tipo_computacao)

    for i in range(amostras):
        var m = dados_pkg.carregar_bmp_grayscale_matriz(caminhos[i])
        var label = _parse_label_de_caminho(caminhos[i])

        var k = 0
        for yy in range(altura):
            for xx in range(largura):
                x_t.dados[i * features + k] = m[yy][xx]
                k = k + 1

        for c in range(10):
            y_t.dados[i * 10 + c] = 1.0 if c == label else Float32(0.0)

    var out = List[tensor_defs.Tensor]()
    out.append(x_t)
    out.append(y_t)
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

        var pred_val = mlp_pkg.inferir(bloco, x_valid)
        var loss_val = tensor_defs.erro_quadratico_medio_escalar(pred_val, y_valid)
        var loss_treino_medio = soma_loss / Float32(lotes) if lotes > 0 else 0.0
        print("Época", epoca, "| Loss treino médio:", loss_treino_medio, "| Loss validação:", loss_val)


def executar_exemplo():
    print("--- Exemplo e000004: reconhecimento de dígitos (0-9) com MLP ---")

    var tipo_computacao = "cpu"
    var dir_dataset = "exemplos/e000004_reconhecimento_digitos/dataset"
    var caminho_ok = "exemplos/e000004_reconhecimento_digitos/dataset/dataset.ok"
    var dir_treino = os.path.join(dir_dataset, "treino")
    var dir_teste = os.path.join(dir_dataset, "teste")

    _garantir_dataset(dir_dataset, caminho_ok)

    var treino = _carregar_dataset_digitos(dir_treino, tipo_computacao)
    var teste = _carregar_dataset_digitos(dir_teste, tipo_computacao)
    var x_treino = treino[0].copy()
    var y_treino = treino[1].copy()
    var x_teste = teste[0].copy()
    var y_teste = teste[1].copy()

    if x_treino.formato[0] == 0 or x_teste.formato[0] == 0:
        print("Falha ao carregar dataset de dígitos.")
        return

    print("Amostras treino:", x_treino.formato[0], "| Features:", x_treino.formato[1])
    print("Amostras teste:", x_teste.formato[0], "| Classes:", y_treino.formato[1])

    var topologia = List[Int]()
    topologia.append(x_treino.formato[1])
    topologia.append(128)
    topologia.append(64)
    topologia.append(10)
    var mlp = mlp_pkg.BlocoMLP(topologia^, tipo_computacao)

    var epocas = 30
    var tamanho_lote = 64
    var taxa_aprendizado: Float32 = 0.02
    print("Epocas:", epocas, "| Lote:", tamanho_lote, "| LR:", taxa_aprendizado)

    _treinar_por_lotes_multiclasse(mlp, x_treino, y_treino, x_teste, y_teste, epocas, tamanho_lote, taxa_aprendizado)

    var pred_treino = mlp_pkg.inferir(mlp, x_treino)
    var pred_teste = mlp_pkg.inferir(mlp, x_teste)
    var acc_treino = _acuracia_multiclasse(pred_treino, y_treino)
    var acc_teste = _acuracia_multiclasse(pred_teste, y_teste)

    print("Acurácia treino (0-9):", acc_treino)
    print("Acurácia teste (0-9):", acc_teste)
    print("--- Fim do exemplo e000004 ---")
