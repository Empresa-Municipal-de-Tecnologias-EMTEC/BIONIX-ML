import src.camadas.mlp as mlp_pkg
import src.dados as dados_pkg
import src.computacao as computacao_pkg
import src.nucleo.Tensor as tensor_defs
import src.uteis as uteis

fn _append_u16_le(mut out: List[Int], var v: Int):
    out.append(v & 0xFF)
    out.append((v >> 8) & 0xFF)


fn _append_u32_le(mut out: List[Int], var v: Int):
    out.append(v & 0xFF)
    out.append((v >> 8) & 0xFF)
    out.append((v >> 16) & 0xFF)
    out.append((v >> 24) & 0xFF)


fn _desenhar_ponto(mut imagem: List[Int], var largura: Int, var altura: Int, var x: Int, var y: Int, var valor: Int):
    if x < 0 or x >= largura or y < 0 or y >= altura:
        return
    imagem[y * largura + x] = valor


fn _plot_espiral(mut imagem: List[Int], var largura: Int, var altura: Int, var classe: Int):
    var c: Float32 = 0.9921147
    var s: Float32 = 0.1253332
    var vx: Float32 = 1.0
    var vy: Float32 = 0.0
    if classe == 1:
        vx = -1.0
        vy = 0.0

    var passos = 820
    var intensidade = 96 if classe == 0 else 224
    for i in range(passos):
        var t = Float32(i) / Float32(passos - 1)
        var r = 0.08 + 0.82 * t
        var px_f = (vx * r * 0.46 + 0.5) * Float32(largura - 1)
        var py_f = (vy * r * 0.46 + 0.5) * Float32(altura - 1)
        var px = Int(px_f)
        var py = Int(py_f)

        for dy in range(-1, 2):
            for dx in range(-1, 2):
                _desenhar_ponto(imagem, largura, altura, px + dx, py + dy, intensidade)

        var nvx = vx * c - vy * s
        var nvy = vx * s + vy * c
        vx = nvx
        vy = nvy


fn _gerar_bmp_espirais_bytes(var largura: Int = 192, var altura: Int = 192) -> List[Int]:
    var imagem = List[Int](capacity=largura * altura)
    for _ in range(largura * altura):
        imagem.append(0)

    _plot_espiral(imagem, largura, altura, 0)
    _plot_espiral(imagem, largura, altura, 1)

    var bytes_por_pixel = 3
    var row_raw = largura * bytes_por_pixel
    var row_stride = ((row_raw + 3) // 4) * 4
    var image_size = row_stride * altura
    var file_size = 54 + image_size

    var out = List[Int](capacity=file_size)
    out.append(0x42)
    out.append(0x4D)
    _append_u32_le(out, file_size)
    _append_u16_le(out, 0)
    _append_u16_le(out, 0)
    _append_u32_le(out, 54)

    _append_u32_le(out, 40)
    _append_u32_le(out, largura)
    _append_u32_le(out, altura)
    _append_u16_le(out, 1)
    _append_u16_le(out, 24)
    _append_u32_le(out, 0)
    _append_u32_le(out, image_size)
    _append_u32_le(out, 2835)
    _append_u32_le(out, 2835)
    _append_u32_le(out, 0)
    _append_u32_le(out, 0)

    for y in reversed(range(altura)):
        for x in range(largura):
            var v = imagem[y * largura + x]
            out.append(v)
            out.append(v)
            out.append(v)
        for _ in range(row_stride - row_raw):
            out.append(0)

    return out^


fn _gravar_binario(var caminho: String, var dados: List[Int]) -> Bool:
    try:
        var f = open(caminho, "w")
        f.write_bytes(dados)
        f.close()
        return True
    except Exception:
        return False


fn _garantir_dataset_bmp(var caminho_bmp: String, var caminho_ok: String):
    var marcador = uteis.ler_texto_seguro(caminho_ok).strip()
    if marcador == "ok" and dados_pkg.diagnosticar_bmp(caminho_bmp):
        print("Dataset BMP já existe; reutilizando:", caminho_bmp)
        return

    print("Gerando dataset BMP de espirais intercaladas...")
    var bytes_bmp = _gerar_bmp_espirais_bytes(192, 192)
    var ok = _gravar_binario(caminho_bmp, bytes_bmp)
    if not ok:
        print("Falha ao gravar BMP em:", caminho_bmp)
        return

    _ = uteis.gravar_texto_seguro(caminho_ok, "ok")
    print("Dataset gerado:", caminho_bmp)


fn _dataset_de_bmp_para_tensores(var caminho_bmp: String, var tipo_computacao: String) -> List[tensor_defs.Tensor]:
    var matriz = dados_pkg.carregar_bmp_grayscale_matriz(caminho_bmp)
    if len(matriz) == 0 or len(matriz[0]) == 0:
        var vazio = List[tensor_defs.Tensor]()
        var fx = List[Int]()
        fx.append(0)
        fx.append(2)
        var fy = List[Int]()
        fy.append(0)
        fy.append(1)
        vazio.append(tensor_defs.Tensor(fx^, tipo_computacao))
        vazio.append(tensor_defs.Tensor(fy^, tipo_computacao))
        return vazio^

    var h = len(matriz)
    var w = len(matriz[0])
    var stride = 2

    var n = 0
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            if matriz[y][x] > 0.05:
                n = n + 1

    var formato_x = List[Int]()
    formato_x.append(n)
    formato_x.append(2)
    var formato_y = List[Int]()
    formato_y.append(n)
    formato_y.append(1)
    var x_t = tensor_defs.Tensor(formato_x^, tipo_computacao)
    var y_t = tensor_defs.Tensor(formato_y^, tipo_computacao)

    var k = 0
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            var v = matriz[y][x]
            if v <= 0.05:
                continue

            var nx = (Float32(x) / Float32(w - 1)) * 2.0 - 1.0
            var ny = (Float32(y) / Float32(h - 1)) * 2.0 - 1.0
            x_t.dados[k * 2 + 0] = nx
            x_t.dados[k * 2 + 1] = ny
            y_t.dados[k] = 1.0 if v > 0.6 else Float32(0.0)
            k = k + 1

    var out = List[tensor_defs.Tensor]()
    out.append(x_t^)
    out.append(y_t^)
    return out^


def executar_exemplo():
    print("--- Exemplo e000003: espirais intercaladas (BMP + autograd + ativações + MLP) ---")

    var tipo_computacao = computacao_pkg.backend_nome_de_id(computacao_pkg.backend_cpu_id())
    var caminho_bmp = "exemplos/e000003_espirais_intercaladas/dataset_espirais.bmp"
    var caminho_ok = "exemplos/e000003_espirais_intercaladas/dataset_espirais.ok"

    # 1) Geração condicional do dataset em BMP
    _garantir_dataset_bmp(caminho_bmp, caminho_ok)

    # 2) Carregamento do dataset usando os mecanismos já existentes
    var dados = _dataset_de_bmp_para_tensores(caminho_bmp, tipo_computacao)
    var entradas = dados[0].copy()
    var alvos = dados[1].copy()

    if len(entradas.dados) == 0:
        print("Falha ao carregar dataset a partir do BMP.")
        return

    print("Amostras:", entradas.formato[0], "| Features:", entradas.formato[1])

    # 3) Treino do bloco MLP (autograd + funções de ativação)
    var mlp = mlp_pkg.BlocoMLP(2, 16, tipo_computacao)
    var loss_final = mlp_pkg.treinar(mlp, entradas, alvos, 0.03, 1200, 300)
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
