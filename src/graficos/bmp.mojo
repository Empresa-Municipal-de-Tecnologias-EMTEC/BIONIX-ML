fn _append_u16_le(mut out: List[Int], var v: Int):
    out.append(v & 0xFF)
    out.append((v >> 8) & 0xFF)


fn _append_u32_le(mut out: List[Int], var v: Int):
    out.append(v & 0xFF)
    out.append((v >> 8) & 0xFF)
    out.append((v >> 16) & 0xFF)
    out.append((v >> 24) & 0xFF)


fn criar_imagem_grayscale(var largura: Int, var altura: Int, var valor_inicial: Int = 0) -> List[Int]:
    var imagem = List[Int](capacity=largura * altura)
    for _ in range(largura * altura):
        imagem.append(valor_inicial)
    return imagem^


fn desenhar_ponto_grayscale(mut imagem: List[Int], var largura: Int, var altura: Int, var x: Int, var y: Int, var valor: Int):
    if x < 0 or x >= largura or y < 0 or y >= altura:
        return
    imagem[y * largura + x] = valor


fn _plotar_espiral_grayscale_variada(
    mut imagem: List[Int],
    var largura: Int,
    var altura: Int,
    var classe: Int,
    var offset_passos_rotacao: Int,
    var delta_intensidade: Int,
):
    var c: Float32 = 0.9921147
    var s: Float32 = 0.1253332
    var vx: Float32 = 1.0
    var vy: Float32 = 0.0
    if classe == 1:
        vx = -1.0
        vy = 0.0

    var rot_steps = offset_passos_rotacao
    if rot_steps < 0:
        rot_steps = -rot_steps
    for _ in range(rot_steps):
        var rvx = vx * c - vy * s
        var rvy = vx * s + vy * c
        vx = rvx
        vy = rvy

    var passos = 820
    var intensidade = (96 if classe == 0 else 224) + delta_intensidade
    if intensidade < 0:
        intensidade = 0
    if intensidade > 255:
        intensidade = 255
    for i in range(passos):
        var t = Float32(i) / Float32(passos - 1)
        var r = 0.08 + 0.82 * t
        var px_f = (vx * r * 0.46 + 0.5) * Float32(largura - 1)
        var py_f = (vy * r * 0.46 + 0.5) * Float32(altura - 1)
        var px = Int(px_f)
        var py = Int(py_f)

        for dy in range(-1, 2):
            for dx in range(-1, 2):
                desenhar_ponto_grayscale(imagem, largura, altura, px + dx, py + dy, intensidade)

        var nvx = vx * c - vy * s
        var nvy = vx * s + vy * c
        vx = nvx
        vy = nvy


fn plotar_espiral_grayscale(mut imagem: List[Int], var largura: Int, var altura: Int, var classe: Int):
    _plotar_espiral_grayscale_variada(imagem, largura, altura, classe, 0, 0)


fn gerar_bmp_24bits_de_grayscale(var imagem: List[Int], var largura: Int, var altura: Int) -> List[Int]:
    debug_assert(len(imagem) == largura * altura, "imagem incompatível com dimensões")

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


fn gerar_bmp_espirais_intercaladas_bytes(var largura: Int = 192, var altura: Int = 192) -> List[Int]:
    var imagem = criar_imagem_grayscale(largura, altura, 0)
    plotar_espiral_grayscale(imagem, largura, altura, 0)
    plotar_espiral_grayscale(imagem, largura, altura, 1)
    return gerar_bmp_24bits_de_grayscale(imagem, largura, altura)


fn gerar_bmp_espirais_intercaladas_variacao_bytes(
    var largura: Int = 192,
    var altura: Int = 192,
    var offset_classe0: Int = 0,
    var offset_classe1: Int = 0,
    var delta_intensidade: Int = 0,
) -> List[Int]:
    var imagem = criar_imagem_grayscale(largura, altura, 0)
    _plotar_espiral_grayscale_variada(imagem, largura, altura, 0, offset_classe0, delta_intensidade)
    _plotar_espiral_grayscale_variada(imagem, largura, altura, 1, offset_classe1, -delta_intensidade)
    return gerar_bmp_24bits_de_grayscale(imagem, largura, altura)
