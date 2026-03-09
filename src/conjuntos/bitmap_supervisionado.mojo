import os
import src.conjuntos.csv_supervisionado as csv_sup
import src.dados as dados_pkg
import src.dados.tipos_normalizacao as norm_tipos
import src.nucleo.Tensor as tensor_defs


fn _termina_com_bmp(var nome: String) -> Bool:
    if len(nome) < 4:
        return False
    var sufixo = nome[len(nome)-4:len(nome)]
    return sufixo == ".bmp" or sufixo == ".BMP"


fn _conjunto_vazio(var tipo_computacao: String) -> csv_sup.ConjuntoSupervisionado:
    var formato_x_vazio = List[Int]()
    formato_x_vazio.append(0)
    formato_x_vazio.append(2)
    var formato_y_vazio = List[Int]()
    formato_y_vazio.append(0)
    formato_y_vazio.append(1)

    var cab = List[String]()
    cab.append("x")
    cab.append("y")
    cab.append("classe")

    return csv_sup.ConjuntoSupervisionado(
        tensor_defs.Tensor(formato_x_vazio^, tipo_computacao),
        tensor_defs.Tensor(formato_y_vazio^, tipo_computacao),
        cab^,
        2,
        norm_tipos.normalizacao_nenhuma_id(),
        List[Float32](),
        List[Float32](),
        norm_tipos.normalizacao_nenhuma_id(),
        0.0,
        1.0,
    )^


fn _coletar_caminhos_bitmap(var caminho_ou_diretorio: String) -> List[String]:
    var caminhos = List[String]()
    try:
        if os.path.isdir(caminho_ou_diretorio):
            var nomes = os.listdir(caminho_ou_diretorio)
            for nome in nomes:
                var nome_str = String(nome)
                if _termina_com_bmp(nome_str):
                    caminhos.append(os.path.join(caminho_ou_diretorio, nome_str))
        else:
            caminhos.append(caminho_ou_diretorio)
    except Exception:
        caminhos.append(caminho_ou_diretorio)

    return caminhos^


fn carregar_bitmap_supervisionado(
    var caminho_ou_diretorio: String,
    var tipo_computacao: String = "cpu",
    var stride: Int = 2,
    var limiar_amostra: Float32 = 0.05,
    var limiar_classe: Float32 = 0.6,
) -> csv_sup.ConjuntoSupervisionado:
    if stride <= 0:
        stride = 1

    var caminhos = _coletar_caminhos_bitmap(caminho_ou_diretorio)
    if len(caminhos) == 0:
        return _conjunto_vazio(tipo_computacao)

    var entradas_flat = List[Float32]()
    var alvos_flat = List[Float32]()
    var n = 0

    for caminho_bmp in caminhos:
        var matriz = List[List[Float32]]()
        try:
            matriz = dados_pkg.carregar_bmp_grayscale_matriz(caminho_bmp)
        except Exception:
            continue
        if len(matriz) == 0 or len(matriz[0]) == 0:
            continue

        var h = len(matriz)
        var w = len(matriz[0])

        for y in range(0, h, stride):
            for x in range(0, w, stride):
                var v = matriz[y][x]
                if v <= limiar_amostra:
                    continue

                var nx = (Float32(x) / Float32(w - 1)) * 2.0 - 1.0
                var ny = (Float32(y) / Float32(h - 1)) * 2.0 - 1.0
                entradas_flat.append(nx)
                entradas_flat.append(ny)
                alvos_flat.append(1.0 if v > limiar_classe else Float32(0.0))
                n = n + 1

    if n == 0:
        return _conjunto_vazio(tipo_computacao)

    var formato_x = List[Int]()
    formato_x.append(n)
    formato_x.append(2)
    var formato_y = List[Int]()
    formato_y.append(n)
    formato_y.append(1)

    var x_t = tensor_defs.Tensor(formato_x^, tipo_computacao)
    var y_t = tensor_defs.Tensor(formato_y^, tipo_computacao)

    for i in range(len(entradas_flat)):
        x_t.dados[i] = entradas_flat[i]
    for i in range(len(alvos_flat)):
        y_t.dados[i] = alvos_flat[i]

    var cab = List[String]()
    cab.append("x")
    cab.append("y")
    cab.append("classe")

    return csv_sup.ConjuntoSupervisionado(
        x_t^,
        y_t^,
        cab^,
        2,
        norm_tipos.normalizacao_nenhuma_id(),
        List[Float32](),
        List[Float32](),
        norm_tipos.normalizacao_nenhuma_id(),
        0.0,
        1.0,
    )^
