import src.dados as dados_pkg
import src.nucleo.Tensor as tensor_defs
import src.uteis as uteis


struct ConjuntoTXTSupervisionado(Movable, Copyable):
    var entradas: tensor_defs.Tensor
    var alvos: tensor_defs.Tensor
    var largura: Int
    var altura: Int

    fn __init__(out self, entradas_in: tensor_defs.Tensor, alvos_in: tensor_defs.Tensor, var largura_in: Int, var altura_in: Int):
        self.entradas = entradas_in^
        self.alvos = alvos_in^
        self.largura = largura_in
        self.altura = altura_in

    fn copy(self) -> ConjuntoTXTSupervisionado:
        return ConjuntoTXTSupervisionado(self.entradas.copy(), self.alvos.copy(), self.largura, self.altura)


fn _split_pipe(var linha: String) -> List[String]:
    var idx = -1
    for i in range(len(linha)):
        if linha[i:i+1] == "|":
            idx = i
            break
    var out = List[String]()
    if idx < 0:
        out.append(linha)
        out.append("")
        return out^
    out.append(linha[0:idx])
    out.append(linha[idx + 1:len(linha)])
    return out^


fn carregar_txt_supervisionado(
    var caminho: String,
    var largura: Int,
    var altura: Int,
    var tipo_computacao: String = "cpu",
) -> ConjuntoTXTSupervisionado:
    debug_assert(largura > 0 and altura > 0, "largura/altura invalidas")
    var linhas = dados_pkg.carregar_txt_linhas(caminho)
    var feat_dim = largura * altura

    var entradas_flat = List[Float32]()
    var alvos_flat = List[Float32]()
    var amostras = 0

    for linha in linhas:
        var l = linha.strip()
        if len(l) == 0:
            continue
        var partes = _split_pipe(l)
        var label_txt = partes[0].strip()
        var pixels_txt = partes[1].strip()
        if len(label_txt) == 0 or len(pixels_txt) == 0:
            continue

        var label = uteis.parse_float_ascii(label_txt)
        var itens = uteis.split_csv_simples(pixels_txt)
        if len(itens) != feat_dim:
            continue

        for i in range(feat_dim):
            var v = uteis.parse_float_ascii(String(itens[i]).strip())
            entradas_flat.append(v)
        alvos_flat.append(label)
        amostras = amostras + 1

    var formato_x = List[Int]()
    formato_x.append(amostras)
    formato_x.append(feat_dim)
    var formato_y = List[Int]()
    formato_y.append(amostras)
    formato_y.append(1)

    var x = tensor_defs.Tensor(formato_x^, tipo_computacao)
    var y = tensor_defs.Tensor(formato_y^, tipo_computacao)

    for i in range(len(entradas_flat)):
        x.dados[i] = entradas_flat[i]
    for i in range(len(alvos_flat)):
        y.dados[i] = alvos_flat[i]

    return ConjuntoTXTSupervisionado(x^, y^, largura, altura)
