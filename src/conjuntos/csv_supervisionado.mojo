import src.dados as dados_pkg
import src.dados.csv as csv_mod
import src.nucleo.Tensor as tensor_defs

struct ConjuntoSupervisionado(Movable, Copyable):
    var entradas: tensor_defs.Tensor
    var alvos: tensor_defs.Tensor
    var cabecalho: List[String]
    var indice_alvo: Int

    fn __init__(
        out self,
        var entradas_in: tensor_defs.Tensor,
        var alvos_in: tensor_defs.Tensor,
        var cabecalho_in: List[String],
        var indice_alvo_in: Int,
    ):
        self.entradas = entradas_in^
        self.alvos = alvos_in^
        self.cabecalho = cabecalho_in^
        self.indice_alvo = indice_alvo_in

    fn copy(self) -> ConjuntoSupervisionado:
        return ConjuntoSupervisionado(
            self.entradas.copy(),
            self.alvos.copy(),
            self.cabecalho.copy(),
            self.indice_alvo,
        )^


fn _digit_value(var ch: String) -> Int:
    if ch == "0":
        return 0
    if ch == "1":
        return 1
    if ch == "2":
        return 2
    if ch == "3":
        return 3
    if ch == "4":
        return 4
    if ch == "5":
        return 5
    if ch == "6":
        return 6
    if ch == "7":
        return 7
    if ch == "8":
        return 8
    if ch == "9":
        return 9
    return -1


fn _parse_float_ascii(var texto: String) -> Float32:
    var s = texto.strip().replace(",", ".")
    if len(s) == 0:
        return 0.0

    var sinal: Float32 = 1.0
    var i: Int = 0
    if s[0:1] == "-":
        sinal = -1.0
        i = 1
    elif s[0:1] == "+":
        i = 1

    var inteiro: Float32 = 0.0
    while i < len(s):
        var ch = s[i:i+1]
        if ch == ".":
            i = i + 1
            break
        var d = _digit_value(ch)
        if d < 0:
            return sinal * inteiro
        inteiro = inteiro * 10.0 + Float32(d)
        i = i + 1

    var frac: Float32 = 0.0
    var base: Float32 = 1.0
    while i < len(s):
        var d = _digit_value(s[i:i+1])
        if d < 0:
            break
        frac = frac * 10.0 + Float32(d)
        base = base * 10.0
        i = i + 1

    return sinal * (inteiro + (frac / base))


fn carregar_csv_supervisionado(
    var caminho: String,
    var indice_coluna_alvo: Int = -1,
    var delimitador: String = ",",
    var detectar_cabecalho: Bool = True,
    var tipo_computacao: String = "cpu",
) -> ConjuntoSupervisionado:
    var parsed = csv_mod.CSVData(List[String](), List[List[String]](), "")
    try:
        parsed = dados_pkg.carregar_csv(caminho, delimitador, detectar_cabecalho)
    except Exception:
        parsed = csv_mod.CSVData(List[String](), List[List[String]](), "")
    if len(parsed.linhas) == 0:
        var formato_x_vazio = List[Int]()
        formato_x_vazio.append(0)
        formato_x_vazio.append(0)
        var formato_y_vazio = List[Int]()
        formato_y_vazio.append(0)
        formato_y_vazio.append(1)
        return ConjuntoSupervisionado(
            tensor_defs.Tensor(formato_x_vazio^, tipo_computacao),
            tensor_defs.Tensor(formato_y_vazio^, tipo_computacao),
            parsed.cabecalho.copy(),
            0,
        )^

    var total_colunas = len(parsed.linhas[0])
    if total_colunas <= 1:
        var formato_x_invalido = List[Int]()
        formato_x_invalido.append(0)
        formato_x_invalido.append(0)
        var formato_y_invalido = List[Int]()
        formato_y_invalido.append(0)
        formato_y_invalido.append(1)
        return ConjuntoSupervisionado(
            tensor_defs.Tensor(formato_x_invalido^, tipo_computacao),
            tensor_defs.Tensor(formato_y_invalido^, tipo_computacao),
            parsed.cabecalho.copy(),
            0,
        )^

    var idx_alvo = indice_coluna_alvo
    if idx_alvo < 0:
        idx_alvo = total_colunas - 1
    if idx_alvo >= total_colunas:
        idx_alvo = total_colunas - 1

    var entradas_flat = List[Float32]()
    var alvos_flat = List[Float32]()
    var linhas_validas: Int = 0

    for r in parsed.linhas:
        if len(r) != total_colunas:
            continue
        var linha_ok = True
        var entrada_linha = List[Float32]()
        var alvo_linha: Float32 = 0.0
        for j in range(total_colunas):
            var campo = r[j]
            if len(campo.strip()) == 0:
                linha_ok = False
                break
            var v = _parse_float_ascii(campo)
            if j == idx_alvo:
                alvo_linha = v
            else:
                entrada_linha.append(v)
        if not linha_ok:
            continue
        for k in range(len(entrada_linha)):
            entradas_flat.append(entrada_linha[k])
        alvos_flat.append(alvo_linha)
        linhas_validas = linhas_validas + 1

    var colunas_entrada = total_colunas - 1
    var formato_x = List[Int]()
    formato_x.append(linhas_validas)
    formato_x.append(colunas_entrada)
    var formato_y = List[Int]()
    formato_y.append(linhas_validas)
    formato_y.append(1)

    var x = tensor_defs.Tensor(formato_x^, tipo_computacao)
    var y = tensor_defs.Tensor(formato_y^, tipo_computacao)
    for i in range(len(entradas_flat)):
        x.dados[i] = entradas_flat[i]
    for i in range(len(alvos_flat)):
        y.dados[i] = alvos_flat[i]

    return ConjuntoSupervisionado(x^, y^, parsed.cabecalho.copy(), idx_alvo)^