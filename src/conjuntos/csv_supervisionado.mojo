import src.dados as dados_pkg
import src.dados.csv as csv_mod
import src.nucleo.Tensor as tensor_defs
import math

struct ConjuntoSupervisionado(Movable, Copyable):
    var entradas: tensor_defs.Tensor
    var alvos: tensor_defs.Tensor
    var cabecalho: List[String]
    var indice_alvo: Int
    var tipo_normalizacao_entradas: String
    var media_entradas: List[Float32]
    var desvio_entradas: List[Float32]
    var tipo_normalizacao_alvo: String
    var media_alvo: Float32
    var desvio_alvo: Float32

    fn __init__(
        out self,
        var entradas_in: tensor_defs.Tensor,
        var alvos_in: tensor_defs.Tensor,
        var cabecalho_in: List[String],
        var indice_alvo_in: Int,
        var tipo_normalizacao_entradas_in: String,
        var media_entradas_in: List[Float32],
        var desvio_entradas_in: List[Float32],
        var tipo_normalizacao_alvo_in: String,
        var media_alvo_in: Float32,
        var desvio_alvo_in: Float32,
    ):
        self.entradas = entradas_in^
        self.alvos = alvos_in^
        self.cabecalho = cabecalho_in^
        self.indice_alvo = indice_alvo_in
        self.tipo_normalizacao_entradas = tipo_normalizacao_entradas_in^
        self.media_entradas = media_entradas_in^
        self.desvio_entradas = desvio_entradas_in^
        self.tipo_normalizacao_alvo = tipo_normalizacao_alvo_in^
        self.media_alvo = media_alvo_in
        self.desvio_alvo = desvio_alvo_in

    fn copy(self) -> ConjuntoSupervisionado:
        return ConjuntoSupervisionado(
            self.entradas.copy(),
            self.alvos.copy(),
            self.cabecalho.copy(),
            self.indice_alvo,
            self.tipo_normalizacao_entradas,
            self.media_entradas.copy(),
            self.desvio_entradas.copy(),
            self.tipo_normalizacao_alvo,
            self.media_alvo,
            self.desvio_alvo,
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


fn _conjunto_vazio(var tipo_computacao: String) -> ConjuntoSupervisionado:
    var formato_x_vazio = List[Int]()
    formato_x_vazio.append(0)
    formato_x_vazio.append(0)
    var formato_y_vazio = List[Int]()
    formato_y_vazio.append(0)
    formato_y_vazio.append(1)
    return ConjuntoSupervisionado(
        tensor_defs.Tensor(formato_x_vazio^, tipo_computacao),
        tensor_defs.Tensor(formato_y_vazio^, tipo_computacao),
        List[String](),
        0,
        "nenhuma",
        List[Float32](),
        List[Float32](),
        "nenhuma",
        0.0,
        1.0,
    )^


fn carregar_csv_supervisionado(
    var caminho: String,
    var indice_coluna_alvo: Int = -1,
    var delimitador: String = ",",
    var detectar_cabecalho: Bool = True,
    var tipo_computacao: String = "cpu",
    var normalizacao_entradas: String = "zscore",
    var normalizacao_alvo: String = "zscore",
) -> ConjuntoSupervisionado:
    #print("[debug csv_sup] inicio")
    var parsed = csv_mod.CSVData(List[String](), List[List[String]](), "")
    #print("[debug csv_sup] csvdata vazio criado")
    try:
        parsed = dados_pkg.carregar_csv(caminho, delimitador, detectar_cabecalho)
        #print("[debug csv_sup] csv carregado. linhas=", len(parsed.linhas))
    except Exception:
        #print("[debug csv_sup] falha no carregar_csv; retornando conjunto vazio")
        return _conjunto_vazio(tipo_computacao)
    if len(parsed.linhas) == 0:
        #print("[debug csv_sup] csv sem linhas")
        return _conjunto_vazio(tipo_computacao)

    var total_colunas = len(parsed.linhas[0])
    #print("[debug csv_sup] total_colunas=", total_colunas)
    if total_colunas <= 1:
        #print("[debug csv_sup] csv com <=1 coluna")
        return _conjunto_vazio(tipo_computacao)

    var idx_alvo = indice_coluna_alvo
    if idx_alvo < 0:
        idx_alvo = total_colunas - 1
    if idx_alvo >= total_colunas:
        idx_alvo = total_colunas - 1

    var entradas_matriz = List[List[Float32]]()
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
        entradas_matriz.append(entrada_linha.copy())
        alvos_flat.append(alvo_linha)
        linhas_validas = linhas_validas + 1

    #print("[debug csv_sup] linhas_validas=", linhas_validas)

    var tipo_norm = "nenhuma"
    var medias = List[Float32]()
    var desvios = List[Float32]()
    if normalizacao_entradas == "zscore" and len(entradas_matriz) > 0 and len(entradas_matriz[0]) > 0:
        #print("[debug csv_sup] iniciando zscore entradas")
        try:
            var norm = dados_pkg.normalizar_zscore(entradas_matriz.copy())
            entradas_matriz = norm.dados_normalizados.copy()
            medias = norm.media_por_coluna.copy()
            desvios = norm.desvio_por_coluna.copy()
            tipo_norm = "zscore"
            #print("[debug csv_sup] zscore entradas ok")
        except Exception:
            #print("[debug csv_sup] zscore entradas falhou")
            tipo_norm = "nenhuma"

    var tipo_norm_alvo = "nenhuma"
    var media_alvo: Float32 = 0.0
    var desvio_alvo: Float32 = 1.0
    if normalizacao_alvo == "zscore" and len(alvos_flat) > 0:
        #print("[debug csv_sup] iniciando zscore alvo")
        var soma_alvo: Float32 = 0.0
        for i in range(len(alvos_flat)):
            soma_alvo = soma_alvo + alvos_flat[i]
        media_alvo = soma_alvo / Float32(len(alvos_flat))

        var soma_quad: Float32 = 0.0
        for i in range(len(alvos_flat)):
            var d = alvos_flat[i] - media_alvo
            soma_quad = soma_quad + d * d
        var variancia = soma_quad / Float32(len(alvos_flat))
        desvio_alvo = Float32(math.sqrt(variancia))

        if desvio_alvo == 0.0:
            for i in range(len(alvos_flat)):
                alvos_flat[i] = 0.0
        else:
            for i in range(len(alvos_flat)):
                alvos_flat[i] = (alvos_flat[i] - media_alvo) / desvio_alvo
        tipo_norm_alvo = "zscore"
        #print("[debug csv_sup] zscore alvo ok")

    var entradas_flat = List[Float32]()
    for i in range(len(entradas_matriz)):
        for j in range(len(entradas_matriz[i])):
            entradas_flat.append(entradas_matriz[i][j])

    var colunas_entrada = total_colunas - 1
    var formato_x = List[Int]()
    formato_x.append(linhas_validas)
    formato_x.append(colunas_entrada)
    var formato_y = List[Int]()
    formato_y.append(linhas_validas)
    formato_y.append(1)

    var x = tensor_defs.Tensor(formato_x^, tipo_computacao)
    var y = tensor_defs.Tensor(formato_y^, tipo_computacao)
    #print("[debug csv_sup] tensores alocados")
    for i in range(len(entradas_flat)):
        x.dados[i] = entradas_flat[i]
    for i in range(len(alvos_flat)):
        y.dados[i] = alvos_flat[i]

    #print("[debug csv_sup] retorno conjunto")

    return ConjuntoSupervisionado(
        x^,
        y^,
        parsed.cabecalho.copy(),
        idx_alvo,
        tipo_norm,
        medias^,
        desvios^,
        tipo_norm_alvo,
        media_alvo,
        desvio_alvo,
    )^


fn normalizar_amostra_entradas(conjunto: ConjuntoSupervisionado, amostra: List[Float32]) -> List[Float32]:
    if conjunto.tipo_normalizacao_entradas != "zscore":
        return amostra.copy()

    var out = List[Float32]()
    for i in range(len(amostra)):
        if i >= len(conjunto.media_entradas) or i >= len(conjunto.desvio_entradas):
            out.append(amostra[i])
            continue
        var d = conjunto.desvio_entradas[i]
        if d == 0.0:
            out.append(0.0)
        else:
            out.append((amostra[i] - conjunto.media_entradas[i]) / d)
    return out^


fn desnormalizar_valor_alvo(conjunto: ConjuntoSupervisionado, valor_normalizado: Float32) -> Float32:
    if conjunto.tipo_normalizacao_alvo != "zscore":
        return valor_normalizado
    return valor_normalizado * conjunto.desvio_alvo + conjunto.media_alvo