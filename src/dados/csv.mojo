# Parser CSV e helpers de I/O (texto e binário).

struct CSVData(Movable, Copyable):
    var cabecalho: List[String]
    var linhas: List[List[String]]
    var texto_original: String

    fn __init__(out self, var cab: List[String], var linhas_in: List[List[String]], var texto: String):
        self.cabecalho = cab^
        self.linhas = linhas_in^
        self.texto_original = texto

fn split_linhas(var texto: String) -> List[String]:
    var partes = List[String]()
    var buffer = ""
    for i in range(len(texto)):
        var c = texto[i:i+1]
        if c == "\n":
            partes.append(buffer)
            buffer = ""
        else:
            buffer = buffer + c
    if len(buffer) > 0:
        partes.append(buffer)
    return partes^

fn split_colunas(var linha: String, var delimitador: String) -> List[String]:
    var campos = List[String]()
    var campo = ""
    var dentro_quotes = False
    var i: Int = 0
    while i < len(linha):
        var c = linha[i:i+1]
        if c == '"':
            dentro_quotes = not dentro_quotes
        elif c == delimitador and not dentro_quotes:
            campos.append(campo)
            campo = ""
        else:
            campo = campo + c
        i = i + 1
    campos.append(campo)
    return campos^

fn detectar_cabecalho_por_conteudo(var primeira_linha: List[String]) -> Bool:
    var nao_numericos: Int = 0
    for f in primeira_linha:
        var s = f.strip()
        if s == "":
            nao_numericos = nao_numericos + 1
            continue
        var numerico = True
        var i: Int = 0
        var zero = "0"[0:1]
        var nine = "9"[0:1]
        var dot = "."[0:1]
        var comma = ","[0:1]
        var minus = "-"[0:1]
        var plus = "+"[0:1]
        var e_l = "e"[0:1]
        var e_U = "E"[0:1]
        while i < len(s):
            var ch = s[i:i+1]
            if not (ch >= zero and ch <= nine) and ch != dot and ch != comma and ch != minus and ch != plus and ch != e_l and ch != e_U:
                numerico = False
                break
            i = i + 1
        if not numerico:
            nao_numericos = nao_numericos + 1
    return nao_numericos * 2 >= len(primeira_linha)

fn parse_csv(var texto: String, var delimitador: String = ",", var detectar_cabecalho: Bool = True) -> CSVData:
    var linhas_texto = split_linhas(texto)
    var todas_linhas = List[List[String]]()
    for linha in linhas_texto:
        var trim = linha.strip()
        if trim == "":
            continue
        todas_linhas.append(split_colunas(linha, delimitador))

    var cab = List[String]()
    var inicio = 0
    if len(todas_linhas) == 0:
        return CSVData(cab, List[List[String]](), texto)^

    if detectar_cabecalho:
        var possivel = todas_linhas[0].copy()
        if detectar_cabecalho_por_conteudo(possivel):
            cab = possivel.copy()
            inicio = 1

    var dados = List[List[String]]()
    for i in range(inicio, len(todas_linhas)):
        dados.append(todas_linhas[i].copy())

    return CSVData(cab, dados, texto)^

fn ler_arquivo_texto(var caminho: String) -> String:
    try:
        var f = open(caminho, "r")
        var conteudo = f.read()
        f.close()
        return conteudo
    except Exception:
        return ""

fn ler_arquivo_binario(var caminho: String) -> List[Int]:
    # Retorna bytes como List[Int] para consumo simples
    try:
        var f = open(caminho, "rb")
        var raw = f.read()
        f.close()
        var out = List[Int]()
        for i in range(len(raw)):
            out.append(Int(raw[i]))
        return out^
    except Exception:
        return List[Int]()
