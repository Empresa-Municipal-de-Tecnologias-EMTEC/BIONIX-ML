# Utilitários de texto e conversões ASCII simples

fn digit_value(var ch: String) -> Int:
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


fn parse_float_ascii(var texto: String) -> Float32:
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
        var d = digit_value(ch)
        if d < 0:
            return sinal * inteiro
        inteiro = inteiro * 10.0 + Float32(d)
        i = i + 1

    var frac: Float32 = 0.0
    var base: Float32 = 1.0
    while i < len(s):
        var d = digit_value(s[i:i+1])
        if d < 0:
            break
        frac = frac * 10.0 + Float32(d)
        base = base * 10.0
        i = i + 1

    return sinal * (inteiro + (frac / base))


fn float_list_para_csv(valores: List[Float32]) -> String:
    var out = ""
    for i in range(len(valores)):
        out = out + String(valores[i])
        if i < len(valores) - 1:
            out = out + ","
    return out


fn split_csv_simples(var texto: String) -> List[String]:
    var itens = List[String]()
    var buffer = ""
    for i in range(len(texto)):
        var c = texto[i:i+1]
        if c == ",":
            itens.append(buffer)
            buffer = ""
        else:
            buffer = buffer + c
    itens.append(buffer)
    return itens^


fn parse_linha_chave_valor(var linha: String) -> List[String]:
    var idx: Int = -1
    for i in range(len(linha)):
        if linha[i:i+1] == "=":
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