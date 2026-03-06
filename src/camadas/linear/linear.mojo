import src.nucleo.Tensor as tensor_defs

struct CamadaLinear(Movable, Copyable):
    var pesos: tensor_defs.Tensor
    var bias: tensor_defs.Tensor
    var tipo_computacao: String

    fn __init__(out self, var num_entradas: Int, var tipo_computacao_in: String = "cpu"):
        self.tipo_computacao = tipo_computacao_in^

        var formato_pesos = List[Int]()
        formato_pesos.append(num_entradas)
        formato_pesos.append(1)
        self.pesos = tensor_defs.Tensor(formato_pesos^, self.tipo_computacao)

        var formato_bias = List[Int]()
        formato_bias.append(1)
        formato_bias.append(1)
        self.bias = tensor_defs.Tensor(formato_bias^, self.tipo_computacao)

        for i in range(len(self.pesos.dados)):
            self.pesos.dados[i] = Float32(i + 1) * 0.01
        self.bias.dados[0] = 0.0

    fn copy(self) -> CamadaLinear:
        var nova = CamadaLinear(self.pesos.formato[0], self.tipo_computacao)
        nova.pesos = self.pesos.copy()
        nova.bias = self.bias.copy()
        return nova^


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


fn prever(camada: CamadaLinear, entradas: tensor_defs.Tensor) -> tensor_defs.Tensor:
    debug_assert(len(entradas.formato) == 2, "entradas deve ser tensor 2D")
    debug_assert(entradas.formato[1] == camada.pesos.formato[0], "número de features incompatível")

    var amostras = entradas.formato[0]
    var features = entradas.formato[1]
    var formato_saida = List[Int]()
    formato_saida.append(amostras)
    formato_saida.append(1)
    var saida = tensor_defs.Tensor(formato_saida^, camada.tipo_computacao)

    for i in range(amostras):
        var soma: Float32 = camada.bias.dados[0]
        for j in range(features):
            soma = soma + entradas.dados[i * features + j] * camada.pesos.dados[j]
        saida.dados[i] = soma
    return saida^


fn inferir(camada: CamadaLinear, entradas: tensor_defs.Tensor) -> tensor_defs.Tensor:
    return prever(camada, entradas)


fn erro_quadratico_medio(predicoes: tensor_defs.Tensor, alvos: tensor_defs.Tensor) -> Float32:
    debug_assert(len(predicoes.dados) == len(alvos.dados), "predições e alvos devem ter mesmo tamanho")
    if len(predicoes.dados) == 0:
        return 0.0
    var soma: Float32 = 0.0
    for i in range(len(predicoes.dados)):
        var d = predicoes.dados[i] - alvos.dados[i]
        soma = soma + d * d
    return soma / Float32(len(predicoes.dados))


fn treinar(
    mut camada: CamadaLinear,
    entradas: tensor_defs.Tensor,
    alvos: tensor_defs.Tensor,
    var taxa_aprendizado: Float32 = 0.05,
    var epocas: Int = 500,
    var imprimir_cada: Int = 100,
) -> Float32:
    debug_assert(len(entradas.formato) == 2, "entradas deve ser tensor 2D")
    debug_assert(len(alvos.formato) == 2, "alvos deve ser tensor 2D")
    debug_assert(entradas.formato[0] == alvos.formato[0], "número de linhas de entradas e alvos deve ser igual")
    debug_assert(alvos.formato[1] == 1, "alvos deve ter uma coluna")

    var amostras = entradas.formato[0]
    var features = entradas.formato[1]
    var n = Float32(amostras)
    var loss_final: Float32 = 0.0

    for epoca in range(epocas):
        var pred = prever(camada, entradas)
        loss_final = erro_quadratico_medio(pred, alvos)

        var grad_w = List[Float32]()
        for _ in range(features):
            grad_w.append(0.0)
        var grad_b: Float32 = 0.0

        for i in range(amostras):
            var erro = pred.dados[i] - alvos.dados[i]
            var grad_pred = 2.0 * erro / n
            grad_b = grad_b + grad_pred
            for j in range(features):
                grad_w[j] = grad_w[j] + grad_pred * entradas.dados[i * features + j]

        for j in range(features):
            camada.pesos.dados[j] = camada.pesos.dados[j] - taxa_aprendizado * grad_w[j]
        camada.bias.dados[0] = camada.bias.dados[0] - taxa_aprendizado * grad_b

        if imprimir_cada > 0 and (epoca % imprimir_cada == 0 or epoca == epocas - 1):
            print("Época", epoca, "| MSE:", loss_final)

    return loss_final


fn _float_list_para_texto(valores: List[Float32]) -> String:
    var out = ""
    for i in range(len(valores)):
        out = out + String(valores[i])
        if i < len(valores) - 1:
            out = out + ","
    return out


fn salvar_pesos(camada: CamadaLinear, var caminho: String):
    var f = open(caminho, "w")
    f.write("tipo=" + camada.tipo_computacao + "\n")
    f.write("num_entradas=" + String(camada.pesos.formato[0]) + "\n")
    f.write("pesos=" + _float_list_para_texto(camada.pesos.dados.copy()) + "\n")
    f.write("bias=" + String(camada.bias.dados[0]) + "\n")
    f.close()


fn _parse_linha_chave_valor(var linha: String) -> List[String]:
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


fn _split_csv_simples(var texto: String) -> List[String]:
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


fn carregar_pesos(var caminho: String, var tipo_computacao_padrao: String = "cpu") -> CamadaLinear:
    var f = open(caminho, "r")
    var conteudo = f.read()
    f.close()

    var tipo = tipo_computacao_padrao
    var num_entradas: Int = 0
    var pesos_lidos = List[Float32]()
    var bias_lido: Float32 = 0.0

    var linha = ""
    for i in range(len(conteudo)):
        var c = conteudo[i:i+1]
        if c == "\n":
            var kv = _parse_linha_chave_valor(linha)
            if len(kv) == 2:
                if kv[0] == "tipo":
                    tipo = kv[1]
                elif kv[0] == "num_entradas":
                    num_entradas = Int(_parse_float_ascii(kv[1]))
                elif kv[0] == "pesos":
                    var itens = _split_csv_simples(kv[1])
                    for it in itens:
                        if it.strip() != "":
                            pesos_lidos.append(_parse_float_ascii(it))
                elif kv[0] == "bias":
                    bias_lido = _parse_float_ascii(kv[1])
            linha = ""
        else:
            linha = linha + c

    if linha != "":
        var kv_last = _parse_linha_chave_valor(linha)
        if len(kv_last) == 2:
            if kv_last[0] == "tipo":
                tipo = kv_last[1]
            elif kv_last[0] == "num_entradas":
                num_entradas = Int(_parse_float_ascii(kv_last[1]))
            elif kv_last[0] == "pesos":
                var itens_last = _split_csv_simples(kv_last[1])
                for it in itens_last:
                    if it.strip() != "":
                        pesos_lidos.append(_parse_float_ascii(it))
            elif kv_last[0] == "bias":
                bias_lido = _parse_float_ascii(kv_last[1])

    if num_entradas <= 0:
        num_entradas = len(pesos_lidos)
    if num_entradas <= 0:
        num_entradas = 1

    var camada = CamadaLinear(num_entradas, tipo)
    var total_pesos = len(pesos_lidos)
    if len(camada.pesos.dados) < total_pesos:
        total_pesos = len(camada.pesos.dados)
    for i in range(total_pesos):
        camada.pesos.dados[i] = pesos_lidos[i]
    camada.bias.dados[0] = bias_lido
    return camada^