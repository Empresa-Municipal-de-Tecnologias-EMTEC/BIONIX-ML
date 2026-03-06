import src.nucleo.Tensor as tensor_defs


fn somar_elemento_a_elemento_cpu(a: tensor_defs.Tensor, b: tensor_defs.Tensor) -> tensor_defs.Tensor:
    debug_assert(len(a.dados) == len(b.dados), "tensores devem ter mesmo tamanho")
    var formato = a.formato.copy()
    var saida = tensor_defs.Tensor(formato^, a.tipo_computacao)
    for i in range(len(a.dados)):
        saida.dados[i] = a.dados[i] + b.dados[i]
    return saida^


fn subtrair_elemento_a_elemento_cpu(a: tensor_defs.Tensor, b: tensor_defs.Tensor) -> tensor_defs.Tensor:
    debug_assert(len(a.dados) == len(b.dados), "tensores devem ter mesmo tamanho")
    var formato = a.formato.copy()
    var saida = tensor_defs.Tensor(formato^, a.tipo_computacao)
    for i in range(len(a.dados)):
        saida.dados[i] = a.dados[i] - b.dados[i]
    return saida^


fn multiplicar_elemento_a_elemento_cpu(a: tensor_defs.Tensor, b: tensor_defs.Tensor) -> tensor_defs.Tensor:
    debug_assert(len(a.dados) == len(b.dados), "tensores devem ter mesmo tamanho")
    var formato = a.formato.copy()
    var saida = tensor_defs.Tensor(formato^, a.tipo_computacao)
    for i in range(len(a.dados)):
        saida.dados[i] = a.dados[i] * b.dados[i]
    return saida^


fn transpor_cpu(a: tensor_defs.Tensor) -> tensor_defs.Tensor:
    debug_assert(len(a.formato) == 2, "transpor requer tensor 2D")
    var linhas = a.formato[0]
    var colunas = a.formato[1]
    var formato = List[Int]()
    formato.append(colunas)
    formato.append(linhas)
    var out = tensor_defs.Tensor(formato^, a.tipo_computacao)
    for i in range(linhas):
        for j in range(colunas):
            out.dados[j * linhas + i] = a.dados[i * colunas + j]
    return out^


fn multiplicar_matrizes_cpu(a: tensor_defs.Tensor, b: tensor_defs.Tensor) -> tensor_defs.Tensor:
    debug_assert(len(a.formato) == 2 and len(b.formato) == 2, "matmul requer tensores 2D")
    var m = a.formato[0]
    var n = a.formato[1]
    debug_assert(n == b.formato[0], "dimensões incompatíveis para matmul")
    var p = b.formato[1]
    var formato = List[Int]()
    formato.append(m)
    formato.append(p)
    var out = tensor_defs.Tensor(formato^, a.tipo_computacao)
    for i in range(m):
        for j in range(p):
            var acc: Float32 = 0.0
            for k in range(n):
                acc = acc + a.dados[i * n + k] * b.dados[k * p + j]
            out.dados[i * p + j] = acc
    return out^


fn adicionar_bias_coluna_cpu(a: tensor_defs.Tensor, b: tensor_defs.Tensor) -> tensor_defs.Tensor:
    debug_assert(len(a.formato) == 2, "entrada deve ser 2D")
    debug_assert(len(b.formato) == 2 and b.formato[0] == 1 and b.formato[1] == 1, "bias deve ter formato [1,1]")
    var formato = a.formato.copy()
    var out = tensor_defs.Tensor(formato^, a.tipo_computacao)
    var valor_bias = b.dados[0]
    for i in range(len(a.dados)):
        out.dados[i] = a.dados[i] + valor_bias
    return out^


fn soma_total_cpu(a: tensor_defs.Tensor) -> Float32:
    var s: Float32 = 0.0
    for i in range(len(a.dados)):
        s = s + a.dados[i]
    return s


fn erro_quadratico_medio_escalar_cpu(pred: tensor_defs.Tensor, alvo: tensor_defs.Tensor) -> Float32:
    debug_assert(len(pred.dados) == len(alvo.dados), "pred e alvo devem ter mesmo tamanho")
    if len(pred.dados) == 0:
        return 0.0
    var soma: Float32 = 0.0
    for i in range(len(pred.dados)):
        var d = pred.dados[i] - alvo.dados[i]
        soma = soma + d * d
    return soma / Float32(len(pred.dados))


fn gradiente_mse_cpu(pred: tensor_defs.Tensor, alvo: tensor_defs.Tensor) -> tensor_defs.Tensor:
    debug_assert(len(pred.dados) == len(alvo.dados), "pred e alvo devem ter mesmo tamanho")
    var formato = pred.formato.copy()
    var out = tensor_defs.Tensor(formato^, pred.tipo_computacao)
    if len(pred.dados) == 0:
        return out^
    var n = Float32(len(pred.dados))
    for i in range(len(pred.dados)):
        out.dados[i] = 2.0 * (pred.dados[i] - alvo.dados[i]) / n
    return out^
