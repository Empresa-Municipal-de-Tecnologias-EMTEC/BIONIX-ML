import src.nucleo.Tensor as tensor_defs

fn identidade(x: tensor_defs.Tensor) -> tensor_defs.Tensor:
    return x.copy()


fn derivada_identidade(entrada: tensor_defs.Tensor, grad_saida: tensor_defs.Tensor) -> tensor_defs.Tensor:
    debug_assert(len(entrada.dados) == len(grad_saida.dados), "gradiente incompatível")
    var formato = entrada.formato.copy()
    var grad = tensor_defs.Tensor(formato^, entrada.tipo_computacao)
    for i in range(len(grad_saida.dados)):
        grad.dados[i] = grad_saida.dados[i]
    return grad^


fn relu(x: tensor_defs.Tensor) -> tensor_defs.Tensor:
    var formato = x.formato.copy()
    var out = tensor_defs.Tensor(formato^, x.tipo_computacao)
    for i in range(len(x.dados)):
        var v = x.dados[i]
        out.dados[i] = v if v > 0.0 else Float32(0.0)
    return out^


fn derivada_relu(entrada: tensor_defs.Tensor, grad_saida: tensor_defs.Tensor) -> tensor_defs.Tensor:
    debug_assert(len(entrada.dados) == len(grad_saida.dados), "gradiente incompatível")
    var formato = entrada.formato.copy()
    var grad = tensor_defs.Tensor(formato^, entrada.tipo_computacao)
    for i in range(len(entrada.dados)):
        grad.dados[i] = grad_saida.dados[i] if entrada.dados[i] > 0.0 else Float32(0.0)
    return grad^


fn hard_sigmoid(x: tensor_defs.Tensor) -> tensor_defs.Tensor:
    var formato = x.formato.copy()
    var out = tensor_defs.Tensor(formato^, x.tipo_computacao)
    for i in range(len(x.dados)):
        var v = 0.2 * x.dados[i] + 0.5
        if v < 0.0:
            v = 0.0
        if v > 1.0:
            v = 1.0
        out.dados[i] = v
    return out^


fn derivada_hard_sigmoid(entrada: tensor_defs.Tensor, grad_saida: tensor_defs.Tensor) -> tensor_defs.Tensor:
    debug_assert(len(entrada.dados) == len(grad_saida.dados), "gradiente incompatível")
    var formato = entrada.formato.copy()
    var grad = tensor_defs.Tensor(formato^, entrada.tipo_computacao)
    for i in range(len(entrada.dados)):
        var x = entrada.dados[i]
        var d: Float32 = 0.0
        if x > -2.5 and x < 2.5:
            d = 0.2
        grad.dados[i] = grad_saida.dados[i] * d
    return grad^
