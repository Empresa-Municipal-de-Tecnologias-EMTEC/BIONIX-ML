fn ativacao_saida_gelu_id() -> Int:
    return 43
fn ativacao_saida_sigmoid_id() -> Int:
    return 42
# Tipos de ativação de saída e perda da MLP (enum-like) e utilitários de validação/conversão.

fn ativacao_saida_hard_sigmoid_id() -> Int:
    return 0


fn ativacao_saida_linear_id() -> Int:
    return 1


fn ativacao_saida_softmax_id() -> Int:
    return 2


fn perda_mse_id() -> Int:
    return 0


fn perda_cross_entropy_id() -> Int:
    return 1


fn _normalizar_nome_ativacao_saida(var nome: String) -> String:
    var n = nome.strip()
    if n == "hard_sigmoid" or n == "HARD_SIGMOID" or n == "hardsigmoid":
        return "hard_sigmoid"
    if n == "linear" or n == "LINEAR":
        return "linear"
    if n == "softmax" or n == "SOFTMAX":
        return "softmax"
    if n == "sigmoid" or n == "SIGMOID":
        return "sigmoid"
    if n == "gelu" or n == "GELU":
        return "gelu"
    var out = ""
    for i in range(len(n)):
        out = out + n[i:i+1]
    return out


fn _normalizar_nome_perda(var nome: String) -> String:
    var n = nome.strip()
    if n == "mse" or n == "MSE":
        return "mse"
    if n == "cross_entropy" or n == "CROSS_ENTROPY" or n == "crossentropy":
        return "cross_entropy"
    var out = ""
    for i in range(len(n)):
        out = out + n[i:i+1]
    return out


fn ativacao_saida_id_de_nome(var nome: String) -> Int:
    var n = _normalizar_nome_ativacao_saida(nome)
    if n == "hard_sigmoid":
        return ativacao_saida_hard_sigmoid_id()
    if n == "linear":
        return ativacao_saida_linear_id()
    if n == "softmax":
        return ativacao_saida_softmax_id()
    if n == "sigmoid":
        return ativacao_saida_sigmoid_id()
    if n == "gelu":
        return ativacao_saida_gelu_id()
    return -1


fn ativacao_saida_nome_de_id(var ativacao_saida_id: Int) -> String:
    if ativacao_saida_id == ativacao_saida_hard_sigmoid_id():
        return "hard_sigmoid"
    if ativacao_saida_id == ativacao_saida_linear_id():
        return "linear"
    if ativacao_saida_id == ativacao_saida_softmax_id():
        return "softmax"
    if ativacao_saida_id == ativacao_saida_sigmoid_id():
        return "sigmoid"
    if ativacao_saida_id == ativacao_saida_gelu_id():
        return "gelu"
    return "desconhecida"


fn ativacao_saida_id_valido(var ativacao_saida_id: Int) -> Bool:
    return ativacao_saida_id == ativacao_saida_hard_sigmoid_id() or ativacao_saida_id == ativacao_saida_linear_id() or ativacao_saida_id == ativacao_saida_softmax_id() or ativacao_saida_id == ativacao_saida_sigmoid_id() or ativacao_saida_id == ativacao_saida_gelu_id()


fn perda_id_de_nome(var nome: String) -> Int:
    var n = _normalizar_nome_perda(nome)
    if n == "mse":
        return perda_mse_id()
    if n == "cross_entropy":
        return perda_cross_entropy_id()
    return -1


fn perda_nome_de_id(var perda_id: Int) -> String:
    if perda_id == perda_mse_id():
        return "mse"
    if perda_id == perda_cross_entropy_id():
        return "cross_entropy"
    return "desconhecida"


fn perda_id_valido(var perda_id: Int) -> Bool:
    return perda_id == perda_mse_id() or perda_id == perda_cross_entropy_id()


fn ativacao_saida_padrao_id(var num_saidas: Int) -> Int:
    if num_saidas <= 1:
        return ativacao_saida_hard_sigmoid_id()
    return ativacao_saida_softmax_id()


fn perda_padrao_id(var num_saidas: Int) -> Int:
    if num_saidas <= 1:
        return perda_mse_id()
    return perda_cross_entropy_id()