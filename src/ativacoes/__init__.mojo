import src.ativacoes.basicas as basicas
import src.nucleo.Tensor as tensor_defs


def identidade(x: tensor_defs.Tensor) -> tensor_defs.Tensor:
    return basicas.identidade(x)


def derivada_identidade(entrada: tensor_defs.Tensor, grad_saida: tensor_defs.Tensor) -> tensor_defs.Tensor:
    return basicas.derivada_identidade(entrada, grad_saida)


def relu(x: tensor_defs.Tensor) -> tensor_defs.Tensor:
    return basicas.relu(x)


def derivada_relu(entrada: tensor_defs.Tensor, grad_saida: tensor_defs.Tensor) -> tensor_defs.Tensor:
    return basicas.derivada_relu(entrada, grad_saida)


def hard_sigmoid(x: tensor_defs.Tensor) -> tensor_defs.Tensor:
    return basicas.hard_sigmoid(x)


def derivada_hard_sigmoid(entrada: tensor_defs.Tensor, grad_saida: tensor_defs.Tensor) -> tensor_defs.Tensor:
    return basicas.derivada_hard_sigmoid(entrada, grad_saida)
