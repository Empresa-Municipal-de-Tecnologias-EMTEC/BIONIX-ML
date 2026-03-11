import src.nucleo.Tensor as tensor_defs
import src.computacao.dispatcher_ativacoes as dispatcher

fn identidade(x: tensor_defs.Tensor) -> tensor_defs.Tensor:
    return dispatcher.identidade(x)


fn derivada_identidade(entrada: tensor_defs.Tensor, grad_saida: tensor_defs.Tensor) -> tensor_defs.Tensor:
    return dispatcher.derivada_identidade(entrada, grad_saida)


fn relu(x: tensor_defs.Tensor) -> tensor_defs.Tensor:
    return dispatcher.relu(x)


fn derivada_relu(entrada: tensor_defs.Tensor, grad_saida: tensor_defs.Tensor) -> tensor_defs.Tensor:
    return dispatcher.derivada_relu(entrada, grad_saida)


fn hard_sigmoid(x: tensor_defs.Tensor) -> tensor_defs.Tensor:
    return dispatcher.hard_sigmoid(x)


fn derivada_hard_sigmoid(entrada: tensor_defs.Tensor, grad_saida: tensor_defs.Tensor) -> tensor_defs.Tensor:
    return dispatcher.derivada_hard_sigmoid(entrada, grad_saida)
