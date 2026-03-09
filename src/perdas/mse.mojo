import src.computacao.dispatcher_tensor as dispatcher
import src.nucleo.Tensor as tensor_defs

fn mse(pred: tensor_defs.Tensor, alvo: tensor_defs.Tensor) -> Float32:
    return dispatcher.erro_quadratico_medio_escalar(pred, alvo)


fn gradiente_mse(pred: tensor_defs.Tensor, alvo: tensor_defs.Tensor) -> tensor_defs.Tensor:
    return dispatcher.gradiente_mse(pred, alvo)
