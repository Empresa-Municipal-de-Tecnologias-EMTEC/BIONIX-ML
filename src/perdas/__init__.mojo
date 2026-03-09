import src.perdas.mse as mse_impl
import src.nucleo.Tensor as tensor_defs


fn mse(pred: tensor_defs.Tensor, alvo: tensor_defs.Tensor) -> Float32:
    return mse_impl.mse(pred, alvo)


fn gradiente_mse(pred: tensor_defs.Tensor, alvo: tensor_defs.Tensor) -> tensor_defs.Tensor:
    return mse_impl.gradiente_mse(pred, alvo)
