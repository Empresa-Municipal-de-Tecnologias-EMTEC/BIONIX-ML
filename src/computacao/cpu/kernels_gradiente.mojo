import src.autograd.mlp as autograd_mlp
import src.nucleo.Tensor as tensor_defs

fn calcular_gradientes_mlp_cpu(
    ctx: autograd_mlp.MLPForwardContext,
    pesos: List[tensor_defs.Tensor],
) -> autograd_mlp.MLPGradientes:
    return autograd_mlp.calcular_gradientes(ctx, pesos)
