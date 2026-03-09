import src.autograd.mlp as mlp
import src.autograd.grafo as grafo
import src.nucleo.Tensor as tensor_defs

alias MLPForwardContext = mlp.MLPForwardContext
alias MLPGradientes = mlp.MLPGradientes
alias GrafoComputacao = grafo.GrafoComputacao


fn construir_contexto_mlp(
    entradas: tensor_defs.Tensor,
    alvos: tensor_defs.Tensor,
    w1: tensor_defs.Tensor,
    b1: tensor_defs.Tensor,
    w2: tensor_defs.Tensor,
    b2: tensor_defs.Tensor,
) -> MLPForwardContext:
    return mlp.construir_contexto(entradas, alvos, w1, b1, w2, b2)


fn calcular_gradientes_mlp(ctx: MLPForwardContext, w2: tensor_defs.Tensor) -> MLPGradientes:
    return mlp.calcular_gradientes(ctx, w2)


fn adicionar_bias_vetor_coluna(a: tensor_defs.Tensor, b: tensor_defs.Tensor) -> tensor_defs.Tensor:
    return mlp.adicionar_bias_vetor_coluna(a, b)


fn criar_grafo_mlp_forward() -> GrafoComputacao:
    return grafo.criar_grafo_mlp_forward()
