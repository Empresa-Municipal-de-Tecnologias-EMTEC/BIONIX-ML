import src.autograd.mlp as mlp
import src.autograd.grafo as grafo
import src.nucleo.Tensor as tensor_defs

alias MLPForwardContext = mlp.MLPForwardContext
alias MLPGradientes = mlp.MLPGradientes
alias GrafoComputacao = grafo.GrafoComputacao


fn construir_contexto_mlp(
    entradas: tensor_defs.Tensor,
    alvos: tensor_defs.Tensor,
    pesos: List[tensor_defs.Tensor],
    biases: List[tensor_defs.Tensor],
) -> MLPForwardContext:
    return mlp.construir_contexto(entradas, alvos, pesos, biases)


fn calcular_gradientes_mlp(ctx: MLPForwardContext, pesos: List[tensor_defs.Tensor]) -> MLPGradientes:
    return mlp.calcular_gradientes(ctx, pesos)


fn adicionar_bias_vetor_coluna(a: tensor_defs.Tensor, b: tensor_defs.Tensor) -> tensor_defs.Tensor:
    return mlp.adicionar_bias_vetor_coluna(a, b)


fn criar_grafo_mlp_forward() -> GrafoComputacao:
    return grafo.criar_grafo_mlp_forward()
