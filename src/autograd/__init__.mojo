import src.autograd.mlp as mlp
import src.nucleo.Tensor as tensor_defs

alias MLPForwardContext = mlp.MLPForwardContext
alias MLPGradientes = mlp.MLPGradientes


def construir_contexto_mlp(
    entradas: tensor_defs.Tensor,
    alvos: tensor_defs.Tensor,
    w1: tensor_defs.Tensor,
    b1: tensor_defs.Tensor,
    w2: tensor_defs.Tensor,
    b2: tensor_defs.Tensor,
) -> MLPForwardContext:
    return mlp.construir_contexto(entradas, alvos, w1, b1, w2, b2)


def calcular_gradientes_mlp(ctx: MLPForwardContext, w2: tensor_defs.Tensor) -> MLPGradientes:
    return mlp.calcular_gradientes(ctx, w2)


def adicionar_bias_vetor_coluna(a: tensor_defs.Tensor, b: tensor_defs.Tensor) -> tensor_defs.Tensor:
    return mlp.adicionar_bias_vetor_coluna(a, b)
