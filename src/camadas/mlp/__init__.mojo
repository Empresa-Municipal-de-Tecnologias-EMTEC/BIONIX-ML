import src.camadas.mlp.mlp as mlp_impl
import src.nucleo.Tensor as tensor_defs

alias BlocoMLP = mlp_impl.BlocoMLP


def prever(bloco: BlocoMLP, entradas: tensor_defs.Tensor) -> tensor_defs.Tensor:
    return mlp_impl.prever(bloco, entradas)


def inferir(bloco: BlocoMLP, entradas: tensor_defs.Tensor) -> tensor_defs.Tensor:
    return mlp_impl.inferir(bloco, entradas)


def treinar(
    mut bloco: BlocoMLP,
    entradas: tensor_defs.Tensor,
    alvos: tensor_defs.Tensor,
    var taxa_aprendizado: Float32 = 0.03,
    var epocas: Int = 1200,
    var imprimir_cada: Int = 200,
) -> Float32:
    return mlp_impl.treinar(bloco, entradas, alvos, taxa_aprendizado, epocas, imprimir_cada)
