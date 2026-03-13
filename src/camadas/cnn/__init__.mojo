import src.camadas.cnn.cnn as cnn_impl
import src.nucleo.Tensor as tensor_defs

alias BlocoCNN = cnn_impl.BlocoCNN


def prever(bloco: BlocoCNN, entradas: tensor_defs.Tensor) -> tensor_defs.Tensor:
    return cnn_impl.prever(bloco, entradas)


def inferir(bloco: BlocoCNN, entradas: tensor_defs.Tensor) -> tensor_defs.Tensor:
    return cnn_impl.inferir(bloco, entradas)


def extrair_features(bloco: BlocoCNN, entradas: tensor_defs.Tensor) -> tensor_defs.Tensor:
    return cnn_impl.extrair_features(bloco, entradas)


def treinar(
    mut bloco: BlocoCNN,
    entradas: tensor_defs.Tensor,
    alvos: tensor_defs.Tensor,
    var taxa_aprendizado: Float32 = 0.05,
    var epocas: Int = 300,
    var imprimir_cada: Int = 50,
) -> Float32:
    return cnn_impl.treinar(bloco, entradas, alvos, taxa_aprendizado, epocas, imprimir_cada)
