import src.camadas.linear.linear as linear_impl
import src.nucleo.Tensor as tensor_defs

alias CamadaLinear = linear_impl.CamadaLinear

def prever(camada: CamadaLinear, entradas: tensor_defs.Tensor):
    return linear_impl.prever(camada, entradas)

def inferir(camada: CamadaLinear, entradas: tensor_defs.Tensor) -> tensor_defs.Tensor:
    return linear_impl.inferir(camada, entradas)

def treinar(
    mut camada: CamadaLinear,
    entradas: tensor_defs.Tensor,
    alvos: tensor_defs.Tensor,
    var taxa_aprendizado: Float32 = 0.05,
    var epocas: Int = 500,
    var imprimir_cada: Int = 100,
) -> Float32:
    return linear_impl.treinar(camada, entradas, alvos, taxa_aprendizado, epocas, imprimir_cada)

def salvar_pesos(camada: CamadaLinear, var caminho: String):
    linear_impl.salvar_pesos(camada, caminho)

def carregar_pesos(var caminho: String, var tipo_computacao_padrao: String = "cpu") -> CamadaLinear:
    return linear_impl.carregar_pesos(caminho, tipo_computacao_padrao)