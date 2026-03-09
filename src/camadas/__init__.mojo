import src.camadas.linear as linear
import src.camadas.mlp as mlp

def criar_linear(var num_entradas: Int, var tipo_computacao: String = "cpu") -> linear.CamadaLinear:
    return linear.CamadaLinear(num_entradas, tipo_computacao)


def criar_mlp(var num_entradas: Int, var num_ocultas: Int = 16, var tipo_computacao: String = "cpu") -> mlp.BlocoMLP:
    return mlp.BlocoMLP(num_entradas, num_ocultas, tipo_computacao)