import src.camadas.linear as linear

def criar_linear(var num_entradas: Int, var tipo_computacao: String = "cpu") -> linear.CamadaLinear:
    return linear.CamadaLinear(num_entradas, tipo_computacao)