import src.camadas.linear as linear
import src.camadas.mlp as mlp
import src.camadas.transformer as transformer
import src.camadas.cnn as cnn

def criar_linear(var num_entradas: Int, var tipo_computacao: String = "cpu") -> linear.CamadaLinear:
    return linear.CamadaLinear(num_entradas, tipo_computacao)


def criar_mlp(var num_entradas: Int, var num_ocultas: Int = 16, var tipo_computacao: String = "cpu") -> mlp.BlocoMLP:
    return mlp.BlocoMLP(num_entradas, num_ocultas, tipo_computacao)


def criar_mlp_topologia(var topologia: List[Int], var tipo_computacao: String = "cpu") -> mlp.BlocoMLP:
    return mlp.BlocoMLP(topologia, tipo_computacao)


def criar_bloco_transformer_base(
    var vocab_size: Int,
    var dimensao_modelo: Int,
    var num_heads: Int = 4,
    var tipo_computacao: String = "cpu",
) -> transformer.BlocoTransformerBase:
    return transformer.criar_bloco_transformer_base(vocab_size, dimensao_modelo, num_heads, tipo_computacao)


def criar_bloco_cnn(
    var altura: Int,
    var largura: Int,
    var num_filtros: Int = 2,
    var kernel_h: Int = 3,
    var kernel_w: Int = 3,
    var tipo_computacao: String = "cpu",
) -> cnn.BlocoCNN:
    return cnn.BlocoCNN(altura, largura, num_filtros, kernel_h, kernel_w, tipo_computacao)