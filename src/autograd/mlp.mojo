import src.nucleo.Tensor as tensor_defs
import src.ativacoes as ativacoes
import src.perdas.mse as perdas_mse
import src.autograd.grafo as grafo

struct MLPForwardContext(Movable, Copyable):
    var entradas: tensor_defs.Tensor
    var alvos: tensor_defs.Tensor
    var z1: tensor_defs.Tensor
    var a1: tensor_defs.Tensor
    var z2: tensor_defs.Tensor
    var pred: tensor_defs.Tensor
    var operacoes: List[String]
    var grafo: grafo.GrafoComputacao

    fn __init__(
        out self,
        entradas_in: tensor_defs.Tensor,
        alvos_in: tensor_defs.Tensor,
        z1_in: tensor_defs.Tensor,
        a1_in: tensor_defs.Tensor,
        z2_in: tensor_defs.Tensor,
        pred_in: tensor_defs.Tensor,
        var ops_in: List[String],
        grafo_in: grafo.GrafoComputacao,
    ):
        self.entradas = entradas_in.copy()
        self.alvos = alvos_in.copy()
        self.z1 = z1_in.copy()
        self.a1 = a1_in.copy()
        self.z2 = z2_in.copy()
        self.pred = pred_in.copy()
        self.operacoes = ops_in^
        self.grafo = grafo_in.copy()


struct MLPGradientes(Movable, Copyable):
    var grad_w1: tensor_defs.Tensor
    var grad_b1: tensor_defs.Tensor
    var grad_w2: tensor_defs.Tensor
    var grad_b2: Float32
    var loss: Float32

    fn __init__(
        out self,
        gw1: tensor_defs.Tensor,
        gb1: tensor_defs.Tensor,
        gw2: tensor_defs.Tensor,
        var gb2: Float32,
        var loss_in: Float32,
    ):
        self.grad_w1 = gw1.copy()
        self.grad_b1 = gb1.copy()
        self.grad_w2 = gw2.copy()
        self.grad_b2 = gb2
        self.loss = loss_in


fn adicionar_bias_vetor_coluna(a: tensor_defs.Tensor, b: tensor_defs.Tensor) -> tensor_defs.Tensor:
    debug_assert(len(a.formato) == 2, "entrada deve ser 2D")
    debug_assert(len(b.formato) == 2 and b.formato[0] == 1 and b.formato[1] == a.formato[1], "bias deve ser [1, colunas]")
    var linhas = a.formato[0]
    var colunas = a.formato[1]
    var formato = a.formato.copy()
    var out = tensor_defs.Tensor(formato^, a.tipo_computacao)
    for i in range(linhas):
        for j in range(colunas):
            out.dados[i * colunas + j] = a.dados[i * colunas + j] + b.dados[j]
    return out^


fn somar_linhas(a: tensor_defs.Tensor) -> tensor_defs.Tensor:
    debug_assert(len(a.formato) == 2, "entrada deve ser 2D")
    var linhas = a.formato[0]
    var colunas = a.formato[1]
    var formato = List[Int]()
    formato.append(1)
    formato.append(colunas)
    var out = tensor_defs.Tensor(formato^, a.tipo_computacao)
    for j in range(colunas):
        var acc: Float32 = 0.0
        for i in range(linhas):
            acc = acc + a.dados[i * colunas + j]
        out.dados[j] = acc
    return out^


fn construir_contexto(
    entradas: tensor_defs.Tensor,
    alvos: tensor_defs.Tensor,
    w1: tensor_defs.Tensor,
    b1: tensor_defs.Tensor,
    w2: tensor_defs.Tensor,
    b2: tensor_defs.Tensor,
) -> MLPForwardContext:
    var ops = List[String]()

    var z1_bruto = tensor_defs.multiplicar_matrizes(entradas, w1)
    ops.append("matmul(x,w1)")
    var z1 = adicionar_bias_vetor_coluna(z1_bruto, b1)
    ops.append("add_bias_hidden")
    var a1 = ativacoes.relu(z1)
    ops.append("relu")
    var z2_bruto = tensor_defs.multiplicar_matrizes(a1, w2)
    ops.append("matmul(a1,w2)")
    var z2 = tensor_defs.adicionar_bias_coluna(z2_bruto, b2)
    ops.append("add_bias_out")
    var pred = ativacoes.hard_sigmoid(z2)
    ops.append("hard_sigmoid")

    var g = grafo.criar_grafo_mlp_forward()
    return MLPForwardContext(entradas, alvos, z1, a1, z2, pred, ops^, g)


fn calcular_gradientes(ctx: MLPForwardContext, w2: tensor_defs.Tensor) -> MLPGradientes:
    var loss = perdas_mse.mse(ctx.pred, ctx.alvos)

    var grad_pred = perdas_mse.gradiente_mse(ctx.pred, ctx.alvos)
    var grad_z2 = ativacoes.derivada_hard_sigmoid(ctx.z2, grad_pred)

    var a1_t = tensor_defs.transpor(ctx.a1)
    var grad_w2 = tensor_defs.multiplicar_matrizes(a1_t, grad_z2)
    var grad_b2 = tensor_defs.soma_total(grad_z2)

    var w2_t = tensor_defs.transpor(w2)
    var grad_a1 = tensor_defs.multiplicar_matrizes(grad_z2, w2_t)
    var grad_z1 = ativacoes.derivada_relu(ctx.z1, grad_a1)

    var x_t = tensor_defs.transpor(ctx.entradas)
    var grad_w1 = tensor_defs.multiplicar_matrizes(x_t, grad_z1)
    var grad_b1 = somar_linhas(grad_z1)

    return MLPGradientes(grad_w1, grad_b1, grad_w2, grad_b2, loss)
