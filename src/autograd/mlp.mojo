import src.nucleo.Tensor as tensor_defs
import src.ativacoes as ativacoes
import src.perdas.mse as perdas_mse
import src.autograd.grafo as grafo

struct MLPForwardContext(Movable, Copyable):
    var entradas: tensor_defs.Tensor
    var alvos: tensor_defs.Tensor
    var zs: List[tensor_defs.Tensor]
    var ativacoes: List[tensor_defs.Tensor]
    var pred: tensor_defs.Tensor
    var operacoes: List[String]
    var grafo: grafo.GrafoComputacao

    fn __init__(
        out self,
        entradas_in: tensor_defs.Tensor,
        alvos_in: tensor_defs.Tensor,
        var zs_in: List[tensor_defs.Tensor],
        var ativacoes_in: List[tensor_defs.Tensor],
        pred_in: tensor_defs.Tensor,
        var ops_in: List[String],
        grafo_in: grafo.GrafoComputacao,
    ):
        self.entradas = entradas_in.copy()
        self.alvos = alvos_in.copy()
        self.zs = List[tensor_defs.Tensor]()
        for z in zs_in:
            self.zs.append(z.copy())
        self.ativacoes = List[tensor_defs.Tensor]()
        for a in ativacoes_in:
            self.ativacoes.append(a.copy())
        self.pred = pred_in.copy()
        self.operacoes = ops_in^
        self.grafo = grafo_in.copy()


struct MLPGradientes(Movable, Copyable):
    var grad_ws: List[tensor_defs.Tensor]
    var grad_bs: List[tensor_defs.Tensor]
    var loss: Float32

    fn __init__(
        out self,
        var gws: List[tensor_defs.Tensor],
        var gbs: List[tensor_defs.Tensor],
        var loss_in: Float32,
    ):
        self.grad_ws = List[tensor_defs.Tensor]()
        for gw in gws:
            self.grad_ws.append(gw.copy())
        self.grad_bs = List[tensor_defs.Tensor]()
        for gb in gbs:
            self.grad_bs.append(gb.copy())
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
    pesos: List[tensor_defs.Tensor],
    biases: List[tensor_defs.Tensor],
) -> MLPForwardContext:
    debug_assert(len(pesos) > 0, "MLP precisa de ao menos uma camada")
    debug_assert(len(pesos) == len(biases), "pesos e biases precisam ter mesmo tamanho")

    var ops = List[String]()

    var zs = List[tensor_defs.Tensor]()
    var ativs = List[tensor_defs.Tensor]()
    ativs.append(entradas.copy())

    var atual = entradas.copy()
    var num_camadas = len(pesos)

    for camada in range(num_camadas):
        var z_bruto = tensor_defs.multiplicar_matrizes(atual, pesos[camada])
        ops.append("matmul(a" + String(camada) + ",w" + String(camada + 1) + ")")

        var z = adicionar_bias_vetor_coluna(z_bruto, biases[camada])
        ops.append("add_bias_" + String(camada + 1))
        zs.append(z.copy())

        if camada < num_camadas - 1:
            atual = ativacoes.relu(z)
            ops.append("relu_" + String(camada + 1))
        else:
            atual = ativacoes.hard_sigmoid(z)
            ops.append("hard_sigmoid_out")

        ativs.append(atual.copy())

    var pred = atual.copy()

    var topologia = List[Int]()
    topologia.append(entradas.formato[1])
    for p in pesos:
        topologia.append(p.formato[1])

    var g = grafo.criar_grafo_mlp_forward_topologia(topologia)
    return MLPForwardContext(entradas, alvos, zs^, ativs^, pred, ops^, g)


fn calcular_gradientes(ctx: MLPForwardContext, pesos: List[tensor_defs.Tensor]) -> MLPGradientes:
    var num_camadas = len(pesos)
    debug_assert(num_camadas > 0, "MLP precisa de ao menos uma camada")
    debug_assert(num_camadas == len(ctx.zs), "contexto inconsistente com número de camadas")

    var loss = perdas_mse.mse(ctx.pred, ctx.alvos)

    var grad_pred = perdas_mse.gradiente_mse(ctx.pred, ctx.alvos)
    var z_saida = ctx.zs[num_camadas - 1].copy()
    var grad_z_atual = ativacoes.derivada_hard_sigmoid(z_saida, grad_pred)

    var grad_ws_reverso = List[tensor_defs.Tensor]()
    var grad_bs_reverso = List[tensor_defs.Tensor]()

    for passo in range(num_camadas):
        var camada = num_camadas - 1 - passo

        var ativ_prev = ctx.ativacoes[camada].copy()
        var ativ_prev_t = tensor_defs.transpor(ativ_prev)
        var grad_w = tensor_defs.multiplicar_matrizes(ativ_prev_t, grad_z_atual)
        var grad_b = somar_linhas(grad_z_atual)

        grad_ws_reverso.append(grad_w.copy())
        grad_bs_reverso.append(grad_b.copy())

        if camada > 0:
            var peso_camada = pesos[camada].copy()
            var w_t = tensor_defs.transpor(peso_camada)
            var grad_a_prev = tensor_defs.multiplicar_matrizes(grad_z_atual, w_t)
            var z_anterior = ctx.zs[camada - 1].copy()
            grad_z_atual = ativacoes.derivada_relu(z_anterior, grad_a_prev)

    var grad_ws = List[tensor_defs.Tensor]()
    var grad_bs = List[tensor_defs.Tensor]()
    for i in range(num_camadas):
        var idx = num_camadas - 1 - i
        grad_ws.append(grad_ws_reverso[idx].copy())
        grad_bs.append(grad_bs_reverso[idx].copy())

    return MLPGradientes(grad_ws^, grad_bs^, loss)
