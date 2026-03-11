import src.nucleo.Tensor as tensor_defs
import src.ativacoes as ativacoes
import src.perdas.mse as perdas_mse
import src.autograd.grafo as grafo
import src.autograd.tipos_mlp as tipos_mlp
import math

struct MLPForwardContext(Movable, Copyable):
    var entradas: tensor_defs.Tensor
    var alvos: tensor_defs.Tensor
    var zs: List[tensor_defs.Tensor]
    var ativacoes: List[tensor_defs.Tensor]
    var pred: tensor_defs.Tensor
    var ativacao_saida_id: Int
    var perda_id: Int
    var operacoes: List[String]
    var grafo: grafo.GrafoComputacao

    fn __init__(
        out self,
        entradas_in: tensor_defs.Tensor,
        alvos_in: tensor_defs.Tensor,
        var zs_in: List[tensor_defs.Tensor],
        var ativacoes_in: List[tensor_defs.Tensor],
        pred_in: tensor_defs.Tensor,
        var ativacao_saida_id_in: Int,
        var perda_id_in: Int,
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
        self.ativacao_saida_id = ativacao_saida_id_in
        self.perda_id = perda_id_in
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


fn _softmax_linhas(z: tensor_defs.Tensor) -> tensor_defs.Tensor:
    debug_assert(len(z.formato) == 2, "softmax espera tensor 2D")
    var linhas = z.formato[0]
    var colunas = z.formato[1]
    var formato = z.formato.copy()
    var out = tensor_defs.Tensor(formato^, z.tipo_computacao)

    for i in range(linhas):
        var max_v = z.dados[i * colunas]
        for j in range(1, colunas):
            var v = z.dados[i * colunas + j]
            if v > max_v:
                max_v = v

        var soma_exp: Float32 = 0.0
        for j in range(colunas):
            var e = Float32(math.exp(Float64(z.dados[i * colunas + j] - max_v)))
            out.dados[i * colunas + j] = e
            soma_exp = soma_exp + e

        if soma_exp <= 0.0:
            soma_exp = 1.0

        for j in range(colunas):
            out.dados[i * colunas + j] = out.dados[i * colunas + j] / soma_exp

    return out^


fn _loss_cross_entropy_media(pred_prob: tensor_defs.Tensor, alvos: tensor_defs.Tensor) -> Float32:
    debug_assert(len(pred_prob.formato) == 2 and len(alvos.formato) == 2, "cross-entropy espera tensores 2D")
    debug_assert(pred_prob.formato[0] == alvos.formato[0] and pred_prob.formato[1] == alvos.formato[1], "pred e alvo incompatíveis")
    var linhas = pred_prob.formato[0]
    var colunas = pred_prob.formato[1]
    if linhas <= 0:
        return 0.0

    var eps: Float32 = 1e-7
    var soma: Float32 = 0.0
    for i in range(linhas):
        for j in range(colunas):
            var y = alvos.dados[i * colunas + j]
            if y > 0.0:
                var p = pred_prob.dados[i * colunas + j]
                if p < eps:
                    p = eps
                if p > 1.0 - eps:
                    p = 1.0 - eps
                soma = soma - y * Float32(math.log(Float64(p)))

    return soma / Float32(linhas)


fn _grad_softmax_cross_entropy(pred_prob: tensor_defs.Tensor, alvos: tensor_defs.Tensor) -> tensor_defs.Tensor:
    debug_assert(len(pred_prob.formato) == 2 and len(alvos.formato) == 2, "grad softmax+ce espera tensores 2D")
    debug_assert(pred_prob.formato[0] == alvos.formato[0] and pred_prob.formato[1] == alvos.formato[1], "pred e alvo incompatíveis")
    var linhas = pred_prob.formato[0]
    var formato = pred_prob.formato.copy()
    var out = tensor_defs.Tensor(formato^, pred_prob.tipo_computacao)
    if linhas <= 0:
        return out^

    var n = Float32(linhas)
    for i in range(len(pred_prob.dados)):
        out.dados[i] = (pred_prob.dados[i] - alvos.dados[i]) / n

    return out^


fn calcular_loss_configurado(pred: tensor_defs.Tensor, alvos: tensor_defs.Tensor, var perda_id: Int) -> Float32:
    if perda_id == tipos_mlp.perda_cross_entropy_id():
        return _loss_cross_entropy_media(pred, alvos)
    return perdas_mse.mse(pred, alvos)


fn construir_contexto(
    entradas: tensor_defs.Tensor,
    alvos: tensor_defs.Tensor,
    pesos: List[tensor_defs.Tensor],
    biases: List[tensor_defs.Tensor],
    var ativacao_saida_id: Int = -1,
    var perda_id: Int = -1,
) -> MLPForwardContext:
    debug_assert(len(pesos) > 0, "MLP precisa de ao menos uma camada")
    debug_assert(len(pesos) == len(biases), "pesos e biases precisam ter mesmo tamanho")

    var ops = List[String]()

    var zs = List[tensor_defs.Tensor]()
    var ativs = List[tensor_defs.Tensor]()
    ativs.append(entradas.copy())

    var atual = entradas.copy()
    var num_camadas = len(pesos)
    var num_saidas = pesos[num_camadas - 1].formato[1]

    if not tipos_mlp.ativacao_saida_id_valido(ativacao_saida_id):
        ativacao_saida_id = tipos_mlp.ativacao_saida_padrao_id(num_saidas)
    if not tipos_mlp.perda_id_valido(perda_id):
        perda_id = tipos_mlp.perda_padrao_id(num_saidas)

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
            if ativacao_saida_id == tipos_mlp.ativacao_saida_softmax_id():
                atual = _softmax_linhas(z)
                ops.append("softmax_out")
            elif ativacao_saida_id == tipos_mlp.ativacao_saida_linear_id():
                atual = z.copy()
                ops.append("linear_out")
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
    return MLPForwardContext(entradas, alvos, zs^, ativs^, pred, ativacao_saida_id, perda_id, ops^, g)


fn calcular_gradientes(ctx: MLPForwardContext, pesos: List[tensor_defs.Tensor]) -> MLPGradientes:
    var num_camadas = len(pesos)
    debug_assert(num_camadas > 0, "MLP precisa de ao menos uma camada")
    debug_assert(num_camadas == len(ctx.zs), "contexto inconsistente com número de camadas")

    var loss: Float32 = 0.0
    _ = loss
    var grad_z_atual = ctx.pred.copy()
    _ = grad_z_atual
    if ctx.perda_id == tipos_mlp.perda_cross_entropy_id():
        debug_assert(ctx.ativacao_saida_id == tipos_mlp.ativacao_saida_softmax_id(), "cross-entropy requer ativação de saída softmax")
        loss = _loss_cross_entropy_media(ctx.pred, ctx.alvos)
        grad_z_atual = _grad_softmax_cross_entropy(ctx.pred, ctx.alvos)
    else:
        loss = perdas_mse.mse(ctx.pred, ctx.alvos)
        var grad_pred = perdas_mse.gradiente_mse(ctx.pred, ctx.alvos)
        grad_z_atual = grad_pred.copy()
        if ctx.ativacao_saida_id == tipos_mlp.ativacao_saida_hard_sigmoid_id():
            var z_saida = ctx.zs[num_camadas - 1].copy()
            grad_z_atual = ativacoes.derivada_hard_sigmoid(z_saida, grad_pred)

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
