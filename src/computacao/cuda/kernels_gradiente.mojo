import src.autograd.mlp as autograd_mlp
import src.autograd.tipos_mlp as tipos_mlp
import src.computacao.cuda.kernels_tensor as kernels_cuda_tensor
import src.perdas.mse as perdas_mse
import src.nucleo.Tensor as tensor_defs
import math


struct CUDAGradienteWorkspace(Movable, Copyable):
    var assinatura: String
    var grad_ws_por_camada: List[tensor_defs.Tensor]
    var grad_bs_por_camada: List[tensor_defs.Tensor]
    var grad_a_prev_por_camada: List[tensor_defs.Tensor]

    fn __init__(out self):
        self.assinatura = ""
        self.grad_ws_por_camada = List[tensor_defs.Tensor]()
        self.grad_bs_por_camada = List[tensor_defs.Tensor]()
        self.grad_a_prev_por_camada = List[tensor_defs.Tensor]()

fn pipeline_id_cuda(var pipeline_memoria_id: Int, var operacao_id: Int) -> Int:
    return pipeline_memoria_id * 1000 + operacao_id


fn criar_workspace_gradiente_cuda() -> CUDAGradienteWorkspace:
    return CUDAGradienteWorkspace()


fn _assinatura_workspace(ctx: autograd_mlp.MLPForwardContext, pesos: List[tensor_defs.Tensor]) -> String:
    var assinatura = "b=" + String(ctx.entradas.formato[0]) + "|c=" + String(len(pesos))
    for camada in range(len(pesos)):
        assinatura = assinatura + "|w" + String(camada) + "=" + String(pesos[camada].formato[0]) + "x" + String(pesos[camada].formato[1])
    return assinatura


fn _reconstruir_workspace_cuda(
    mut workspace: CUDAGradienteWorkspace,
    ctx: autograd_mlp.MLPForwardContext,
    pesos: List[tensor_defs.Tensor],
):
    workspace.grad_ws_por_camada = List[tensor_defs.Tensor]()
    workspace.grad_bs_por_camada = List[tensor_defs.Tensor]()
    workspace.grad_a_prev_por_camada = List[tensor_defs.Tensor]()

    var num_camadas = len(pesos)
    for camada in range(num_camadas):
        var fan_in = pesos[camada].formato[0]
        var fan_out = pesos[camada].formato[1]

        var fw = List[Int]()
        fw.append(fan_in)
        fw.append(fan_out)
        workspace.grad_ws_por_camada.append(tensor_defs.Tensor(fw^, ctx.entradas.tipo_computacao))

        var fb = List[Int]()
        fb.append(1)
        fb.append(fan_out)
        workspace.grad_bs_por_camada.append(tensor_defs.Tensor(fb^, ctx.entradas.tipo_computacao))

        var fa = List[Int]()
        fa.append(ctx.entradas.formato[0])
        fa.append(fan_in)
        workspace.grad_a_prev_por_camada.append(tensor_defs.Tensor(fa^, ctx.entradas.tipo_computacao))


fn _garantir_workspace_cuda(
    mut workspace: CUDAGradienteWorkspace,
    ctx: autograd_mlp.MLPForwardContext,
    pesos: List[tensor_defs.Tensor],
):
    var assinatura = _assinatura_workspace(ctx, pesos)
    if assinatura == workspace.assinatura:
        return
    workspace.assinatura = assinatura
    _reconstruir_workspace_cuda(workspace, ctx, pesos)


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


fn calcular_gradientes_mlp_cuda(
    ctx: autograd_mlp.MLPForwardContext,
    pesos: List[tensor_defs.Tensor],
    var pipeline_id: Int,
) -> autograd_mlp.MLPGradientes:
    var workspace = criar_workspace_gradiente_cuda()
    return calcular_gradientes_mlp_cuda_com_workspace(ctx, pesos, pipeline_id, workspace)


fn calcular_gradientes_mlp_cuda_com_workspace(
    ctx: autograd_mlp.MLPForwardContext,
    pesos: List[tensor_defs.Tensor],
    var pipeline_id: Int,
    mut workspace: CUDAGradienteWorkspace,
) -> autograd_mlp.MLPGradientes:
    var num_camadas = len(pesos)
    debug_assert(num_camadas > 0, "MLP precisa de ao menos uma camada")
    debug_assert(num_camadas == len(ctx.zs), "contexto inconsistente com número de camadas")
    _garantir_workspace_cuda(workspace, ctx, pesos)

    var loss: Float32 = 0.0
    var grad_z_atual = ctx.pred.copy()

    if ctx.perda_id == tipos_mlp.perda_cross_entropy_id():
        debug_assert(ctx.ativacao_saida_id == tipos_mlp.ativacao_saida_softmax_id(), "cross-entropy requer ativação de saída softmax")
        loss = _loss_cross_entropy_media(ctx.pred, ctx.alvos)
        grad_z_atual = kernels_cuda_tensor.grad_softmax_cross_entropy_cuda(
            ctx.pred,
            ctx.alvos,
            pipeline_id_cuda(ctx.entradas.id_pipeline_memoria, 604),
        )
    else:
        loss = perdas_mse.mse(ctx.pred, ctx.alvos)
        var grad_pred = kernels_cuda_tensor.gradiente_mse_cuda(
            ctx.pred,
            ctx.alvos,
            pipeline_id_cuda(ctx.entradas.id_pipeline_memoria, 607),
        )
        grad_z_atual = grad_pred.copy()
        if ctx.ativacao_saida_id == tipos_mlp.ativacao_saida_hard_sigmoid_id():
            var z_saida = ctx.zs[num_camadas - 1].copy()
            # Mantém caminho já existente para hard-sigmoid de saída.
            # O caminho ReLU oculto foi fundido no passo_backprop_mlp_cuda.
            var grad_saida_hard = tensor_defs.Tensor(z_saida.formato.copy(), z_saida.tipo_computacao)
            for i in range(len(grad_saida_hard.dados)):
                var x = z_saida.dados[i]
                var d: Float32 = 0.0
                if x > -2.5 and x < 2.5:
                    d = 0.2
                grad_saida_hard.dados[i] = grad_pred.dados[i] * d
            grad_z_atual = grad_saida_hard^

    var ativacoes_forward = List[tensor_defs.Tensor]()
    for a in ctx.ativacoes:
        ativacoes_forward.append(a.copy())
    var zs_forward = List[tensor_defs.Tensor]()
    for z in ctx.zs:
        zs_forward.append(z.copy())
    var pesos_local = List[tensor_defs.Tensor]()
    for w in pesos:
        pesos_local.append(w.copy())

    kernels_cuda_tensor.backward_mlp_cuda_em_tensores(
        ativacoes_forward,
        zs_forward,
        pesos_local,
        grad_z_atual,
        workspace.grad_ws_por_camada,
        workspace.grad_bs_por_camada,
        workspace.grad_a_prev_por_camada,
        pipeline_id_cuda(ctx.entradas.id_pipeline_memoria, 615),
    )

    var grad_ws = List[tensor_defs.Tensor]()
    var grad_bs = List[tensor_defs.Tensor]()
    for i in range(num_camadas):
        var gw = workspace.grad_ws_por_camada[i].copy()
        var gb = workspace.grad_bs_por_camada[i].copy()
        gw.id_pipeline_ultima_operacao = pipeline_id
        gb.id_pipeline_ultima_operacao = pipeline_id
        grad_ws.append(gw.copy())
        grad_bs.append(gb.copy())

    return autograd_mlp.MLPGradientes(grad_ws^, grad_bs^, loss)
