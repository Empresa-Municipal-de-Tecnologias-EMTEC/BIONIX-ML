import src.autograd.mlp as autograd_mlp
import src.nucleo.Tensor as tensor_defs

fn pipeline_id_cuda(var pipeline_memoria_id: Int, var operacao_id: Int) -> Int:
    return pipeline_memoria_id * 1000 + operacao_id


fn calcular_gradientes_mlp_cuda(
    ctx: autograd_mlp.MLPForwardContext,
    pesos: List[tensor_defs.Tensor],
    var pipeline_id: Int,
) -> autograd_mlp.MLPGradientes:
    var grads = autograd_mlp.calcular_gradientes(ctx, pesos)
    for i in range(len(grads.grad_ws)):
        grads.grad_ws[i].id_pipeline_ultima_operacao = pipeline_id
    for i in range(len(grads.grad_bs)):
        grads.grad_bs[i].id_pipeline_ultima_operacao = pipeline_id
    return grads^
