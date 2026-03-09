import src.autograd.mlp as autograd_mlp
import src.nucleo.Tensor as tensor_defs

fn pipeline_id_cuda(var pipeline_memoria_id: Int, var operacao_id: Int) -> Int:
    return pipeline_memoria_id * 1000 + operacao_id


fn calcular_gradientes_mlp_cuda(
    ctx: autograd_mlp.MLPForwardContext,
    w2: tensor_defs.Tensor,
    var pipeline_id: Int,
) raises -> autograd_mlp.MLPGradientes:
    _ = ctx
    _ = w2
    raise Exception("CUDA calcular_gradientes_mlp não implementado. pipeline_id=" + String(pipeline_id))
