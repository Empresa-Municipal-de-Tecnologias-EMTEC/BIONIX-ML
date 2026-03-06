import src.nucleo.Tensor as tensor_defs


# Contratos/stubs para kernels CUDA de Tensor.
# Implementação real (kernel launch, stream, memory transfer) pendente.

fn pipeline_id_cuda(var pipeline_memoria_id: Int, var operacao_id: Int) -> Int:
    return pipeline_memoria_id * 1000 + operacao_id


fn somar_elemento_a_elemento_cuda(a: tensor_defs.Tensor, b: tensor_defs.Tensor, var pipeline_id: Int) -> tensor_defs.Tensor:
    raise Exception("CUDA somar_elemento_a_elemento não implementado. pipeline_id=" + String(pipeline_id))


fn subtrair_elemento_a_elemento_cuda(a: tensor_defs.Tensor, b: tensor_defs.Tensor, var pipeline_id: Int) -> tensor_defs.Tensor:
    raise Exception("CUDA subtrair_elemento_a_elemento não implementado. pipeline_id=" + String(pipeline_id))


fn multiplicar_elemento_a_elemento_cuda(a: tensor_defs.Tensor, b: tensor_defs.Tensor, var pipeline_id: Int) -> tensor_defs.Tensor:
    raise Exception("CUDA multiplicar_elemento_a_elemento não implementado. pipeline_id=" + String(pipeline_id))


fn transpor_cuda(a: tensor_defs.Tensor, var pipeline_id: Int) -> tensor_defs.Tensor:
    raise Exception("CUDA transpor não implementado. pipeline_id=" + String(pipeline_id))


fn multiplicar_matrizes_cuda(a: tensor_defs.Tensor, b: tensor_defs.Tensor, var pipeline_id: Int) -> tensor_defs.Tensor:
    raise Exception("CUDA multiplicar_matrizes não implementado. pipeline_id=" + String(pipeline_id))


fn adicionar_bias_coluna_cuda(a: tensor_defs.Tensor, b: tensor_defs.Tensor, var pipeline_id: Int) -> tensor_defs.Tensor:
    raise Exception("CUDA adicionar_bias_coluna não implementado. pipeline_id=" + String(pipeline_id))


fn gradiente_mse_cuda(pred: tensor_defs.Tensor, alvo: tensor_defs.Tensor, var pipeline_id: Int) -> tensor_defs.Tensor:
    raise Exception("CUDA gradiente_mse não implementado. pipeline_id=" + String(pipeline_id))
