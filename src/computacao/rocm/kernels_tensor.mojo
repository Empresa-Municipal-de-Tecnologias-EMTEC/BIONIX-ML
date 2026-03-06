import src.nucleo.Tensor as tensor_defs


# Contratos/stubs para kernels ROCm de Tensor.
# Implementação real (HIP kernels/streams) pendente.

fn pipeline_id_rocm(var pipeline_memoria_id: Int, var operacao_id: Int) -> Int:
    return pipeline_memoria_id * 1000 + operacao_id


fn somar_elemento_a_elemento_rocm(a: tensor_defs.Tensor, b: tensor_defs.Tensor, var pipeline_id: Int) -> tensor_defs.Tensor:
    raise Exception("ROCm somar_elemento_a_elemento não implementado. pipeline_id=" + String(pipeline_id))


fn subtrair_elemento_a_elemento_rocm(a: tensor_defs.Tensor, b: tensor_defs.Tensor, var pipeline_id: Int) -> tensor_defs.Tensor:
    raise Exception("ROCm subtrair_elemento_a_elemento não implementado. pipeline_id=" + String(pipeline_id))


fn multiplicar_elemento_a_elemento_rocm(a: tensor_defs.Tensor, b: tensor_defs.Tensor, var pipeline_id: Int) -> tensor_defs.Tensor:
    raise Exception("ROCm multiplicar_elemento_a_elemento não implementado. pipeline_id=" + String(pipeline_id))


fn transpor_rocm(a: tensor_defs.Tensor, var pipeline_id: Int) -> tensor_defs.Tensor:
    raise Exception("ROCm transpor não implementado. pipeline_id=" + String(pipeline_id))


fn multiplicar_matrizes_rocm(a: tensor_defs.Tensor, b: tensor_defs.Tensor, var pipeline_id: Int) -> tensor_defs.Tensor:
    raise Exception("ROCm multiplicar_matrizes não implementado. pipeline_id=" + String(pipeline_id))


fn adicionar_bias_coluna_rocm(a: tensor_defs.Tensor, b: tensor_defs.Tensor, var pipeline_id: Int) -> tensor_defs.Tensor:
    raise Exception("ROCm adicionar_bias_coluna não implementado. pipeline_id=" + String(pipeline_id))


fn gradiente_mse_rocm(pred: tensor_defs.Tensor, alvo: tensor_defs.Tensor, var pipeline_id: Int) -> tensor_defs.Tensor:
    raise Exception("ROCm gradiente_mse não implementado. pipeline_id=" + String(pipeline_id))
