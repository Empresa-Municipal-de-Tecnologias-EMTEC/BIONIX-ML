import src.nucleo.Tensor as tensor_defs


# Contratos/stubs para kernels Vulkan de Tensor.
# Implementação real (SPIR-V, pipeline cache, descriptor sets) pendente.

fn pipeline_id_vulkan(var pipeline_memoria_id: Int, var operacao_id: Int) -> Int:
    return pipeline_memoria_id * 1000 + operacao_id


fn somar_elemento_a_elemento_vulkan(a: tensor_defs.Tensor, b: tensor_defs.Tensor, var pipeline_id: Int) -> tensor_defs.Tensor:
    raise Exception("Vulkan somar_elemento_a_elemento não implementado. pipeline_id=" + String(pipeline_id))


fn subtrair_elemento_a_elemento_vulkan(a: tensor_defs.Tensor, b: tensor_defs.Tensor, var pipeline_id: Int) -> tensor_defs.Tensor:
    raise Exception("Vulkan subtrair_elemento_a_elemento não implementado. pipeline_id=" + String(pipeline_id))


fn multiplicar_elemento_a_elemento_vulkan(a: tensor_defs.Tensor, b: tensor_defs.Tensor, var pipeline_id: Int) -> tensor_defs.Tensor:
    raise Exception("Vulkan multiplicar_elemento_a_elemento não implementado. pipeline_id=" + String(pipeline_id))


fn transpor_vulkan(a: tensor_defs.Tensor, var pipeline_id: Int) -> tensor_defs.Tensor:
    raise Exception("Vulkan transpor não implementado. pipeline_id=" + String(pipeline_id))


fn multiplicar_matrizes_vulkan(a: tensor_defs.Tensor, b: tensor_defs.Tensor, var pipeline_id: Int) -> tensor_defs.Tensor:
    raise Exception("Vulkan multiplicar_matrizes não implementado. pipeline_id=" + String(pipeline_id))


fn adicionar_bias_coluna_vulkan(a: tensor_defs.Tensor, b: tensor_defs.Tensor, var pipeline_id: Int) -> tensor_defs.Tensor:
    raise Exception("Vulkan adicionar_bias_coluna não implementado. pipeline_id=" + String(pipeline_id))


fn gradiente_mse_vulkan(pred: tensor_defs.Tensor, alvo: tensor_defs.Tensor, var pipeline_id: Int) -> tensor_defs.Tensor:
    raise Exception("Vulkan gradiente_mse não implementado. pipeline_id=" + String(pipeline_id))
