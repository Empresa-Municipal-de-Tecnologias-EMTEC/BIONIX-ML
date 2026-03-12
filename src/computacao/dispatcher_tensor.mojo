import src.computacao.tipos as tipos
import src.computacao.cpu.kernels_tensor as kernels_cpu
import src.computacao.vulkan.kernels_tensor as kernels_vulkan
import src.computacao.rocm.kernels_tensor as kernels_rocm
import src.computacao.cuda.kernels_tensor as kernels_cuda
import src.nucleo.Tensor as tensor_defs


fn _backend_execucao_efetivo(var backend_id: Int) -> Int:
    if backend_id == tipos.backend_cpu_id():
        return backend_id
    if backend_id == tipos.backend_vulkan_id():
        return backend_id
    if backend_id == tipos.backend_rocm_id():
        return backend_id
    if backend_id == tipos.backend_cuda_id():
        return backend_id
    return -1


fn somar_elemento_a_elemento(a: tensor_defs.Tensor, b: tensor_defs.Tensor) -> tensor_defs.Tensor:
    var backend = _backend_execucao_efetivo(a.id_backend)
    if backend == tipos.backend_cpu_id():
        return kernels_cpu.somar_elemento_a_elemento_cpu(a, b)
    if backend == tipos.backend_vulkan_id():
        var pipeline_id = kernels_vulkan.pipeline_id_vulkan(a.id_pipeline_memoria, 1)
        return kernels_vulkan.somar_elemento_a_elemento_vulkan(a, b, pipeline_id)
    if backend == tipos.backend_rocm_id():
        var pipeline_id = kernels_rocm.pipeline_id_rocm(a.id_pipeline_memoria, 1)
        return kernels_rocm.somar_elemento_a_elemento_rocm(a, b, pipeline_id)
    if backend == tipos.backend_cuda_id():
        var pipeline_id = kernels_cuda.pipeline_id_cuda(a.id_pipeline_memoria, 1)
        return kernels_cuda.somar_elemento_a_elemento_cuda(a, b, pipeline_id)
    raise Exception("dispatcher_tensor.somar_elemento_a_elemento: backend inválido")


fn subtrair_elemento_a_elemento(a: tensor_defs.Tensor, b: tensor_defs.Tensor) -> tensor_defs.Tensor:
    var backend = _backend_execucao_efetivo(a.id_backend)
    if backend == tipos.backend_cpu_id():
        return kernels_cpu.subtrair_elemento_a_elemento_cpu(a, b)
    if backend == tipos.backend_vulkan_id():
        var pipeline_id = kernels_vulkan.pipeline_id_vulkan(a.id_pipeline_memoria, 2)
        return kernels_vulkan.subtrair_elemento_a_elemento_vulkan(a, b, pipeline_id)
    if backend == tipos.backend_rocm_id():
        var pipeline_id = kernels_rocm.pipeline_id_rocm(a.id_pipeline_memoria, 2)
        return kernels_rocm.subtrair_elemento_a_elemento_rocm(a, b, pipeline_id)
    if backend == tipos.backend_cuda_id():
        var pipeline_id = kernels_cuda.pipeline_id_cuda(a.id_pipeline_memoria, 2)
        return kernels_cuda.subtrair_elemento_a_elemento_cuda(a, b, pipeline_id)
    raise Exception("dispatcher_tensor.subtrair_elemento_a_elemento: backend inválido")


fn multiplicar_elemento_a_elemento(a: tensor_defs.Tensor, b: tensor_defs.Tensor) -> tensor_defs.Tensor:
    var backend = _backend_execucao_efetivo(a.id_backend)
    if backend == tipos.backend_cpu_id():
        return kernels_cpu.multiplicar_elemento_a_elemento_cpu(a, b)
    if backend == tipos.backend_vulkan_id():
        var pipeline_id = kernels_vulkan.pipeline_id_vulkan(a.id_pipeline_memoria, 3)
        return kernels_vulkan.multiplicar_elemento_a_elemento_vulkan(a, b, pipeline_id)
    if backend == tipos.backend_rocm_id():
        var pipeline_id = kernels_rocm.pipeline_id_rocm(a.id_pipeline_memoria, 3)
        return kernels_rocm.multiplicar_elemento_a_elemento_rocm(a, b, pipeline_id)
    if backend == tipos.backend_cuda_id():
        var pipeline_id = kernels_cuda.pipeline_id_cuda(a.id_pipeline_memoria, 3)
        return kernels_cuda.multiplicar_elemento_a_elemento_cuda(a, b, pipeline_id)
    raise Exception("dispatcher_tensor.multiplicar_elemento_a_elemento: backend inválido")


fn transpor(a: tensor_defs.Tensor) -> tensor_defs.Tensor:
    var backend = _backend_execucao_efetivo(a.id_backend)
    if backend == tipos.backend_cpu_id():
        return kernels_cpu.transpor_cpu(a)
    if backend == tipos.backend_vulkan_id():
        var pipeline_id = kernels_vulkan.pipeline_id_vulkan(a.id_pipeline_memoria, 4)
        return kernels_vulkan.transpor_vulkan(a, pipeline_id)
    if backend == tipos.backend_rocm_id():
        var pipeline_id = kernels_rocm.pipeline_id_rocm(a.id_pipeline_memoria, 4)
        return kernels_rocm.transpor_rocm(a, pipeline_id)
    if backend == tipos.backend_cuda_id():
        var pipeline_id = kernels_cuda.pipeline_id_cuda(a.id_pipeline_memoria, 4)
        return kernels_cuda.transpor_cuda(a, pipeline_id)
    raise Exception("dispatcher_tensor.transpor: backend inválido")


fn multiplicar_matrizes(a: tensor_defs.Tensor, b: tensor_defs.Tensor) -> tensor_defs.Tensor:
    var backend = _backend_execucao_efetivo(a.id_backend)
    if backend == tipos.backend_cpu_id():
        return kernels_cpu.multiplicar_matrizes_cpu(a, b)
    if backend == tipos.backend_vulkan_id():
        var pipeline_id = kernels_vulkan.pipeline_id_vulkan(a.id_pipeline_memoria, 5)
        return kernels_vulkan.multiplicar_matrizes_vulkan(a, b, pipeline_id)
    if backend == tipos.backend_rocm_id():
        var pipeline_id = kernels_rocm.pipeline_id_rocm(a.id_pipeline_memoria, 5)
        return kernels_rocm.multiplicar_matrizes_rocm(a, b, pipeline_id)
    if backend == tipos.backend_cuda_id():
        var pipeline_id = kernels_cuda.pipeline_id_cuda(a.id_pipeline_memoria, 5)
        return kernels_cuda.multiplicar_matrizes_cuda(a, b, pipeline_id)
    raise Exception("dispatcher_tensor.multiplicar_matrizes: backend inválido")


fn adicionar_bias_coluna(a: tensor_defs.Tensor, b: tensor_defs.Tensor) -> tensor_defs.Tensor:
    var backend = _backend_execucao_efetivo(a.id_backend)
    if backend == tipos.backend_cpu_id():
        return kernels_cpu.adicionar_bias_coluna_cpu(a, b)
    if backend == tipos.backend_vulkan_id():
        var pipeline_id = kernels_vulkan.pipeline_id_vulkan(a.id_pipeline_memoria, 6)
        return kernels_vulkan.adicionar_bias_coluna_vulkan(a, b, pipeline_id)
    if backend == tipos.backend_rocm_id():
        var pipeline_id = kernels_rocm.pipeline_id_rocm(a.id_pipeline_memoria, 6)
        return kernels_rocm.adicionar_bias_coluna_rocm(a, b, pipeline_id)
    if backend == tipos.backend_cuda_id():
        var pipeline_id = kernels_cuda.pipeline_id_cuda(a.id_pipeline_memoria, 6)
        return kernels_cuda.adicionar_bias_coluna_cuda(a, b, pipeline_id)
    raise Exception("dispatcher_tensor.adicionar_bias_coluna: backend inválido")


fn soma_total(a: tensor_defs.Tensor) -> Float32:
    var s: Float32 = 0.0
    for i in range(len(a.dados)):
        s = s + a.dados[i]
    return s


fn erro_quadratico_medio_escalar(pred: tensor_defs.Tensor, alvo: tensor_defs.Tensor) -> Float32:
    debug_assert(len(pred.dados) == len(alvo.dados), "pred e alvo devem ter mesmo tamanho")
    if len(pred.dados) == 0:
        return 0.0
    var soma: Float32 = 0.0
    for i in range(len(pred.dados)):
        var d = pred.dados[i] - alvo.dados[i]
        soma = soma + d * d
    return soma / Float32(len(pred.dados))


fn gradiente_mse(pred: tensor_defs.Tensor, alvo: tensor_defs.Tensor) -> tensor_defs.Tensor:
    var backend = _backend_execucao_efetivo(pred.id_backend)
    if backend == tipos.backend_cpu_id():
        return kernels_cpu.gradiente_mse_cpu(pred, alvo)
    if backend == tipos.backend_vulkan_id():
        var pipeline_id = kernels_vulkan.pipeline_id_vulkan(pred.id_pipeline_memoria, 7)
        return kernels_vulkan.gradiente_mse_vulkan(pred, alvo, pipeline_id)
    if backend == tipos.backend_rocm_id():
        var pipeline_id = kernels_rocm.pipeline_id_rocm(pred.id_pipeline_memoria, 7)
        return kernels_rocm.gradiente_mse_rocm(pred, alvo, pipeline_id)
    if backend == tipos.backend_cuda_id():
        var pipeline_id = kernels_cuda.pipeline_id_cuda(pred.id_pipeline_memoria, 7)
        return kernels_cuda.gradiente_mse_cuda(pred, alvo, pipeline_id)
    raise Exception("dispatcher_tensor.gradiente_mse: backend inválido")
