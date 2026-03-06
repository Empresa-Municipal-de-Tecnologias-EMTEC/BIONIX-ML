import src.computacao.tipos as tipos
import src.computacao.cpu.kernels_tensor as kernels_cpu
import src.nucleo.Tensor as tensor_defs


fn _backend_execucao_efetivo(var backend_id: Int) -> Int:
    # CPU real; Vulkan/ROCm ainda não implementados para compute.
    # Mantemos fallback em CPU até os kernels específicos ficarem prontos.
    if backend_id == tipos.backend_cpu_id():
        return backend_id
    if backend_id == tipos.backend_vulkan_id():
        return tipos.backend_cpu_id()
    if backend_id == tipos.backend_rocm_id():
        return tipos.backend_cpu_id()
    return tipos.backend_cpu_id()


fn somar_elemento_a_elemento(a: tensor_defs.Tensor, b: tensor_defs.Tensor) -> tensor_defs.Tensor:
    var backend = _backend_execucao_efetivo(a.id_backend)
    if backend == tipos.backend_cpu_id():
        return kernels_cpu.somar_elemento_a_elemento_cpu(a, b)
    return kernels_cpu.somar_elemento_a_elemento_cpu(a, b)


fn subtrair_elemento_a_elemento(a: tensor_defs.Tensor, b: tensor_defs.Tensor) -> tensor_defs.Tensor:
    var backend = _backend_execucao_efetivo(a.id_backend)
    if backend == tipos.backend_cpu_id():
        return kernels_cpu.subtrair_elemento_a_elemento_cpu(a, b)
    return kernels_cpu.subtrair_elemento_a_elemento_cpu(a, b)


fn multiplicar_elemento_a_elemento(a: tensor_defs.Tensor, b: tensor_defs.Tensor) -> tensor_defs.Tensor:
    var backend = _backend_execucao_efetivo(a.id_backend)
    if backend == tipos.backend_cpu_id():
        return kernels_cpu.multiplicar_elemento_a_elemento_cpu(a, b)
    return kernels_cpu.multiplicar_elemento_a_elemento_cpu(a, b)


fn transpor(a: tensor_defs.Tensor) -> tensor_defs.Tensor:
    var backend = _backend_execucao_efetivo(a.id_backend)
    if backend == tipos.backend_cpu_id():
        return kernels_cpu.transpor_cpu(a)
    return kernels_cpu.transpor_cpu(a)


fn multiplicar_matrizes(a: tensor_defs.Tensor, b: tensor_defs.Tensor) -> tensor_defs.Tensor:
    var backend = _backend_execucao_efetivo(a.id_backend)
    if backend == tipos.backend_cpu_id():
        return kernels_cpu.multiplicar_matrizes_cpu(a, b)
    return kernels_cpu.multiplicar_matrizes_cpu(a, b)


fn adicionar_bias_coluna(a: tensor_defs.Tensor, b: tensor_defs.Tensor) -> tensor_defs.Tensor:
    var backend = _backend_execucao_efetivo(a.id_backend)
    if backend == tipos.backend_cpu_id():
        return kernels_cpu.adicionar_bias_coluna_cpu(a, b)
    return kernels_cpu.adicionar_bias_coluna_cpu(a, b)


fn soma_total(a: tensor_defs.Tensor) -> Float32:
    var backend = _backend_execucao_efetivo(a.id_backend)
    if backend == tipos.backend_cpu_id():
        return kernels_cpu.soma_total_cpu(a)
    return kernels_cpu.soma_total_cpu(a)


fn erro_quadratico_medio_escalar(pred: tensor_defs.Tensor, alvo: tensor_defs.Tensor) -> Float32:
    var backend = _backend_execucao_efetivo(pred.id_backend)
    if backend == tipos.backend_cpu_id():
        return kernels_cpu.erro_quadratico_medio_escalar_cpu(pred, alvo)
    return kernels_cpu.erro_quadratico_medio_escalar_cpu(pred, alvo)


fn gradiente_mse(pred: tensor_defs.Tensor, alvo: tensor_defs.Tensor) -> tensor_defs.Tensor:
    var backend = _backend_execucao_efetivo(pred.id_backend)
    if backend == tipos.backend_cpu_id():
        return kernels_cpu.gradiente_mse_cpu(pred, alvo)
    return kernels_cpu.gradiente_mse_cpu(pred, alvo)
