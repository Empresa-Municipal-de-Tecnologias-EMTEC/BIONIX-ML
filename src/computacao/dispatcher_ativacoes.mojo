import src.computacao.tipos as tipos
import src.computacao.cpu.kernels_ativacoes as kernels_cpu
import src.nucleo.Tensor as tensor_defs


fn _backend_execucao_efetivo(var backend_id: Int) -> Int:
    if backend_id == tipos.backend_cpu_id():
        return backend_id
    if backend_id == tipos.backend_vulkan_id():
        return tipos.backend_cpu_id()
    if backend_id == tipos.backend_rocm_id():
        return tipos.backend_cpu_id()
    if backend_id == tipos.backend_cuda_id():
        return tipos.backend_cpu_id()
    return tipos.backend_cpu_id()


fn identidade(x: tensor_defs.Tensor) -> tensor_defs.Tensor:
    var backend = _backend_execucao_efetivo(x.id_backend)
    if backend == tipos.backend_cpu_id():
        return kernels_cpu.identidade_cpu(x)
    return kernels_cpu.identidade_cpu(x)


fn derivada_identidade(entrada: tensor_defs.Tensor, grad_saida: tensor_defs.Tensor) -> tensor_defs.Tensor:
    var backend = _backend_execucao_efetivo(entrada.id_backend)
    if backend == tipos.backend_cpu_id():
        return kernels_cpu.derivada_identidade_cpu(entrada, grad_saida)
    return kernels_cpu.derivada_identidade_cpu(entrada, grad_saida)


fn relu(x: tensor_defs.Tensor) -> tensor_defs.Tensor:
    var backend = _backend_execucao_efetivo(x.id_backend)
    if backend == tipos.backend_cpu_id():
        return kernels_cpu.relu_cpu(x)
    return kernels_cpu.relu_cpu(x)


fn derivada_relu(entrada: tensor_defs.Tensor, grad_saida: tensor_defs.Tensor) -> tensor_defs.Tensor:
    var backend = _backend_execucao_efetivo(entrada.id_backend)
    if backend == tipos.backend_cpu_id():
        return kernels_cpu.derivada_relu_cpu(entrada, grad_saida)
    return kernels_cpu.derivada_relu_cpu(entrada, grad_saida)


fn hard_sigmoid(x: tensor_defs.Tensor) -> tensor_defs.Tensor:
    var backend = _backend_execucao_efetivo(x.id_backend)
    if backend == tipos.backend_cpu_id():
        return kernels_cpu.hard_sigmoid_cpu(x)
    return kernels_cpu.hard_sigmoid_cpu(x)


fn derivada_hard_sigmoid(entrada: tensor_defs.Tensor, grad_saida: tensor_defs.Tensor) -> tensor_defs.Tensor:
    var backend = _backend_execucao_efetivo(entrada.id_backend)
    if backend == tipos.backend_cpu_id():
        return kernels_cpu.derivada_hard_sigmoid_cpu(entrada, grad_saida)
    return kernels_cpu.derivada_hard_sigmoid_cpu(entrada, grad_saida)
