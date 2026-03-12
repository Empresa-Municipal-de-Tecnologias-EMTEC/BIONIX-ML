import src.computacao.tipos as tipos
import src.computacao.cpu.kernels_ativacoes as kernels_cpu
import src.computacao.vulkan.kernels_ativacoes as kernels_vulkan
import src.computacao.rocm.kernels_ativacoes as kernels_rocm
import src.computacao.cuda.kernels_ativacoes as kernels_cuda
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


fn identidade(x: tensor_defs.Tensor) -> tensor_defs.Tensor:
    var backend = _backend_execucao_efetivo(x.id_backend)
    if backend == tipos.backend_cpu_id():
        return kernels_cpu.identidade_cpu(x)
    if backend == tipos.backend_vulkan_id():
        return kernels_vulkan.identidade_vulkan(x)
    if backend == tipos.backend_rocm_id():
        return kernels_rocm.identidade_rocm(x)
    if backend == tipos.backend_cuda_id():
        return kernels_cuda.identidade_cuda(x)
    raise Exception("dispatcher_ativacoes.identidade: backend inválido")


fn derivada_identidade(entrada: tensor_defs.Tensor, grad_saida: tensor_defs.Tensor) -> tensor_defs.Tensor:
    var backend = _backend_execucao_efetivo(entrada.id_backend)
    if backend == tipos.backend_cpu_id():
        return kernels_cpu.derivada_identidade_cpu(entrada, grad_saida)
    if backend == tipos.backend_vulkan_id():
        return kernels_vulkan.derivada_identidade_vulkan(entrada, grad_saida)
    if backend == tipos.backend_rocm_id():
        return kernels_rocm.derivada_identidade_rocm(entrada, grad_saida)
    if backend == tipos.backend_cuda_id():
        return kernels_cuda.derivada_identidade_cuda(entrada, grad_saida)
    raise Exception("dispatcher_ativacoes.derivada_identidade: backend inválido")


fn relu(x: tensor_defs.Tensor) -> tensor_defs.Tensor:
    var backend = _backend_execucao_efetivo(x.id_backend)
    if backend == tipos.backend_cpu_id():
        return kernels_cpu.relu_cpu(x)
    if backend == tipos.backend_vulkan_id():
        return kernels_vulkan.relu_vulkan(x)
    if backend == tipos.backend_rocm_id():
        return kernels_rocm.relu_rocm(x)
    if backend == tipos.backend_cuda_id():
        return kernels_cuda.relu_cuda(x)
    raise Exception("dispatcher_ativacoes.relu: backend inválido")


fn derivada_relu(entrada: tensor_defs.Tensor, grad_saida: tensor_defs.Tensor) -> tensor_defs.Tensor:
    var backend = _backend_execucao_efetivo(entrada.id_backend)
    if backend == tipos.backend_cpu_id():
        return kernels_cpu.derivada_relu_cpu(entrada, grad_saida)
    if backend == tipos.backend_vulkan_id():
        return kernels_vulkan.derivada_relu_vulkan(entrada, grad_saida)
    if backend == tipos.backend_rocm_id():
        return kernels_rocm.derivada_relu_rocm(entrada, grad_saida)
    if backend == tipos.backend_cuda_id():
        return kernels_cuda.derivada_relu_cuda(entrada, grad_saida)
    raise Exception("dispatcher_ativacoes.derivada_relu: backend inválido")


fn hard_sigmoid(x: tensor_defs.Tensor) -> tensor_defs.Tensor:
    var backend = _backend_execucao_efetivo(x.id_backend)
    if backend == tipos.backend_cpu_id():
        return kernels_cpu.hard_sigmoid_cpu(x)
    if backend == tipos.backend_vulkan_id():
        return kernels_vulkan.hard_sigmoid_vulkan(x)
    if backend == tipos.backend_rocm_id():
        return kernels_rocm.hard_sigmoid_rocm(x)
    if backend == tipos.backend_cuda_id():
        return kernels_cuda.hard_sigmoid_cuda(x)
    raise Exception("dispatcher_ativacoes.hard_sigmoid: backend inválido")


fn derivada_hard_sigmoid(entrada: tensor_defs.Tensor, grad_saida: tensor_defs.Tensor) -> tensor_defs.Tensor:
    var backend = _backend_execucao_efetivo(entrada.id_backend)
    if backend == tipos.backend_cpu_id():
        return kernels_cpu.derivada_hard_sigmoid_cpu(entrada, grad_saida)
    if backend == tipos.backend_vulkan_id():
        return kernels_vulkan.derivada_hard_sigmoid_vulkan(entrada, grad_saida)
    if backend == tipos.backend_rocm_id():
        return kernels_rocm.derivada_hard_sigmoid_rocm(entrada, grad_saida)
    if backend == tipos.backend_cuda_id():
        return kernels_cuda.derivada_hard_sigmoid_cuda(entrada, grad_saida)
    raise Exception("dispatcher_ativacoes.derivada_hard_sigmoid: backend inválido")
