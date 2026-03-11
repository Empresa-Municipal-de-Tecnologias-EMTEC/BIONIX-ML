import src.computacao.tipos as tipos
import src.computacao.cpu.kernels_gradiente as kernels_cpu
import src.computacao.vulkan.kernels_gradiente as kernels_vulkan
import src.computacao.rocm.kernels_gradiente as kernels_rocm
import src.computacao.cuda.kernels_gradiente as kernels_cuda
import src.autograd.mlp as autograd_mlp
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


fn calcular_gradientes_mlp(
    ctx: autograd_mlp.MLPForwardContext,
    pesos: List[tensor_defs.Tensor],
    var manter_gradientes_na_ram_principal: Bool = False,
) -> autograd_mlp.MLPGradientes:
    var backend = _backend_execucao_efetivo(ctx.entradas.id_backend)
    if manter_gradientes_na_ram_principal:
        backend = tipos.backend_cpu_id()
    if backend == tipos.backend_vulkan_id():
        var pipeline_id = ctx.entradas.id_pipeline_memoria * 1000 + 201
        return kernels_vulkan.calcular_gradientes_mlp_vulkan(ctx, pesos, pipeline_id)
    if backend == tipos.backend_rocm_id():
        var pipeline_id = ctx.entradas.id_pipeline_memoria * 1000 + 301
        return kernels_rocm.calcular_gradientes_mlp_rocm(ctx, pesos, pipeline_id)
    if backend == tipos.backend_cuda_id():
        var pipeline_id = ctx.entradas.id_pipeline_memoria * 1000 + 401
        return kernels_cuda.calcular_gradientes_mlp_cuda(ctx, pesos, pipeline_id)
    if backend == tipos.backend_cpu_id():
        return kernels_cpu.calcular_gradientes_mlp_cpu(ctx, pesos)
    debug_assert(False, "dispatcher_gradiente.calcular_gradientes_mlp: backend inválido")
    return kernels_cpu.calcular_gradientes_mlp_cpu(ctx, pesos)
