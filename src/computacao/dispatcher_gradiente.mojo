import src.computacao.tipos as tipos
import src.computacao.cpu.kernels_gradiente as kernels_cpu
import src.computacao.cuda.kernels_gradiente as kernels_cuda
import src.computacao.rocm.kernels_gradiente as kernels_rocm
import src.computacao.vulkan.kernels_gradiente as kernels_vulkan
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
    return tipos.backend_cpu_id()


fn calcular_gradientes_mlp(
    ctx: autograd_mlp.MLPForwardContext,
    w2: tensor_defs.Tensor,
    var manter_gradientes_na_ram_principal: Bool = True,
) -> autograd_mlp.MLPGradientes:
    var backend = tipos.backend_cpu_id() if manter_gradientes_na_ram_principal else _backend_execucao_efetivo(ctx.entradas.id_backend)
    var pipeline_id = ctx.entradas.id_pipeline_memoria * 1000 + 100
    if backend == tipos.backend_cpu_id():
        return kernels_cpu.calcular_gradientes_mlp_cpu(ctx, w2)
    if backend == tipos.backend_cuda_id():
        return kernels_cuda.calcular_gradientes_mlp_cuda(ctx, w2, pipeline_id)
    if backend == tipos.backend_rocm_id():
        return kernels_rocm.calcular_gradientes_mlp_rocm(ctx, w2, pipeline_id)
    if backend == tipos.backend_vulkan_id():
        return kernels_vulkan.calcular_gradientes_mlp_vulkan(ctx, w2, pipeline_id)
    return kernels_cpu.calcular_gradientes_mlp_cpu(ctx, w2)
