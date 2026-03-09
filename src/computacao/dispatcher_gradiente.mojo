import src.computacao.tipos as tipos
import src.computacao.cpu.kernels_gradiente as kernels_cpu
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
    pesos: List[tensor_defs.Tensor],
    var manter_gradientes_na_ram_principal: Bool = True,
) -> autograd_mlp.MLPGradientes:
    var backend = tipos.backend_cpu_id() if manter_gradientes_na_ram_principal else _backend_execucao_efetivo(ctx.entradas.id_backend)
    debug_assert(
        backend == tipos.backend_cpu_id(),
        "dispatcher_gradiente: backend não implementado sem fallback automático: " + tipos.backend_nome_de_id(backend),
    )
    return kernels_cpu.calcular_gradientes_mlp_cpu(ctx, pesos)
