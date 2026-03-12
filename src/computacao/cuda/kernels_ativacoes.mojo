import src.nucleo.Tensor as tensor_defs
from gpu import global_idx
from gpu.host import DeviceContext, DeviceBuffer
from sys import has_nvidia_gpu_accelerator


fn _kernel_relu(
    entrada: UnsafePointer[Float32],
    saida: UnsafePointer[Float32],
    len_total: Int,
):
    var tid = global_idx.x
    if tid >= UInt(len_total):
        return
    var v = entrada[tid]
    saida[tid] = v if v > 0.0 else Float32(0.0)


fn _kernel_hard_sigmoid(
    entrada: UnsafePointer[Float32],
    saida: UnsafePointer[Float32],
    len_total: Int,
):
    var tid = global_idx.x
    if tid >= UInt(len_total):
        return
    var v = 0.2 * entrada[tid] + 0.5
    if v < 0.0:
        v = 0.0
    if v > 1.0:
        v = 1.0
    saida[tid] = v


fn _kernel_derivada_relu(
    entrada: UnsafePointer[Float32],
    grad_saida: UnsafePointer[Float32],
    grad: UnsafePointer[Float32],
    len_total: Int,
):
    var tid = global_idx.x
    if tid >= UInt(len_total):
        return
    grad[tid] = grad_saida[tid] if entrada[tid] > 0.0 else Float32(0.0)


fn _kernel_derivada_hard_sigmoid(
    entrada: UnsafePointer[Float32],
    grad_saida: UnsafePointer[Float32],
    grad: UnsafePointer[Float32],
    len_total: Int,
):
    var tid = global_idx.x
    if tid >= UInt(len_total):
        return
    var x = entrada[tid]
    var d: Float32 = 0.0
    if x > -2.5 and x < 2.5:
        d = 0.2
    grad[tid] = grad_saida[tid] * d


fn _copiar_host_para_dev(mut dev: DeviceBuffer[DType.float32], dados: List[Float32], var len_total: Int) raises:
    with dev.map_to_host() as host:
        for i in range(len_total):
            host[i] = dados[i]


fn _copiar_dev_para_tensor(mut dev: DeviceBuffer[DType.float32], mut out: tensor_defs.Tensor, var len_total: Int) raises:
    with dev.map_to_host() as host:
        for i in range(len_total):
            out.dados[i] = host[i]


fn identidade_cuda(x: tensor_defs.Tensor) -> tensor_defs.Tensor:
    return x.copy()


fn derivada_identidade_cuda(entrada: tensor_defs.Tensor, grad_saida: tensor_defs.Tensor) -> tensor_defs.Tensor:
    debug_assert(len(entrada.dados) == len(grad_saida.dados), "gradiente incompatível")
    return grad_saida.copy()


fn relu_cuda(x: tensor_defs.Tensor) -> tensor_defs.Tensor:
    var formato = x.formato.copy()
    var out = tensor_defs.Tensor(formato^, x.tipo_computacao)
    var len_total = len(x.dados)
    @parameter
    if has_nvidia_gpu_accelerator():
        try:
            with DeviceContext(0, api="cuda") as ctx:
                var in_dev = ctx.enqueue_create_buffer[DType.float32](len_total)
                var out_dev = ctx.enqueue_create_buffer[DType.float32](len_total)
                _copiar_host_para_dev(in_dev, x.dados, len_total)
                var block_dim = 256
                var grid_dim = (len_total + block_dim - 1) // block_dim
                ctx.enqueue_function_experimental[_kernel_relu](
                    in_dev,
                    out_dev,
                    len_total,
                    grid_dim=(grid_dim),
                    block_dim=(block_dim),
                )
                ctx.synchronize()
                _copiar_dev_para_tensor(out_dev, out, len_total)
        except _:
            debug_assert(False, "falha ao executar kernel CUDA de relu")
    else:
        debug_assert(False, "kernels CUDA nao disponiveis nesta compilacao")
    return out^


fn derivada_relu_cuda(entrada: tensor_defs.Tensor, grad_saida: tensor_defs.Tensor) -> tensor_defs.Tensor:
    debug_assert(len(entrada.dados) == len(grad_saida.dados), "gradiente incompatível")
    var formato = entrada.formato.copy()
    var grad = tensor_defs.Tensor(formato^, entrada.tipo_computacao)
    var len_total = len(entrada.dados)
    @parameter
    if has_nvidia_gpu_accelerator():
        try:
            with DeviceContext(0, api="cuda") as ctx:
                var in_dev = ctx.enqueue_create_buffer[DType.float32](len_total)
                var grad_saida_dev = ctx.enqueue_create_buffer[DType.float32](len_total)
                var out_dev = ctx.enqueue_create_buffer[DType.float32](len_total)
                _copiar_host_para_dev(in_dev, entrada.dados, len_total)
                _copiar_host_para_dev(grad_saida_dev, grad_saida.dados, len_total)
                var block_dim = 256
                var grid_dim = (len_total + block_dim - 1) // block_dim
                ctx.enqueue_function_experimental[_kernel_derivada_relu](
                    in_dev,
                    grad_saida_dev,
                    out_dev,
                    len_total,
                    grid_dim=(grid_dim),
                    block_dim=(block_dim),
                )
                ctx.synchronize()
                _copiar_dev_para_tensor(out_dev, grad, len_total)
        except _:
            debug_assert(False, "falha ao executar kernel CUDA de derivada relu")
    else:
        debug_assert(False, "kernels CUDA nao disponiveis nesta compilacao")
    return grad^


fn hard_sigmoid_cuda(x: tensor_defs.Tensor) -> tensor_defs.Tensor:
    var formato = x.formato.copy()
    var out = tensor_defs.Tensor(formato^, x.tipo_computacao)
    var len_total = len(x.dados)
    @parameter
    if has_nvidia_gpu_accelerator():
        try:
            with DeviceContext(0, api="cuda") as ctx:
                var in_dev = ctx.enqueue_create_buffer[DType.float32](len_total)
                var out_dev = ctx.enqueue_create_buffer[DType.float32](len_total)
                _copiar_host_para_dev(in_dev, x.dados, len_total)
                var block_dim = 256
                var grid_dim = (len_total + block_dim - 1) // block_dim
                ctx.enqueue_function_experimental[_kernel_hard_sigmoid](
                    in_dev,
                    out_dev,
                    len_total,
                    grid_dim=(grid_dim),
                    block_dim=(block_dim),
                )
                ctx.synchronize()
                _copiar_dev_para_tensor(out_dev, out, len_total)
        except _:
            debug_assert(False, "falha ao executar kernel CUDA de hard_sigmoid")
    else:
        debug_assert(False, "kernels CUDA nao disponiveis nesta compilacao")
    return out^


fn derivada_hard_sigmoid_cuda(entrada: tensor_defs.Tensor, grad_saida: tensor_defs.Tensor) -> tensor_defs.Tensor:
    debug_assert(len(entrada.dados) == len(grad_saida.dados), "gradiente incompatível")
    var formato = entrada.formato.copy()
    var grad = tensor_defs.Tensor(formato^, entrada.tipo_computacao)
    var len_total = len(entrada.dados)
    @parameter
    if has_nvidia_gpu_accelerator():
        try:
            with DeviceContext(0, api="cuda") as ctx:
                var in_dev = ctx.enqueue_create_buffer[DType.float32](len_total)
                var grad_saida_dev = ctx.enqueue_create_buffer[DType.float32](len_total)
                var out_dev = ctx.enqueue_create_buffer[DType.float32](len_total)
                _copiar_host_para_dev(in_dev, entrada.dados, len_total)
                _copiar_host_para_dev(grad_saida_dev, grad_saida.dados, len_total)
                var block_dim = 256
                var grid_dim = (len_total + block_dim - 1) // block_dim
                ctx.enqueue_function_experimental[_kernel_derivada_hard_sigmoid](
                    in_dev,
                    grad_saida_dev,
                    out_dev,
                    len_total,
                    grid_dim=(grid_dim),
                    block_dim=(block_dim),
                )
                ctx.synchronize()
                _copiar_dev_para_tensor(out_dev, grad, len_total)
        except _:
            debug_assert(False, "falha ao executar kernel CUDA de derivada hard_sigmoid")
    else:
        debug_assert(False, "kernels CUDA nao disponiveis nesta compilacao")
    return grad^
