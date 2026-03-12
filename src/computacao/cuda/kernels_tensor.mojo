import src.nucleo.Tensor as tensor_defs
from gpu import global_idx
from gpu.host import DeviceContext, DeviceBuffer
from sys import has_nvidia_gpu_accelerator


# Implementação de referência para backend CUDA sem fallback para kernels CPU.

fn _kernel_add(
    in0: UnsafePointer[Float32],
    in1: UnsafePointer[Float32],
    out_ptr: UnsafePointer[Float32],
    len_total: Int,
):
    var tid = global_idx.x
    if tid >= UInt(len_total):
        return
    out_ptr[tid] = in0[tid] + in1[tid]


fn _kernel_sub(
    in0: UnsafePointer[Float32],
    in1: UnsafePointer[Float32],
    out_ptr: UnsafePointer[Float32],
    len_total: Int,
):
    var tid = global_idx.x
    if tid >= UInt(len_total):
        return
    out_ptr[tid] = in0[tid] - in1[tid]


fn _kernel_mul(
    in0: UnsafePointer[Float32],
    in1: UnsafePointer[Float32],
    out_ptr: UnsafePointer[Float32],
    len_total: Int,
):
    var tid = global_idx.x
    if tid >= UInt(len_total):
        return
    out_ptr[tid] = in0[tid] * in1[tid]


fn _kernel_transpose(
    in0: UnsafePointer[Float32],
    out_ptr: UnsafePointer[Float32],
    linhas: Int,
    colunas: Int,
):
    var tid = global_idx.x
    var total = UInt(linhas * colunas)
    if tid >= total:
        return
    var idx = Int(tid)
    var i = idx // colunas
    var j = idx - i * colunas
    out_ptr[j * linhas + i] = in0[idx]


fn _kernel_matmul(
    a: UnsafePointer[Float32],
    b: UnsafePointer[Float32],
    out_ptr: UnsafePointer[Float32],
    m: Int,
    n: Int,
    p: Int,
):
    var tid = global_idx.x
    var total = UInt(m * p)
    if tid >= total:
        return
    var idx = Int(tid)
    var i = idx // p
    var j = idx - i * p
    var acc: Float32 = 0.0
    for k in range(n):
        acc = acc + a[i * n + k] * b[k * p + j]
    out_ptr[idx] = acc


fn _kernel_add_bias(
    in0: UnsafePointer[Float32],
    out_ptr: UnsafePointer[Float32],
    len_total: Int,
    bias: Float32,
):
    var tid = global_idx.x
    if tid >= UInt(len_total):
        return
    out_ptr[tid] = in0[tid] + bias


fn _kernel_grad_mse(
    pred: UnsafePointer[Float32],
    alvo: UnsafePointer[Float32],
    out_ptr: UnsafePointer[Float32],
    len_total: Int,
):
    var tid = global_idx.x
    if tid >= UInt(len_total):
        return
    var n = Float32(len_total)
    out_ptr[tid] = 2.0 * (pred[tid] - alvo[tid]) / n


fn _copiar_lista_para_device(mut dev: DeviceBuffer[DType.float32], dados: List[Float32], var len_total: Int) raises:
    with dev.map_to_host() as host:
        for i in range(len_total):
            host[i] = dados[i]


fn _copiar_device_para_tensor(mut dev: DeviceBuffer[DType.float32], mut out: tensor_defs.Tensor, var len_total: Int) raises:
    with dev.map_to_host() as host:
        for i in range(len_total):
            out.dados[i] = host[i]

fn pipeline_id_cuda(var pipeline_memoria_id: Int, var operacao_id: Int) -> Int:
    return pipeline_memoria_id * 1000 + operacao_id


fn somar_elemento_a_elemento_cuda(a: tensor_defs.Tensor, b: tensor_defs.Tensor, var pipeline_id: Int) -> tensor_defs.Tensor:
    debug_assert(len(a.dados) == len(b.dados), "tensores devem ter mesmo tamanho")
    var formato = a.formato.copy()
    var out = tensor_defs.Tensor(formato^, a.tipo_computacao)
    var len_total = len(a.dados)
    @parameter
    if has_nvidia_gpu_accelerator():
        try:
            with DeviceContext(0, api="cuda") as ctx:
                var in0 = ctx.enqueue_create_buffer[DType.float32](len_total)
                var in1 = ctx.enqueue_create_buffer[DType.float32](len_total)
                var out_dev = ctx.enqueue_create_buffer[DType.float32](len_total)
                _copiar_lista_para_device(in0, a.dados, len_total)
                _copiar_lista_para_device(in1, b.dados, len_total)
                var block_dim = 256
                var grid_dim = (len_total + block_dim - 1) // block_dim
                ctx.enqueue_function_experimental[_kernel_add](
                    in0,
                    in1,
                    out_dev,
                    len_total,
                    grid_dim=(grid_dim),
                    block_dim=(block_dim),
                )
                ctx.synchronize()
                _copiar_device_para_tensor(out_dev, out, len_total)
        except _:
            debug_assert(False, "falha ao executar kernel CUDA de soma")
    else:
        debug_assert(False, "kernels CUDA nao disponiveis nesta compilacao")
    out.id_pipeline_ultima_operacao = pipeline_id
    return out^


fn subtrair_elemento_a_elemento_cuda(a: tensor_defs.Tensor, b: tensor_defs.Tensor, var pipeline_id: Int) -> tensor_defs.Tensor:
    debug_assert(len(a.dados) == len(b.dados), "tensores devem ter mesmo tamanho")
    var formato = a.formato.copy()
    var out = tensor_defs.Tensor(formato^, a.tipo_computacao)
    var len_total = len(a.dados)
    @parameter
    if has_nvidia_gpu_accelerator():
        try:
            with DeviceContext(0, api="cuda") as ctx:
                var in0 = ctx.enqueue_create_buffer[DType.float32](len_total)
                var in1 = ctx.enqueue_create_buffer[DType.float32](len_total)
                var out_dev = ctx.enqueue_create_buffer[DType.float32](len_total)
                _copiar_lista_para_device(in0, a.dados, len_total)
                _copiar_lista_para_device(in1, b.dados, len_total)
                var block_dim = 256
                var grid_dim = (len_total + block_dim - 1) // block_dim
                ctx.enqueue_function_experimental[_kernel_sub](
                    in0,
                    in1,
                    out_dev,
                    len_total,
                    grid_dim=(grid_dim),
                    block_dim=(block_dim),
                )
                ctx.synchronize()
                _copiar_device_para_tensor(out_dev, out, len_total)
        except _:
            debug_assert(False, "falha ao executar kernel CUDA de subtracao")
    else:
        debug_assert(False, "kernels CUDA nao disponiveis nesta compilacao")
    out.id_pipeline_ultima_operacao = pipeline_id
    return out^


fn multiplicar_elemento_a_elemento_cuda(a: tensor_defs.Tensor, b: tensor_defs.Tensor, var pipeline_id: Int) -> tensor_defs.Tensor:
    debug_assert(len(a.dados) == len(b.dados), "tensores devem ter mesmo tamanho")
    var formato = a.formato.copy()
    var out = tensor_defs.Tensor(formato^, a.tipo_computacao)
    var len_total = len(a.dados)
    @parameter
    if has_nvidia_gpu_accelerator():
        try:
            with DeviceContext(0, api="cuda") as ctx:
                var in0 = ctx.enqueue_create_buffer[DType.float32](len_total)
                var in1 = ctx.enqueue_create_buffer[DType.float32](len_total)
                var out_dev = ctx.enqueue_create_buffer[DType.float32](len_total)
                _copiar_lista_para_device(in0, a.dados, len_total)
                _copiar_lista_para_device(in1, b.dados, len_total)
                var block_dim = 256
                var grid_dim = (len_total + block_dim - 1) // block_dim
                ctx.enqueue_function_experimental[_kernel_mul](
                    in0,
                    in1,
                    out_dev,
                    len_total,
                    grid_dim=(grid_dim),
                    block_dim=(block_dim),
                )
                ctx.synchronize()
                _copiar_device_para_tensor(out_dev, out, len_total)
        except _:
            debug_assert(False, "falha ao executar kernel CUDA de multiplicacao")
    else:
        debug_assert(False, "kernels CUDA nao disponiveis nesta compilacao")
    out.id_pipeline_ultima_operacao = pipeline_id
    return out^


fn transpor_cuda(a: tensor_defs.Tensor, var pipeline_id: Int) -> tensor_defs.Tensor:
    debug_assert(len(a.formato) == 2, "transpor requer tensor 2D")
    var linhas = a.formato[0]
    var colunas = a.formato[1]
    var formato = List[Int]()
    formato.append(colunas)
    formato.append(linhas)
    var out = tensor_defs.Tensor(formato^, a.tipo_computacao)
    var len_total = len(a.dados)
    @parameter
    if has_nvidia_gpu_accelerator():
        try:
            with DeviceContext(0, api="cuda") as ctx:
                var in0 = ctx.enqueue_create_buffer[DType.float32](len_total)
                var out_dev = ctx.enqueue_create_buffer[DType.float32](len_total)
                _copiar_lista_para_device(in0, a.dados, len_total)
                var block_dim = 256
                var grid_dim = (len_total + block_dim - 1) // block_dim
                ctx.enqueue_function_experimental[_kernel_transpose](
                    in0,
                    out_dev,
                    linhas,
                    colunas,
                    grid_dim=(grid_dim),
                    block_dim=(block_dim),
                )
                ctx.synchronize()
                _copiar_device_para_tensor(out_dev, out, len_total)
        except _:
            debug_assert(False, "falha ao executar kernel CUDA de transposicao")
    else:
        debug_assert(False, "kernels CUDA nao disponiveis nesta compilacao")
    out.id_pipeline_ultima_operacao = pipeline_id
    return out^


fn multiplicar_matrizes_cuda(a: tensor_defs.Tensor, b: tensor_defs.Tensor, var pipeline_id: Int) -> tensor_defs.Tensor:
    debug_assert(len(a.formato) == 2 and len(b.formato) == 2, "matmul requer tensores 2D")
    var m = a.formato[0]
    var n = a.formato[1]
    debug_assert(n == b.formato[0], "dimensões incompatíveis para matmul")
    var p = b.formato[1]
    var formato = List[Int]()
    formato.append(m)
    formato.append(p)
    var out = tensor_defs.Tensor(formato^, a.tipo_computacao)
    var len_a = len(a.dados)
    var len_b = len(b.dados)
    var len_out = len(out.dados)
    @parameter
    if has_nvidia_gpu_accelerator():
        try:
            with DeviceContext(0, api="cuda") as ctx:
                var a_dev = ctx.enqueue_create_buffer[DType.float32](len_a)
                var b_dev = ctx.enqueue_create_buffer[DType.float32](len_b)
                var out_dev = ctx.enqueue_create_buffer[DType.float32](len_out)
                _copiar_lista_para_device(a_dev, a.dados, len_a)
                _copiar_lista_para_device(b_dev, b.dados, len_b)
                var block_dim = 256
                var grid_dim = (len_out + block_dim - 1) // block_dim
                ctx.enqueue_function_experimental[_kernel_matmul](
                    a_dev,
                    b_dev,
                    out_dev,
                    m,
                    n,
                    p,
                    grid_dim=(grid_dim),
                    block_dim=(block_dim),
                )
                ctx.synchronize()
                _copiar_device_para_tensor(out_dev, out, len_out)
        except _:
            debug_assert(False, "falha ao executar kernel CUDA de matmul")
    else:
        debug_assert(False, "kernels CUDA nao disponiveis nesta compilacao")
    out.id_pipeline_ultima_operacao = pipeline_id
    return out^


fn adicionar_bias_coluna_cuda(a: tensor_defs.Tensor, b: tensor_defs.Tensor, var pipeline_id: Int) -> tensor_defs.Tensor:
    debug_assert(len(a.formato) == 2, "entrada deve ser 2D")
    debug_assert(len(b.formato) == 2 and b.formato[0] == 1 and b.formato[1] == 1, "bias deve ter formato [1,1]")
    var formato = a.formato.copy()
    var out = tensor_defs.Tensor(formato^, a.tipo_computacao)
    var valor_bias = b.dados[0]
    var len_total = len(a.dados)
    @parameter
    if has_nvidia_gpu_accelerator():
        try:
            with DeviceContext(0, api="cuda") as ctx:
                var in0 = ctx.enqueue_create_buffer[DType.float32](len_total)
                var out_dev = ctx.enqueue_create_buffer[DType.float32](len_total)
                _copiar_lista_para_device(in0, a.dados, len_total)
                var block_dim = 256
                var grid_dim = (len_total + block_dim - 1) // block_dim
                ctx.enqueue_function_experimental[_kernel_add_bias](
                    in0,
                    out_dev,
                    len_total,
                    valor_bias,
                    grid_dim=(grid_dim),
                    block_dim=(block_dim),
                )
                ctx.synchronize()
                _copiar_device_para_tensor(out_dev, out, len_total)
        except _:
            debug_assert(False, "falha ao executar kernel CUDA de add_bias")
    else:
        debug_assert(False, "kernels CUDA nao disponiveis nesta compilacao")
    out.id_pipeline_ultima_operacao = pipeline_id
    return out^


fn gradiente_mse_cuda(pred: tensor_defs.Tensor, alvo: tensor_defs.Tensor, var pipeline_id: Int) -> tensor_defs.Tensor:
    debug_assert(len(pred.dados) == len(alvo.dados), "pred e alvo devem ter mesmo tamanho")
    var formato = pred.formato.copy()
    var out = tensor_defs.Tensor(formato^, pred.tipo_computacao)
    if len(pred.dados) == 0:
        out.id_pipeline_ultima_operacao = pipeline_id
        return out^
    var len_total = len(pred.dados)
    @parameter
    if has_nvidia_gpu_accelerator():
        try:
            with DeviceContext(0, api="cuda") as ctx:
                var pred_dev = ctx.enqueue_create_buffer[DType.float32](len_total)
                var alvo_dev = ctx.enqueue_create_buffer[DType.float32](len_total)
                var out_dev = ctx.enqueue_create_buffer[DType.float32](len_total)
                _copiar_lista_para_device(pred_dev, pred.dados, len_total)
                _copiar_lista_para_device(alvo_dev, alvo.dados, len_total)
                var block_dim = 256
                var grid_dim = (len_total + block_dim - 1) // block_dim
                ctx.enqueue_function_experimental[_kernel_grad_mse](
                    pred_dev,
                    alvo_dev,
                    out_dev,
                    len_total,
                    grid_dim=(grid_dim),
                    block_dim=(block_dim),
                )
                ctx.synchronize()
                _copiar_device_para_tensor(out_dev, out, len_total)
        except _:
            debug_assert(False, "falha ao executar kernel CUDA de grad_mse")
    else:
        debug_assert(False, "kernels CUDA nao disponiveis nesta compilacao")
    out.id_pipeline_ultima_operacao = pipeline_id
    return out^
