fn _kernel_avgpool2x2_stride2(
    entrada: UnsafePointer[Float32],
    h: Int,
    w: Int,
    out_ptr: UnsafePointer[Float32],
    out_h: Int,
    out_w: Int,
):
    var tid = global_idx.x
    var total = UInt(out_h * out_w)
    if tid >= total:
        return
    var idx = Int(tid)
    var y = idx // out_w
    var x = idx - y * out_w
    var y0 = y * 2
    var x0 = x * 2
    var s = entrada[y0 * w + x0] + entrada[y0 * w + x0 + 1] + entrada[(y0 + 1) * w + x0] + entrada[(y0 + 1) * w + x0 + 1]
    out_ptr[idx] = s * 0.25

fn avgpool2x2_stride2_cuda(
    entrada: List[Float32],
    h: Int,
    w: Int,
    tipo_computacao: String = "cuda",
    pipeline_id: Int = 0,
) -> List[Float32]:
    var out_h = h // 2
    var out_w = w // 2
    var total = out_h * out_w
    var out = List[Float32](capacity=total)
    for _ in range(total):
        out.append(0.0)
    @parameter
    if has_nvidia_gpu_accelerator():
        try:
            with DeviceContext(0, api="cuda") as ctx:
                var in_dev = ctx.enqueue_create_buffer[DType.float32](len(entrada))
                var out_dev = ctx.enqueue_create_buffer[DType.float32](total)
                _copiar_lista_para_device(in_dev, entrada, len(entrada))
                var block_dim = 128
                var grid_dim = (total + block_dim - 1) // block_dim
                ctx.enqueue_function_experimental[_kernel_avgpool2x2_stride2](
                    in_dev,
                    h,
                    w,
                    out_dev,
                    out_h,
                    out_w,
                    grid_dim=(grid_dim),
                    block_dim=(block_dim),
                )
                ctx.synchronize()
                _copiar_device_para_tensor(out_dev, out, total)
        except _:
            debug_assert(False, "falha ao executar avgpool2x2_stride2_cuda")
    else:
        debug_assert(False, "kernels CUDA nao disponiveis nesta compilacao")
    return out^
fn _kernel_conv2d_valid_relu(
    imagem: UnsafePointer[Float32],
    altura: Int,
    largura: Int,
    kernel: UnsafePointer[Float32],
    kh: Int,
    kw: Int,
    out_ptr: UnsafePointer[Float32],
    out_h: Int,
    out_w: Int,
):
    var tid = global_idx.x
    var total = UInt(out_h * out_w)
    if tid >= total:
        return
    var idx = Int(tid)
    var y = idx // out_w
    var x = idx - y * out_w
    var acc: Float32 = 0.0
    for ky in range(kh):
        for kx in range(kw):
            var iy = y + ky
            var ix = x + kx
            var v = imagem[iy * largura + ix]
            var w = kernel[ky * kw + kx]
            acc = acc + v * w
    out_ptr[idx] = acc if acc > 0.0 else Float32(0.0)

fn conv2d_valid_relu_cuda(
    imagem: List[Float32],
    altura: Int,
    largura: Int,
    kernel: List[Float32],
    kh: Int,
    kw: Int,
    tipo_computacao: String = "cuda",
    pipeline_id: Int = 0,
) -> List[Float32]:
    # Parametrização padrão do projeto: listas, tipos explícitos, pipeline_id opcional
    var out_h = altura - kh + 1
    var out_w = largura - kw + 1
    var total = out_h * out_w
    var out = List[Float32](capacity=total)
    for _ in range(total):
        out.append(0.0)
    @parameter
    if has_nvidia_gpu_accelerator():
        try:
            with DeviceContext(0, api="cuda") as ctx:
                var img_dev = ctx.enqueue_create_buffer[DType.float32](len(imagem))
                var ker_dev = ctx.enqueue_create_buffer[DType.float32](len(kernel))
                var out_dev = ctx.enqueue_create_buffer[DType.float32](total)
                _copiar_lista_para_device(img_dev, imagem, len(imagem))
                _copiar_lista_para_device(ker_dev, kernel, len(kernel))
                var block_dim = 128
                var grid_dim = (total + block_dim - 1) // block_dim
                ctx.enqueue_function_experimental[_kernel_conv2d_valid_relu](
                    img_dev,
                    altura,
                    largura,
                    ker_dev,
                    kh,
                    kw,
                    out_dev,
                    out_h,
                    out_w,
                    grid_dim=(grid_dim),
                    block_dim=(block_dim),
                )
                ctx.synchronize()
                _copiar_device_para_tensor(out_dev, out, total)
        except _:
            debug_assert(False, "falha ao executar conv2d_valid_relu_cuda")
    else:
        debug_assert(False, "kernels CUDA nao disponiveis nesta compilacao")
    return out^
fn _kernel_linear_bias_gelu(
    a: UnsafePointer[Float32],
    w: UnsafePointer[Float32],
    b: UnsafePointer[Float32],
    out_ptr: UnsafePointer[Float32],
    batch: Int,
    fan_in: Int,
    fan_out: Int,
):
    var tid = global_idx.x
    var total = UInt(batch * fan_out)
    if tid >= total:
        return
    var idx = Int(tid)
    var i = idx // fan_out
    var j = idx - i * fan_out
    var acc: Float32 = 0.0
    for k in range(fan_in):
        acc = acc + a[i * fan_in + k] * w[k * fan_out + j]
    acc = acc + b[j]
    # GELU (approx)
    var c = 0.7978845608 * (acc + 0.044715 * acc * acc * acc)
    out_ptr[idx] = 0.5 * acc * (1.0 + tanh(c))
fn _kernel_linear_bias_sigmoid(
    a: UnsafePointer[Float32],
    w: UnsafePointer[Float32],
    b: UnsafePointer[Float32],
    out_ptr: UnsafePointer[Float32],
    batch: Int,
    fan_in: Int,
    fan_out: Int,
):
    var tid = global_idx.x
    var total = UInt(batch * fan_out)
    if tid >= total:
        return
    var idx = Int(tid)
    var i = idx // fan_out
    var j = idx - i * fan_out
    var acc: Float32 = 0.0
    for k in range(fan_in):
        acc = acc + a[i * fan_in + k] * w[k * fan_out + j]
    acc = acc + b[j]
    # Sigmoid
    out_ptr[idx] = 1.0 / (1.0 + exp(-acc))
import src.nucleo.Tensor as tensor_defs
import src.computacao.cuda.device_buffer_pool as buffer_pool
from gpu import global_idx
from gpu.host import DeviceContext, DeviceBuffer
from sys import has_nvidia_gpu_accelerator
from math import exp, tanh


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


fn _kernel_matmul_a_transposto_b(
    a: UnsafePointer[Float32],
    b: UnsafePointer[Float32],
    out_ptr: UnsafePointer[Float32],
    batch: Int,
    fan_in: Int,
    fan_out: Int,
):
    var tid = global_idx.x
    var total = UInt(fan_in * fan_out)
    if tid >= total:
        return
    var idx = Int(tid)
    var i = idx // fan_out
    var j = idx - i * fan_out
    var acc: Float32 = 0.0
    for k in range(batch):
        acc = acc + a[k * fan_in + i] * b[k * fan_out + j]
    out_ptr[idx] = acc


fn _kernel_matmul_b_transposto(
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
        acc = acc + a[i * n + k] * b[j * n + k]
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


fn _kernel_add_bias_vetor_coluna(
    in0: UnsafePointer[Float32],
    bias: UnsafePointer[Float32],
    out_ptr: UnsafePointer[Float32],
    linhas: Int,
    colunas: Int,
):
    var tid = global_idx.x
    var total = UInt(linhas * colunas)
    if tid >= total:
        return
    var idx = Int(tid)
    var j = idx % colunas
    out_ptr[idx] = in0[idx] + bias[j]


fn _kernel_somar_linhas(
    in0: UnsafePointer[Float32],
    out_ptr: UnsafePointer[Float32],
    linhas: Int,
    colunas: Int,
):
    var tid = global_idx.x
    if tid >= UInt(colunas):
        return
    var j = Int(tid)
    var acc: Float32 = 0.0
    for i in range(linhas):
        acc = acc + in0[i * colunas + j]
    out_ptr[j] = acc


fn _kernel_softmax_linhas(
    in0: UnsafePointer[Float32],
    out_ptr: UnsafePointer[Float32],
    linhas: Int,
    colunas: Int,
):
    var tid = global_idx.x
    if tid >= UInt(linhas):
        return
    var i = Int(tid)
    var base = i * colunas

    var max_v = in0[base]
    for j in range(1, colunas):
        var v = in0[base + j]
        if v > max_v:
            max_v = v

    var soma_exp: Float32 = 0.0
    for j in range(colunas):
        var e = exp(in0[base + j] - max_v)
        out_ptr[base + j] = e
        soma_exp = soma_exp + e

    if soma_exp <= 0.0:
        soma_exp = 1.0

    for j in range(colunas):
        out_ptr[base + j] = out_ptr[base + j] / soma_exp


fn _kernel_grad_softmax_cross_entropy(
    pred: UnsafePointer[Float32],
    alvo: UnsafePointer[Float32],
    out_ptr: UnsafePointer[Float32],
    len_total: Int,
    n_linhas: Int,
):
    var tid = global_idx.x
    if tid >= UInt(len_total):
        return
    var n = Float32(n_linhas)
    out_ptr[tid] = (pred[tid] - alvo[tid]) / n


fn _kernel_derivada_relu_local(
    entrada: UnsafePointer[Float32],
    grad_saida: UnsafePointer[Float32],
    grad: UnsafePointer[Float32],
    len_total: Int,
):
    var tid = global_idx.x
    if tid >= UInt(len_total):
        return
    grad[tid] = grad_saida[tid] if entrada[tid] > 0.0 else Float32(0.0)


fn _kernel_relu_inplace(
    dados: UnsafePointer[Float32],
    len_total: Int,
):
    var tid = global_idx.x
    if tid >= UInt(len_total):
        return
    var v = dados[tid]
    dados[tid] = v if v > 0.0 else Float32(0.0)


fn _kernel_copy(
    origem: UnsafePointer[Float32],
    destino: UnsafePointer[Float32],
    len_total: Int,
):
    var tid = global_idx.x
    if tid >= UInt(len_total):
        return
    destino[tid] = origem[tid]


fn _kernel_hard_sigmoid_inplace_local(
    dados: UnsafePointer[Float32],
    len_total: Int,
):
    var tid = global_idx.x
    if tid >= UInt(len_total):
        return
    var v = 0.2 * dados[tid] + 0.5
    if v < 0.0:
        v = 0.0
    if v > 1.0:
        v = 1.0
    dados[tid] = v


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


fn _kernel_sgd_step(
    param: UnsafePointer[Float32],
    grad: UnsafePointer[Float32],
    out_ptr: UnsafePointer[Float32],
    len_total: Int,
    taxa_aprendizado: Float32,
):
    var tid = global_idx.x
    if tid >= UInt(len_total):
        return
    out_ptr[tid] = param[tid] - taxa_aprendizado * grad[tid]


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


fn aplicar_sgd_em_tensor_cuda(
    param: tensor_defs.Tensor,
    grad: tensor_defs.Tensor,
    var taxa_aprendizado: Float32,
    var pipeline_id: Int,
) -> tensor_defs.Tensor:
    debug_assert(len(param.dados) == len(grad.dados), "param e grad devem ter mesmo tamanho")
    debug_assert(param.formato == grad.formato, "param e grad com formatos incompatíveis")

    var formato = param.formato.copy()
    var out = tensor_defs.Tensor(formato^, param.tipo_computacao)
    var len_total = len(param.dados)

    @parameter
    if has_nvidia_gpu_accelerator():
        try:
            with DeviceContext(0, api="cuda") as ctx:
                var param_dev = buffer_pool.device_buffer_pool.acquire(len_total)
                var grad_dev = buffer_pool.device_buffer_pool.acquire(len_total)
                var out_dev = buffer_pool.device_buffer_pool.acquire(len_total)

                _copiar_lista_para_device(param_dev, param.dados, len_total)
                _copiar_lista_para_device(grad_dev, grad.dados, len_total)

                var block_dim = 256
                var grid_dim = (len_total + block_dim - 1) // block_dim
                ctx.enqueue_function_experimental[_kernel_sgd_step](
                    param_dev,
                    grad_dev,
                    out_dev,
                    len_total,
                    taxa_aprendizado,
                    grid_dim=(grid_dim),
                    block_dim=(block_dim),
                )

                ctx.synchronize()
                _copiar_device_para_tensor(out_dev, out, len_total)
                buffer_pool.device_buffer_pool.release(param_dev)
                buffer_pool.device_buffer_pool.release(grad_dev)
                buffer_pool.device_buffer_pool.release(out_dev)
        except _:
            debug_assert(False, "falha ao executar aplicar_sgd_em_tensor_cuda")
    else:
        debug_assert(False, "kernels CUDA nao disponiveis nesta compilacao")

    out.id_pipeline_ultima_operacao = pipeline_id
    return out^


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


fn adicionar_bias_vetor_coluna_cuda(a: tensor_defs.Tensor, b: tensor_defs.Tensor, var pipeline_id: Int) -> tensor_defs.Tensor:
    debug_assert(len(a.formato) == 2, "entrada deve ser 2D")
    debug_assert(len(b.formato) == 2 and b.formato[0] == 1 and b.formato[1] == a.formato[1], "bias deve ser [1,colunas]")
    var linhas = a.formato[0]
    var colunas = a.formato[1]
    var formato = a.formato.copy()
    var out = tensor_defs.Tensor(formato^, a.tipo_computacao)
    var len_a = len(a.dados)
    var len_b = len(b.dados)
    var len_out = len(out.dados)
    @parameter
    if has_nvidia_gpu_accelerator():
        try:
            with DeviceContext(0, api="cuda") as ctx:
                var in0 = ctx.enqueue_create_buffer[DType.float32](len_a)
                var bias_dev = ctx.enqueue_create_buffer[DType.float32](len_b)
                var out_dev = ctx.enqueue_create_buffer[DType.float32](len_out)
                _copiar_lista_para_device(in0, a.dados, len_a)
                _copiar_lista_para_device(bias_dev, b.dados, len_b)
                var block_dim = 256
                var grid_dim = (len_out + block_dim - 1) // block_dim
                ctx.enqueue_function_experimental[_kernel_add_bias_vetor_coluna](
                    in0,
                    bias_dev,
                    out_dev,
                    linhas,
                    colunas,
                    grid_dim=(grid_dim),
                    block_dim=(block_dim),
                )
                ctx.synchronize()
                _copiar_device_para_tensor(out_dev, out, len_out)
        except _:
            debug_assert(False, "falha ao executar kernel CUDA de add_bias_vetor_coluna")
    else:
        debug_assert(False, "kernels CUDA nao disponiveis nesta compilacao")
    out.id_pipeline_ultima_operacao = pipeline_id
    return out^


fn somar_linhas_cuda(a: tensor_defs.Tensor, var pipeline_id: Int) -> tensor_defs.Tensor:
    debug_assert(len(a.formato) == 2, "entrada deve ser 2D")
    var linhas = a.formato[0]
    var colunas = a.formato[1]
    var formato = List[Int]()
    formato.append(1)
    formato.append(colunas)
    var out = tensor_defs.Tensor(formato^, a.tipo_computacao)
    if linhas <= 0 or colunas <= 0:
        out.id_pipeline_ultima_operacao = pipeline_id
        return out^

    var len_in = len(a.dados)
    var len_out = len(out.dados)
    @parameter
    if has_nvidia_gpu_accelerator():
        try:
            with DeviceContext(0, api="cuda") as ctx:
                var in0 = ctx.enqueue_create_buffer[DType.float32](len_in)
                var out_dev = ctx.enqueue_create_buffer[DType.float32](len_out)
                _copiar_lista_para_device(in0, a.dados, len_in)
                var block_dim = 256
                var grid_dim = (colunas + block_dim - 1) // block_dim
                ctx.enqueue_function_experimental[_kernel_somar_linhas](
                    in0,
                    out_dev,
                    linhas,
                    colunas,
                    grid_dim=(grid_dim),
                    block_dim=(block_dim),
                )
                ctx.synchronize()
                _copiar_device_para_tensor(out_dev, out, len_out)
        except _:
            debug_assert(False, "falha ao executar kernel CUDA de somar_linhas")
    else:
        debug_assert(False, "kernels CUDA nao disponiveis nesta compilacao")
    out.id_pipeline_ultima_operacao = pipeline_id
    return out^


fn softmax_linhas_cuda(z: tensor_defs.Tensor, var pipeline_id: Int) -> tensor_defs.Tensor:
    debug_assert(len(z.formato) == 2, "softmax espera tensor 2D")
    var linhas = z.formato[0]
    var colunas = z.formato[1]
    var formato = z.formato.copy()
    var out = tensor_defs.Tensor(formato^, z.tipo_computacao)
    if linhas <= 0 or colunas <= 0:
        out.id_pipeline_ultima_operacao = pipeline_id
        return out^

    var len_total = len(z.dados)
    @parameter
    if has_nvidia_gpu_accelerator():
        try:
            with DeviceContext(0, api="cuda") as ctx:
                var in0 = ctx.enqueue_create_buffer[DType.float32](len_total)
                var out_dev = ctx.enqueue_create_buffer[DType.float32](len_total)
                _copiar_lista_para_device(in0, z.dados, len_total)
                var block_dim = 128
                var grid_dim = (linhas + block_dim - 1) // block_dim
                ctx.enqueue_function_experimental[_kernel_softmax_linhas](
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
            debug_assert(False, "falha ao executar kernel CUDA de softmax_linhas")
    else:
        debug_assert(False, "kernels CUDA nao disponiveis nesta compilacao")
    out.id_pipeline_ultima_operacao = pipeline_id
    return out^


fn grad_softmax_cross_entropy_cuda(pred_prob: tensor_defs.Tensor, alvos: tensor_defs.Tensor, var pipeline_id: Int) -> tensor_defs.Tensor:
    debug_assert(len(pred_prob.formato) == 2 and len(alvos.formato) == 2, "grad softmax+ce espera tensores 2D")
    debug_assert(pred_prob.formato[0] == alvos.formato[0] and pred_prob.formato[1] == alvos.formato[1], "pred e alvo incompatíveis")
    var linhas = pred_prob.formato[0]
    var formato = pred_prob.formato.copy()
    var out = tensor_defs.Tensor(formato^, pred_prob.tipo_computacao)
    var len_total = len(pred_prob.dados)
    if linhas <= 0 or len_total <= 0:
        out.id_pipeline_ultima_operacao = pipeline_id
        return out^

    @parameter
    if has_nvidia_gpu_accelerator():
        try:
            with DeviceContext(0, api="cuda") as ctx:
                var pred_dev = ctx.enqueue_create_buffer[DType.float32](len_total)
                var alvo_dev = ctx.enqueue_create_buffer[DType.float32](len_total)
                var out_dev = ctx.enqueue_create_buffer[DType.float32](len_total)
                _copiar_lista_para_device(pred_dev, pred_prob.dados, len_total)
                _copiar_lista_para_device(alvo_dev, alvos.dados, len_total)
                var block_dim = 256
                var grid_dim = (len_total + block_dim - 1) // block_dim
                ctx.enqueue_function_experimental[_kernel_grad_softmax_cross_entropy](
                    pred_dev,
                    alvo_dev,
                    out_dev,
                    len_total,
                    linhas,
                    grid_dim=(grid_dim),
                    block_dim=(block_dim),
                )
                ctx.synchronize()
                _copiar_device_para_tensor(out_dev, out, len_total)
        except _:
            debug_assert(False, "falha ao executar kernel CUDA de grad_softmax_cross_entropy")
    else:
        debug_assert(False, "kernels CUDA nao disponiveis nesta compilacao")
    out.id_pipeline_ultima_operacao = pipeline_id
    return out^


fn linear_bias_relu_forward_cuda(
    entrada: tensor_defs.Tensor,
    pesos: tensor_defs.Tensor,
    bias: tensor_defs.Tensor,
    var pipeline_id: Int,
) -> List[tensor_defs.Tensor]:
    debug_assert(len(entrada.formato) == 2 and len(pesos.formato) == 2, "linear+relu espera tensores 2D")
    debug_assert(entrada.formato[1] == pesos.formato[0], "dimensões incompatíveis para matmul")
    debug_assert(len(bias.formato) == 2 and bias.formato[0] == 1 and bias.formato[1] == pesos.formato[1], "bias deve ser [1, fan_out]")

    var batch = entrada.formato[0]
    var fan_in = entrada.formato[1]
    var fan_out = pesos.formato[1]

    var formato_z = List[Int]()
    formato_z.append(batch)
    formato_z.append(fan_out)
    var z_out = tensor_defs.Tensor(formato_z.copy(), entrada.tipo_computacao)
    var relu_out = tensor_defs.Tensor(formato_z^, entrada.tipo_computacao)

    var len_a = len(entrada.dados)
    var len_w = len(pesos.dados)
    var len_b = len(bias.dados)
    var len_out = len(z_out.dados)

    @parameter
    if has_nvidia_gpu_accelerator():
        try:
            with DeviceContext(0, api="cuda") as ctx:
                var a_dev = ctx.enqueue_create_buffer[DType.float32](len_a)
                var w_dev = ctx.enqueue_create_buffer[DType.float32](len_w)
                var b_dev = ctx.enqueue_create_buffer[DType.float32](len_b)
                var z_dev = ctx.enqueue_create_buffer[DType.float32](len_out)

                _copiar_lista_para_device(a_dev, entrada.dados, len_a)
                _copiar_lista_para_device(w_dev, pesos.dados, len_w)
                _copiar_lista_para_device(b_dev, bias.dados, len_b)

                var block_dim = 256

                var grid_matmul = (len_out + block_dim - 1) // block_dim
                ctx.enqueue_function_experimental[_kernel_matmul](
                    a_dev,
                    w_dev,
                    z_dev,
                    batch,
                    fan_in,
                    fan_out,
                    grid_dim=(grid_matmul),
                    block_dim=(block_dim),
                )

                var grid_bias = (len_out + block_dim - 1) // block_dim
                ctx.enqueue_function_experimental[_kernel_add_bias_vetor_coluna](
                    z_dev,
                    b_dev,
                    z_dev,
                    batch,
                    fan_out,
                    grid_dim=(grid_bias),
                    block_dim=(block_dim),
                )

                # Copiamos z (pré-ativação) para manter contexto de autograd.
                ctx.synchronize()
                _copiar_device_para_tensor(z_dev, z_out, len_out)

                # Aplica ReLU in-place no mesmo fluxo/contexto.
                var grid_relu = (len_out + block_dim - 1) // block_dim
                ctx.enqueue_function_experimental[_kernel_relu_inplace](
                    z_dev,
                    len_out,
                    grid_dim=(grid_relu),
                    block_dim=(block_dim),
                )

                ctx.synchronize()
                _copiar_device_para_tensor(z_dev, relu_out, len_out)
        except _:
            debug_assert(False, "falha ao executar linear_bias_relu_forward_cuda")
    else:
        debug_assert(False, "kernels CUDA nao disponiveis nesta compilacao")

    z_out.id_pipeline_ultima_operacao = pipeline_id
    relu_out.id_pipeline_ultima_operacao = pipeline_id
    var out = List[tensor_defs.Tensor]()
    out.append(z_out.copy())
    out.append(relu_out.copy())
    return out^


fn linear_bias_softmax_forward_cuda(
    entrada: tensor_defs.Tensor,
    pesos: tensor_defs.Tensor,
    bias: tensor_defs.Tensor,
    var pipeline_id: Int,
) -> List[tensor_defs.Tensor]:
    debug_assert(len(entrada.formato) == 2 and len(pesos.formato) == 2, "linear+softmax espera tensores 2D")
    debug_assert(entrada.formato[1] == pesos.formato[0], "dimensões incompatíveis para matmul")
    debug_assert(len(bias.formato) == 2 and bias.formato[0] == 1 and bias.formato[1] == pesos.formato[1], "bias deve ser [1, fan_out]")

    var batch = entrada.formato[0]
    var fan_in = entrada.formato[1]
    var fan_out = pesos.formato[1]

    var formato = List[Int]()
    formato.append(batch)
    formato.append(fan_out)
    var z_out = tensor_defs.Tensor(formato.copy(), entrada.tipo_computacao)
    var pred_out = tensor_defs.Tensor(formato^, entrada.tipo_computacao)

    var len_a = len(entrada.dados)
    var len_w = len(pesos.dados)
    var len_b = len(bias.dados)
    var len_out = len(z_out.dados)

    @parameter
    if has_nvidia_gpu_accelerator():
        try:
            with DeviceContext(0, api="cuda") as ctx:
                var a_dev = ctx.enqueue_create_buffer[DType.float32](len_a)
                var w_dev = ctx.enqueue_create_buffer[DType.float32](len_w)
                var b_dev = ctx.enqueue_create_buffer[DType.float32](len_b)
                var z_dev = ctx.enqueue_create_buffer[DType.float32](len_out)

                _copiar_lista_para_device(a_dev, entrada.dados, len_a)
                _copiar_lista_para_device(w_dev, pesos.dados, len_w)
                _copiar_lista_para_device(b_dev, bias.dados, len_b)

                var block_dim = 256

                var grid_matmul = (len_out + block_dim - 1) // block_dim
                ctx.enqueue_function_experimental[_kernel_matmul](
                    a_dev,
                    w_dev,
                    z_dev,
                    batch,
                    fan_in,
                    fan_out,
                    grid_dim=(grid_matmul),
                    block_dim=(block_dim),
                )

                var grid_bias = (len_out + block_dim - 1) // block_dim
                ctx.enqueue_function_experimental[_kernel_add_bias_vetor_coluna](
                    z_dev,
                    b_dev,
                    z_dev,
                    batch,
                    fan_out,
                    grid_dim=(grid_bias),
                    block_dim=(block_dim),
                )

                ctx.synchronize()
                _copiar_device_para_tensor(z_dev, z_out, len_out)

                var softmax_block = 128
                var softmax_grid = (batch + softmax_block - 1) // softmax_block
                ctx.enqueue_function_experimental[_kernel_softmax_linhas](
                    z_dev,
                    z_dev,
                    batch,
                    fan_out,
                    grid_dim=(softmax_grid),
                    block_dim=(softmax_block),
                )

                ctx.synchronize()
                _copiar_device_para_tensor(z_dev, pred_out, len_out)
        except _:
            debug_assert(False, "falha ao executar linear_bias_softmax_forward_cuda")
    else:
        debug_assert(False, "kernels CUDA nao disponiveis nesta compilacao")

    z_out.id_pipeline_ultima_operacao = pipeline_id
    pred_out.id_pipeline_ultima_operacao = pipeline_id
    var out = List[tensor_defs.Tensor]()
    out.append(z_out.copy())
    out.append(pred_out.copy())
    return out^


fn mlp_forward_cuda_fused(
    entradas: tensor_defs.Tensor,
    pesos: List[tensor_defs.Tensor],
    biases: List[tensor_defs.Tensor],
    var ativacao_saida_id: Int,
    var ativacao_saida_softmax_id: Int,
    var ativacao_saida_linear_id: Int,
    var ativacao_saida_hard_sigmoid_id: Int,
    mut zs_out: List[tensor_defs.Tensor],
    mut ativs_out: List[tensor_defs.Tensor],
    var pipeline_id: Int,
) -> tensor_defs.Tensor:
    debug_assert(len(pesos) > 0 and len(pesos) == len(biases), "pesos e biases invalidos")
    debug_assert(len(entradas.formato) == 2, "entradas deve ser 2D")

    zs_out = List[tensor_defs.Tensor]()
    ativs_out = List[tensor_defs.Tensor]()
    ativs_out.append(entradas.copy())

    var num_camadas = len(pesos)
    var pred_host = entradas.copy()
    var batch = entradas.formato[0]
    var fan_in_atual = entradas.formato[1]

    var max_len_a = len(entradas.dados)
    var max_len_w = 1
    var max_len_b = 1
    var max_len_out = 1
    for camada in range(num_camadas):
        var w = pesos[camada].copy()
        var b = biases[camada].copy()
        var len_w = len(w.dados)
        var len_b = len(b.dados)
        var len_out = entradas.formato[0] * w.formato[1]
        if len_w > max_len_w:
            max_len_w = len_w
        if len_b > max_len_b:
            max_len_b = len_b
        if len_out > max_len_out:
            max_len_out = len_out
        if len_out > max_len_a:
            max_len_a = len_out

    @parameter
    if has_nvidia_gpu_accelerator():
        try:
            with DeviceContext(0, api="cuda") as ctx:
                var a_dev = ctx.enqueue_create_buffer[DType.float32](max_len_a)
                var w_dev = ctx.enqueue_create_buffer[DType.float32](max_len_w)
                var b_dev = ctx.enqueue_create_buffer[DType.float32](max_len_b)
                var z_dev = ctx.enqueue_create_buffer[DType.float32](max_len_out)

                var block_dim = 256

                var len_entrada = len(entradas.dados)
                _copiar_lista_para_device(a_dev, entradas.dados, len_entrada)

                for camada in range(num_camadas):
                    var w = pesos[camada].copy()
                    var b = biases[camada].copy()

                    var fan_in = fan_in_atual
                    var fan_out = w.formato[1]

                    var len_a = batch * fan_in
                    var len_w = len(w.dados)
                    var len_b = len(b.dados)
                    var len_out = batch * fan_out

                    _copiar_lista_para_device(w_dev, w.dados, len_w)
                    _copiar_lista_para_device(b_dev, b.dados, len_b)

                    var eh_saida = camada == num_camadas - 1
                    var usar_sigmoid = eh_saida and ativacao_saida_id == 42  # 42: id para sigmoid
                    var usar_gelu = eh_saida and ativacao_saida_id == 43    # 43: id para gelu
                    if usar_sigmoid:
                        var grid_fused = (len_out + block_dim - 1) // block_dim
                        ctx.enqueue_function_experimental[_kernel_linear_bias_sigmoid](
                            a_dev,
                            w_dev,
                            b_dev,
                            z_dev,
                            batch,
                            fan_in,
                            fan_out,
                            grid_dim=(grid_fused),
                            block_dim=(block_dim),
                        )
                        ctx.synchronize()
                        var formato_a = List[Int]()
                        formato_a.append(batch)
                        formato_a.append(fan_out)
                        var ativ_host = tensor_defs.Tensor(formato_a^, entradas.tipo_computacao)
                        _copiar_device_para_tensor(z_dev, ativ_host, len_out)
                        ativ_host.id_pipeline_ultima_operacao = pipeline_id
                        zs_out.append(ativ_host.copy())
                        ativs_out.append(ativ_host.copy())
                        pred_host = ativ_host.copy()
                        fan_in_atual = fan_out
                    elif usar_gelu:
                        var grid_fused = (len_out + block_dim - 1) // block_dim
                        ctx.enqueue_function_experimental[_kernel_linear_bias_gelu](
                            a_dev,
                            w_dev,
                            b_dev,
                            z_dev,
                            batch,
                            fan_in,
                            fan_out,
                            grid_dim=(grid_fused),
                            block_dim=(block_dim),
                        )
                        ctx.synchronize()
                        var formato_a = List[Int]()
                        formato_a.append(batch)
                        formato_a.append(fan_out)
                        var ativ_host = tensor_defs.Tensor(formato_a^, entradas.tipo_computacao)
                        _copiar_device_para_tensor(z_dev, ativ_host, len_out)
                        ativ_host.id_pipeline_ultima_operacao = pipeline_id
                        zs_out.append(ativ_host.copy())
                        ativs_out.append(ativ_host.copy())
                        pred_host = ativ_host.copy()
                        fan_in_atual = fan_out
                    else:
                        var grid_matmul = (len_out + block_dim - 1) // block_dim
                        ctx.enqueue_function_experimental[_kernel_matmul](
                            a_dev,
                            w_dev,
                            z_dev,
                            batch,
                            fan_in,
                            fan_out,
                            grid_dim=(grid_matmul),
                            block_dim=(block_dim),
                        )

                        var grid_bias = (len_out + block_dim - 1) // block_dim
                        ctx.enqueue_function_experimental[_kernel_add_bias_vetor_coluna](
                            z_dev,
                            b_dev,
                            z_dev,
                            batch,
                            fan_out,
                            grid_dim=(grid_bias),
                            block_dim=(block_dim),
                        )

                        if not eh_saida:
                            var grid_relu = (len_out + block_dim - 1) // block_dim
                            ctx.enqueue_function_experimental[_kernel_relu_inplace](
                                z_dev,
                                len_out,
                                grid_dim=(grid_relu),
                                block_dim=(block_dim),
                            )

                            # Mantém ativação no device para a próxima camada, evitando upload host->device.
                            var grid_copy_hidden = (len_out + block_dim - 1) // block_dim
                            ctx.enqueue_function_experimental[_kernel_copy](
                                z_dev,
                                a_dev,
                                len_out,
                                grid_dim=(grid_copy_hidden),
                                block_dim=(block_dim),
                            )

                            # Para ReLU, a derivada depende apenas do sinal; usar ativação como proxy de z evita uma cópia extra.
                            ctx.synchronize()
                            var formato_a_hidden = List[Int]()
                            formato_a_hidden.append(batch)
                            formato_a_hidden.append(fan_out)
                            var ativ_hidden = tensor_defs.Tensor(formato_a_hidden^, entradas.tipo_computacao)
                            _copiar_device_para_tensor(z_dev, ativ_hidden, len_out)
                            ativ_hidden.id_pipeline_ultima_operacao = pipeline_id
                            zs_out.append(ativ_hidden.copy())
                            ativs_out.append(ativ_hidden.copy())
                            fan_in_atual = fan_out
                        else:
                            var preservar_pre_ativacao = ativacao_saida_id == ativacao_saida_hard_sigmoid_id
                            if preservar_pre_ativacao:
                                ctx.synchronize()
                                var formato_z = List[Int]()
                                formato_z.append(batch)
                                formato_z.append(fan_out)
                                var z_host = tensor_defs.Tensor(formato_z^, entradas.tipo_computacao)
                                _copiar_device_para_tensor(z_dev, z_host, len_out)
                                z_host.id_pipeline_ultima_operacao = pipeline_id
                                zs_out.append(z_host.copy())

                            if ativacao_saida_id == ativacao_saida_softmax_id:
                                var softmax_block = 128
                                var softmax_grid = (batch + softmax_block - 1) // softmax_block
                                ctx.enqueue_function_experimental[_kernel_softmax_linhas](
                                    z_dev,
                                    z_dev,
                                    batch,
                                    fan_out,
                                    grid_dim=(softmax_grid),
                                    block_dim=(softmax_block),
                                )
                            elif ativacao_saida_id == ativacao_saida_hard_sigmoid_id:
                                var grid_hs = (len_out + block_dim - 1) // block_dim
                                ctx.enqueue_function_experimental[_kernel_hard_sigmoid_inplace_local](
                                    z_dev,
                                    len_out,
                                    grid_dim=(grid_hs),
                                    block_dim=(block_dim),
                                )
                            elif ativacao_saida_id == ativacao_saida_linear_id:
                                pass

                            ctx.synchronize()
                            var formato_a = List[Int]()
                            formato_a.append(batch)
                            formato_a.append(fan_out)
                            var ativ_host = tensor_defs.Tensor(formato_a^, entradas.tipo_computacao)
                            _copiar_device_para_tensor(z_dev, ativ_host, len_out)
                            ativ_host.id_pipeline_ultima_operacao = pipeline_id

                            # Para softmax/linear, o backward atual nao depende de z de saida;
                            # usar ativacao como proxy reduz uma copia device->host por iteracao.
                            if not preservar_pre_ativacao:
                                zs_out.append(ativ_host.copy())

                            ativs_out.append(ativ_host.copy())
                            pred_host = ativ_host.copy()
        except _:
            debug_assert(False, "falha ao executar mlp_forward_cuda_fused")
    else:
        debug_assert(False, "kernels CUDA nao disponiveis nesta compilacao")

    return pred_host^


fn passo_backprop_mlp_cuda(
    ativ_prev: tensor_defs.Tensor,
    grad_z_atual: tensor_defs.Tensor,
    peso_camada: tensor_defs.Tensor,
    z_derivada_relu: tensor_defs.Tensor,
    var calcular_grad_a_prev: Bool,
    var aplicar_derivada_relu: Bool,
    var pipeline_id: Int,
) -> List[tensor_defs.Tensor]:
    debug_assert(len(ativ_prev.formato) == 2 and len(grad_z_atual.formato) == 2, "passo backprop espera tensores 2D")
    debug_assert(ativ_prev.formato[0] == grad_z_atual.formato[0], "batch incompatível")
    debug_assert(len(peso_camada.formato) == 2, "peso da camada deve ser 2D")

    var batch = ativ_prev.formato[0]
    var fan_in = ativ_prev.formato[1]
    var fan_out = grad_z_atual.formato[1]

    debug_assert(peso_camada.formato[0] == fan_in and peso_camada.formato[1] == fan_out, "peso incompatível com gradiente da camada")
    if aplicar_derivada_relu:
        debug_assert(len(z_derivada_relu.dados) == batch * fan_in, "z da derivada relu incompatível com grad_a_prev")

    var formato_grad_w = List[Int]()
    formato_grad_w.append(fan_in)
    formato_grad_w.append(fan_out)
    var grad_w = tensor_defs.Tensor(formato_grad_w^, ativ_prev.tipo_computacao)

    var formato_grad_b = List[Int]()
    formato_grad_b.append(1)
    formato_grad_b.append(fan_out)
    var grad_b = tensor_defs.Tensor(formato_grad_b^, ativ_prev.tipo_computacao)

    var formato_grad_a_prev = List[Int]()
    formato_grad_a_prev.append(batch)
    formato_grad_a_prev.append(fan_in)
    var grad_a_prev = tensor_defs.Tensor(formato_grad_a_prev^, ativ_prev.tipo_computacao)

    var len_ativ_prev = len(ativ_prev.dados)
    var len_grad_z = len(grad_z_atual.dados)
    var len_grad_w = len(grad_w.dados)
    var len_grad_b = len(grad_b.dados)
    var len_peso = len(peso_camada.dados)
    var len_grad_a_prev = len(grad_a_prev.dados)

    @parameter
    if has_nvidia_gpu_accelerator():
        try:
            with DeviceContext(0, api="cuda") as ctx:
                var ativ_prev_dev = ctx.enqueue_create_buffer[DType.float32](len_ativ_prev)
                var grad_z_dev = ctx.enqueue_create_buffer[DType.float32](len_grad_z)
                var grad_w_dev = ctx.enqueue_create_buffer[DType.float32](len_grad_w)
                var grad_b_dev = ctx.enqueue_create_buffer[DType.float32](len_grad_b)
                var peso_dev = buffer_pool.device_buffer_pool.acquire(len_peso)
                var grad_a_prev_dev = buffer_pool.device_buffer_pool.acquire(len_grad_a_prev)

                var block_dim = 256

                var grid_matmul_w = (len_grad_w + block_dim - 1) // block_dim
                ctx.enqueue_function_experimental[_kernel_matmul_a_transposto_b](
                    ativ_prev_dev,
                    grad_z_dev,
                    grad_w_dev,
                    batch,
                    fan_in,
                    fan_out,
                    grid_dim=(grid_matmul_w),
                    block_dim=(block_dim),
                )

                var grid_sum_cols = (fan_out + block_dim - 1) // block_dim
                ctx.enqueue_function_experimental[_kernel_somar_linhas](
                    grad_z_dev,
                    grad_b_dev,
                    batch,
                    fan_out,
                    grid_dim=(grid_sum_cols),
                    block_dim=(block_dim),
                )

                if calcular_grad_a_prev:
                    var grid_matmul_a = (len_grad_a_prev + block_dim - 1) // block_dim
                    ctx.enqueue_function_experimental[_kernel_matmul_b_transposto](
                        grad_z_dev,
                        peso_dev,
                        grad_a_prev_dev,
                        batch,
                        fan_out,
                        fan_in,
                        grid_dim=(grid_matmul_a),
                        block_dim=(block_dim),
                    )

                    if aplicar_derivada_relu:
                        var grid_relu = (len_grad_a_prev + block_dim - 1) // block_dim
                        ctx.enqueue_function_experimental[_kernel_derivada_relu_local](
                            ativ_prev_dev,
                            grad_a_prev_dev,
                            grad_a_prev_dev,
                            len_grad_a_prev,
                            grid_dim=(grid_relu),
                            block_dim=(block_dim),
                        )

                    ctx.synchronize()
                    _copiar_device_para_tensor(grad_a_prev_dev, grad_a_prev, len_grad_a_prev)
                else:
                    ctx.synchronize()

                _copiar_device_para_tensor(grad_w_dev, grad_w, len_grad_w)
                _copiar_device_para_tensor(grad_b_dev, grad_b, len_grad_b)
        except _:
            debug_assert(False, "falha ao executar passo_backprop_mlp_cuda")
    else:
        debug_assert(False, "kernels CUDA nao disponiveis nesta compilacao")

    grad_w.id_pipeline_ultima_operacao = pipeline_id
    grad_b.id_pipeline_ultima_operacao = pipeline_id
    if calcular_grad_a_prev:
        grad_a_prev.id_pipeline_ultima_operacao = pipeline_id

    var out = List[tensor_defs.Tensor]()
    out.append(grad_w)
    out.append(grad_b)
    out.append(grad_a_prev)
    return out^


fn passo_backprop_mlp_cuda_em_tensores(
    ativ_prev: tensor_defs.Tensor,
    grad_z_atual: tensor_defs.Tensor,
    peso_camada: tensor_defs.Tensor,
    z_derivada_relu: tensor_defs.Tensor,
    mut grad_w_out: tensor_defs.Tensor,
    mut grad_b_out: tensor_defs.Tensor,
    mut grad_a_prev_out: tensor_defs.Tensor,
    var calcular_grad_a_prev: Bool,
    var aplicar_derivada_relu: Bool,
    var pipeline_id: Int,
):
    debug_assert(len(ativ_prev.formato) == 2 and len(grad_z_atual.formato) == 2, "passo backprop espera tensores 2D")
    debug_assert(ativ_prev.formato[0] == grad_z_atual.formato[0], "batch incompatível")
    debug_assert(len(peso_camada.formato) == 2, "peso da camada deve ser 2D")

    var batch = ativ_prev.formato[0]
    var fan_in = ativ_prev.formato[1]
    var fan_out = grad_z_atual.formato[1]

    debug_assert(peso_camada.formato[0] == fan_in and peso_camada.formato[1] == fan_out, "peso incompatível com gradiente da camada")
    debug_assert(len(grad_w_out.formato) == 2 and grad_w_out.formato[0] == fan_in and grad_w_out.formato[1] == fan_out, "grad_w_out com formato inválido")
    debug_assert(len(grad_b_out.formato) == 2 and grad_b_out.formato[0] == 1 and grad_b_out.formato[1] == fan_out, "grad_b_out com formato inválido")

    if calcular_grad_a_prev:
        debug_assert(len(grad_a_prev_out.formato) == 2 and grad_a_prev_out.formato[0] == batch and grad_a_prev_out.formato[1] == fan_in, "grad_a_prev_out com formato inválido")
    if aplicar_derivada_relu:
        debug_assert(len(z_derivada_relu.dados) == batch * fan_in, "z da derivada relu incompatível com grad_a_prev")

    var len_ativ_prev = len(ativ_prev.dados)
    var len_grad_z = len(grad_z_atual.dados)
    var len_grad_w = len(grad_w_out.dados)
    var len_grad_b = len(grad_b_out.dados)
    var len_peso = len(peso_camada.dados)
    var len_grad_a_prev = len(grad_a_prev_out.dados)

    @parameter
    if has_nvidia_gpu_accelerator():
        try:
            with DeviceContext(0, api="cuda") as ctx:
                var ativ_prev_dev = ctx.enqueue_create_buffer[DType.float32](len_ativ_prev)
                var grad_z_dev = ctx.enqueue_create_buffer[DType.float32](len_grad_z)
                var grad_w_dev = ctx.enqueue_create_buffer[DType.float32](len_grad_w)
                var grad_b_dev = ctx.enqueue_create_buffer[DType.float32](len_grad_b)
                var peso_dev = buffer_pool.device_buffer_pool.acquire(len_peso)
                var grad_a_prev_dev = buffer_pool.device_buffer_pool.acquire(len_grad_a_prev)

                var len_grad_z_atual = len(grad_z_atual.dados)
                _copiar_lista_para_device(grad_z_dev, grad_z_atual.dados, len_grad_z_atual)

                for passo in range(num_camadas):
                    var camada = num_camadas - 1 - passo
                    var ativ_prev = ativacoes_forward[camada]
                    var peso_camada = pesos[camada]
                    var calcular_grad_a_prev = camada > 0

                    var batch = ativ_prev.formato[0]
                    var fan_in = ativ_prev.formato[1]
                    var fan_out = peso_camada.formato[1]

                    var len_ativ_prev = len(ativ_prev.dados)
                    var len_grad_z = len_grad_z_atual
                    var len_grad_w = len(grad_ws_out[camada].dados)
                    var len_grad_b = len(grad_bs_out[camada].dados)
                    var len_peso = len(peso_camada.dados)
                    var len_grad_a_prev = len(grad_a_prev_out[camada].dados)

                    _copiar_lista_para_device(ativ_prev_dev, ativ_prev.dados, len_ativ_prev)
                    _copiar_lista_para_device(peso_dev, peso_camada.dados, len_peso)
                    _copiar_lista_para_device(bias_dev, bias_camada.dados, len_grad_b)

                    var grid_matmul_w = (len_grad_w + block_dim - 1) // block_dim
                    ctx.enqueue_function_experimental[_kernel_matmul_a_transposto_b](
                        ativ_prev_dev,
                        grad_z_dev,
                        grad_w_dev,
                        batch,
                        fan_in,
                        fan_out,
                        grid_dim=(grid_matmul_w),
                        block_dim=(block_dim),
                    )

                    var grid_sum_cols = (fan_out + block_dim - 1) // block_dim
                    ctx.enqueue_function_experimental[_kernel_somar_linhas](
                        grad_z_dev,
                        grad_b_dev,
                        batch,
                        fan_out,
                        grid_dim=(grid_sum_cols),
                        block_dim=(block_dim),
                    )

                    if calcular_grad_a_prev:
                        var grid_matmul_a = (len_grad_a_prev + block_dim - 1) // block_dim
                        ctx.enqueue_function_experimental[_kernel_matmul_b_transposto](
                            grad_z_dev,
                            peso_dev,
                            grad_a_prev_dev,
                            batch,
                            fan_out,
                            fan_in,
                            grid_dim=(grid_matmul_a),
                            block_dim=(block_dim),
                        )

                        var grid_relu = (len_grad_a_prev + block_dim - 1) // block_dim
                        ctx.enqueue_function_experimental[_kernel_derivada_relu_local](
                            ativ_prev_dev,
                            grad_a_prev_dev,
                            grad_a_prev_dev,
                            len_grad_a_prev,
                            grid_dim=(grid_relu),
                            block_dim=(block_dim),
                        )

                        var grid_copy_grad = (len_grad_a_prev + block_dim - 1) // block_dim
                        ctx.enqueue_function_experimental[_kernel_copy](
                            grad_a_prev_dev,
                            grad_z_dev,
                            len_grad_a_prev,
                            grid_dim=(grid_copy_grad),
                            block_dim=(block_dim),
                        )

                        len_grad_z_atual = len_grad_a_prev
                    else:
                        ctx.synchronize()

                    _copiar_device_para_tensor(grad_w_dev, grad_ws_out[camada], len_grad_w)
                    _copiar_device_para_tensor(grad_b_dev, grad_bs_out[camada], len_grad_b)
                    grad_ws_out[camada].id_pipeline_ultima_operacao = pipeline_id
                    grad_bs_out[camada].id_pipeline_ultima_operacao = pipeline_id

                    var formato_peso = peso_camada.formato.copy()
                    var peso_atualizado = tensor_defs.Tensor(formato_peso^, peso_camada.tipo_computacao)
                    _copiar_device_para_tensor(peso_upd_dev, peso_atualizado, len_peso)
                    peso_atualizado.id_pipeline_ultima_operacao = pipeline_id
                    pesos[camada] = peso_atualizado.copy()

                    var formato_bias = bias_camada.formato.copy()
                    var bias_atualizado = tensor_defs.Tensor(formato_bias^, bias_camada.tipo_computacao)
                    _copiar_device_para_tensor(bias_upd_dev, bias_atualizado, len_grad_b)
                    bias_atualizado.id_pipeline_ultima_operacao = pipeline_id
                    biases[camada] = bias_atualizado.copy()
        except _:
            debug_assert(False, "falha ao executar passo_backprop_mlp_cuda_em_tensores")
    else:
        debug_assert(False, "kernels CUDA nao disponiveis nesta compilacao")

    grad_w_out.id_pipeline_ultima_operacao = pipeline_id
    grad_b_out.id_pipeline_ultima_operacao = pipeline_id
    if calcular_grad_a_prev:
        grad_a_prev_out.id_pipeline_ultima_operacao = pipeline_id


fn backward_mlp_cuda_em_tensores(
    ativacoes_forward: List[tensor_defs.Tensor],
    zs_forward: List[tensor_defs.Tensor],
    pesos: List[tensor_defs.Tensor],
    grad_z_saida: tensor_defs.Tensor,
    mut grad_ws_out: List[tensor_defs.Tensor],
    mut grad_bs_out: List[tensor_defs.Tensor],
    mut grad_a_prev_out: List[tensor_defs.Tensor],
    var pool_max_len_ativ_prev: Int,
    var pool_max_len_grad_z: Int,
    var pool_max_len_grad_w: Int,
    var pool_max_len_grad_b: Int,
    var pool_max_len_peso: Int,
    var pool_max_len_grad_a_prev: Int,
    var pipeline_id: Int,
    var copiar_grad_a_prev_para_host: Bool = False,
):
    var num_camadas = len(pesos)
    debug_assert(num_camadas > 0, "MLP precisa de ao menos uma camada")
    debug_assert(len(ativacoes_forward) == num_camadas + 1, "ativacoes_forward inconsistente")
    debug_assert(len(zs_forward) == num_camadas, "zs_forward inconsistente")
    debug_assert(len(grad_ws_out) == num_camadas and len(grad_bs_out) == num_camadas and len(grad_a_prev_out) == num_camadas, "buffers de saída inconsistentes")

    var grad_z_atual = grad_z_saida.copy()

    @parameter
    if has_nvidia_gpu_accelerator():
        try:
            with DeviceContext(0, api="cuda") as ctx:
                var block_dim = 256

                # Reuso inter-iteracao via workspace: usa plano persistente de tamanhos maximos.
                var max_len_ativ_prev = pool_max_len_ativ_prev
                var max_len_grad_z = pool_max_len_grad_z
                var max_len_grad_w = pool_max_len_grad_w
                var max_len_grad_b = pool_max_len_grad_b
                var max_len_peso = pool_max_len_peso
                var max_len_grad_a_prev = pool_max_len_grad_a_prev

                if max_len_ativ_prev <= 0:
                    max_len_ativ_prev = 1
                if max_len_grad_z <= 0:
                    max_len_grad_z = len(grad_z_atual.dados)
                if max_len_grad_w <= 0:
                    max_len_grad_w = 1
                if max_len_grad_b <= 0:
                    max_len_grad_b = 1
                if max_len_peso <= 0:
                    max_len_peso = 1
                if max_len_grad_a_prev <= 0:
                    max_len_grad_a_prev = 1

                var ativ_prev_dev = ctx.enqueue_create_buffer[DType.float32](max_len_ativ_prev)
                var grad_z_dev = ctx.enqueue_create_buffer[DType.float32](max_len_grad_z)
                var grad_w_dev = ctx.enqueue_create_buffer[DType.float32](max_len_grad_w)
                var grad_b_dev = ctx.enqueue_create_buffer[DType.float32](max_len_grad_b)
                var peso_dev = buffer_pool.device_buffer_pool.acquire(len_peso)
                var grad_a_prev_dev = buffer_pool.device_buffer_pool.acquire(len_grad_a_prev)
                var bias_dev = ctx.enqueue_create_buffer[DType.float32](max_len_grad_b)
                var peso_upd_dev = ctx.enqueue_create_buffer[DType.float32](max_len_peso)
                var bias_upd_dev = ctx.enqueue_create_buffer[DType.float32](max_len_grad_b)

                var len_grad_z_atual = len(grad_z_atual.dados)
                _copiar_lista_para_device(grad_z_dev, grad_z_atual.dados, len_grad_z_atual)

                for passo in range(num_camadas):
                    var camada = num_camadas - 1 - passo
                    var ativ_prev = ativacoes_forward[camada]
                    var peso_camada = pesos[camada]
                    var bias_camada = grad_bs_out[camada]  # Usar grad_bs_out como referência para bias_camada
                    var calcular_grad_a_prev = camada > 0

                    var batch = ativ_prev.formato[0]
                    var fan_in = ativ_prev.formato[1]
                    var fan_out = peso_camada.formato[1]

                    var len_ativ_prev = len(ativ_prev.dados)
                    var len_grad_w = len(grad_ws_out[camada].dados)
                    var len_grad_b = len(grad_bs_out[camada].dados)
                    var len_peso = len(peso_camada.dados)
                    var len_grad_a_prev = len(grad_a_prev_out[camada].dados)

                    _copiar_lista_para_device(ativ_prev_dev, ativ_prev.dados, len_ativ_prev)
                    _copiar_lista_para_device(peso_dev, peso_camada.dados, len_peso)
                    _copiar_lista_para_device(bias_dev, bias_camada.dados, len_grad_b)

                    var grid_matmul_w = (len_grad_w + block_dim - 1) // block_dim
                    ctx.enqueue_function_experimental[_kernel_matmul_a_transposto_b](
                        ativ_prev_dev,
                        grad_z_dev,
                        grad_w_dev,
                        batch,
                        fan_in,
                        fan_out,
                        grid_dim=(grid_matmul_w),
                        block_dim=(block_dim),
                    )

                    var grid_sum_cols = (fan_out + block_dim - 1) // block_dim
                    ctx.enqueue_function_experimental[_kernel_somar_linhas](
                        grad_z_dev,
                        grad_b_dev,
                        batch,
                        fan_out,
                        grid_dim=(grid_sum_cols),
                        block_dim=(block_dim),
                    )

                    if calcular_grad_a_prev:
                        var grid_matmul_a = (len_grad_a_prev + block_dim - 1) // block_dim
                        ctx.enqueue_function_experimental[_kernel_matmul_b_transposto](
                            grad_z_dev,
                            peso_dev,
                            grad_a_prev_dev,
                            batch,
                            fan_out,
                            fan_in,
                            grid_dim=(grid_matmul_a),
                            block_dim=(block_dim),
                        )

                        var grid_relu = (len_grad_a_prev + block_dim - 1) // block_dim
                        ctx.enqueue_function_experimental[_kernel_derivada_relu_local](
                            ativ_prev_dev,
                            grad_a_prev_dev,
                            grad_a_prev_dev,
                            len_grad_a_prev,
                            grid_dim=(grid_relu),
                            block_dim=(block_dim),
                        )

                        var grid_copy_grad = (len_grad_a_prev + block_dim - 1) // block_dim
                        ctx.enqueue_function_experimental[_kernel_copy](
                            grad_a_prev_dev,
                            grad_z_dev,
                            len_grad_a_prev,
                            grid_dim=(grid_copy_grad),
                            block_dim=(block_dim),
                        )

                        len_grad_z_atual = len_grad_a_prev
                    else:
                        ctx.synchronize()

                    _copiar_device_para_tensor(grad_w_dev, grad_ws_out[camada], len_grad_w)
                    _copiar_device_para_tensor(grad_b_dev, grad_bs_out[camada], len_grad_b)
                    grad_ws_out[camada].id_pipeline_ultima_operacao = pipeline_id
                    grad_bs_out[camada].id_pipeline_ultima_operacao = pipeline_id

                    # Remover atualização de pesos/biases aqui, pois não faz sentido em backward puro
        except _:
            debug_assert(False, "falha ao executar backward_mlp_cuda_em_tensores")
    else:
        debug_assert(False, "kernels CUDA nao disponiveis nesta compilacao")


fn backward_mlp_cuda_em_tensores_com_sgd(
    ativacoes_forward: List[tensor_defs.Tensor],
    zs_forward: List[tensor_defs.Tensor],
    mut pesos: List[tensor_defs.Tensor],
    mut biases: List[tensor_defs.Tensor],
    grad_z_saida: tensor_defs.Tensor,
    mut grad_ws_out: List[tensor_defs.Tensor],
    mut grad_bs_out: List[tensor_defs.Tensor],
    mut grad_a_prev_out: List[tensor_defs.Tensor],
    var pool_max_len_ativ_prev: Int,
    var pool_max_len_grad_z: Int,
    var pool_max_len_grad_w: Int,
    var pool_max_len_grad_b: Int,
    var pool_max_len_peso: Int,
    var pool_max_len_grad_a_prev: Int,
    var taxa_aprendizado: Float32,
    var copiar_grad_a_prev_para_host: Bool = False,
    var pipeline_id: Int,
):
    var num_camadas = len(pesos)
    debug_assert(num_camadas > 0, "MLP precisa de ao menos uma camada")
    debug_assert(len(biases) == num_camadas, "biases inconsistente")
    debug_assert(len(ativacoes_forward) == num_camadas + 1, "ativacoes_forward inconsistente")
    debug_assert(len(zs_forward) == num_camadas, "zs_forward inconsistente")
    debug_assert(len(grad_ws_out) == num_camadas and len(grad_bs_out) == num_camadas and len(grad_a_prev_out) == num_camadas, "buffers de saída inconsistentes")

    var grad_z_atual = grad_z_saida.copy()

    @parameter
    if has_nvidia_gpu_accelerator():
        try:
            with DeviceContext(0, api="cuda") as ctx:
                var block_dim = 256

                var max_len_ativ_prev = pool_max_len_ativ_prev
                var max_len_grad_z = pool_max_len_grad_z
                var max_len_grad_w = pool_max_len_grad_w
                var max_len_grad_b = pool_max_len_grad_b
                var max_len_peso = pool_max_len_peso
                var max_len_grad_a_prev = pool_max_len_grad_a_prev

                if max_len_ativ_prev <= 0:
                    max_len_ativ_prev = 1
                if max_len_grad_z <= 0:
                    max_len_grad_z = len(grad_z_atual.dados)
                if max_len_grad_w <= 0:
                    max_len_grad_w = 1
                if max_len_grad_b <= 0:
                    max_len_grad_b = 1
                if max_len_peso <= 0:
                    max_len_peso = 1
                if max_len_grad_a_prev <= 0:
                    max_len_grad_a_prev = 1

                var ativ_prev_dev = ctx.enqueue_create_buffer[DType.float32](max_len_ativ_prev)
                var grad_z_dev = ctx.enqueue_create_buffer[DType.float32](max_len_grad_z)
                var grad_w_dev = ctx.enqueue_create_buffer[DType.float32](max_len_grad_w)
                var grad_b_dev = ctx.enqueue_create_buffer[DType.float32](max_len_grad_b)
                var peso_dev = buffer_pool.device_buffer_pool.acquire(len_peso)
                var bias_dev = ctx.enqueue_create_buffer[DType.float32](max_len_grad_b)
                var peso_upd_dev = ctx.enqueue_create_buffer[DType.float32](max_len_peso)
                var bias_upd_dev = ctx.enqueue_create_buffer[DType.float32](max_len_grad_b)
                var grad_a_prev_dev = ctx.enqueue_create_buffer[DType.float32](max_len_grad_a_prev)

                var len_grad_z_atual = len(grad_z_atual.dados)
                _copiar_lista_para_device(grad_z_dev, grad_z_atual.dados, len_grad_z_atual)

                for passo in range(num_camadas):
                    var camada = num_camadas - 1 - passo
                    var ativ_prev = ativacoes_forward[camada]
                    var peso_camada = pesos[camada]
                    var bias_camada = biases[camada]
                    var calcular_grad_a_prev = camada > 0

                    var batch = ativ_prev.formato[0]
                    var fan_in = ativ_prev.formato[1]
                    var fan_out = peso_camada.formato[1]

                    var len_ativ_prev = len(ativ_prev.dados)
                    var len_grad_w = len(grad_ws_out[camada].dados)
                    var len_grad_b = len(grad_bs_out[camada].dados)
                    var len_peso = len(peso_camada.dados)
                    var len_grad_a_prev = len(grad_a_prev_out[camada].dados)

                    _copiar_lista_para_device(ativ_prev_dev, ativ_prev.dados, len_ativ_prev)
                    _copiar_lista_para_device(peso_dev, peso_camada.dados, len_peso)
                    _copiar_lista_para_device(bias_dev, bias_camada.dados, len_grad_b)

                    var grid_matmul_w = (len_grad_w + block_dim - 1) // block_dim
                    ctx.enqueue_function_experimental[_kernel_matmul_a_transposto_b](
                        ativ_prev_dev,
                        grad_z_dev,
                        grad_w_dev,
                        batch,
                        fan_in,
                        fan_out,
                        grid_dim=(grid_matmul_w),
                        block_dim=(block_dim),
                    )

                    var grid_sum_cols = (fan_out + block_dim - 1) // block_dim
                    ctx.enqueue_function_experimental[_kernel_somar_linhas](
                        grad_z_dev,
                        grad_b_dev,
                        batch,
                        fan_out,
                        grid_dim=(grid_sum_cols),
                        block_dim=(block_dim),
                    )

                    if calcular_grad_a_prev:
                        var grid_matmul_a = (len_grad_a_prev + block_dim - 1) // block_dim
                        ctx.enqueue_function_experimental[_kernel_matmul_b_transposto](
                            grad_z_dev,
                            peso_dev,
                            grad_a_prev_dev,
                            batch,
                            fan_out,
                            fan_in,
                            grid_dim=(grid_matmul_a),
                            block_dim=(block_dim),
                        )

                        var grid_relu = (len_grad_a_prev + block_dim - 1) // block_dim
                        ctx.enqueue_function_experimental[_kernel_derivada_relu_local](
                            ativ_prev_dev,
                            grad_a_prev_dev,
                            grad_a_prev_dev,
                            len_grad_a_prev,
                            grid_dim=(grid_relu),
                            block_dim=(block_dim),
                        )

                        var grid_copy_grad = (len_grad_a_prev + block_dim - 1) // block_dim
                        ctx.enqueue_function_experimental[_kernel_copy](
                            grad_a_prev_dev,
                            grad_z_dev,
                            len_grad_a_prev,
                            grid_dim=(grid_copy_grad),
                            block_dim=(block_dim),
                        )

                        len_grad_z_atual = len_grad_a_prev

                    var grid_sgd_w = (len_peso + block_dim - 1) // block_dim
                    ctx.enqueue_function_experimental[_kernel_sgd_step](
                        peso_dev,
                        grad_w_dev,
                        peso_upd_dev,
                        len_peso,
                        taxa_aprendizado,
                        grid_dim=(grid_sgd_w),
                        block_dim=(block_dim),
                    )

                    var grid_sgd_b = (len_grad_b + block_dim - 1) // block_dim
                    ctx.enqueue_function_experimental[_kernel_sgd_step](
                        bias_dev,
                        grad_b_dev,
                        bias_upd_dev,
                        len_grad_b,
                        taxa_aprendizado,
                        grid_dim=(grid_sgd_b),
                        block_dim=(block_dim),
                    )

                    ctx.synchronize()

                    if copiar_grad_a_prev_para_host and calcular_grad_a_prev:
                        _copiar_device_para_tensor(grad_a_prev_dev, grad_a_prev_out[camada], len_grad_a_prev)
                        grad_a_prev_out[camada].id_pipeline_ultima_operacao = pipeline_id

                    _copiar_device_para_tensor(grad_w_dev, grad_ws_out[camada], len_grad_w)
                    _copiar_device_para_tensor(grad_b_dev, grad_bs_out[camada], len_grad_b)
                    grad_ws_out[camada].id_pipeline_ultima_operacao = pipeline_id
                    grad_bs_out[camada].id_pipeline_ultima_operacao = pipeline_id

                    var formato_peso = peso_camada.formato.copy()
                    var peso_atualizado = tensor_defs.Tensor(formato_peso^, peso_camada.tipo_computacao)
                    _copiar_device_para_tensor(peso_upd_dev, peso_atualizado, len_peso)
                    peso_atualizado.id_pipeline_ultima_operacao = pipeline_id
                    pesos[camada] = peso_atualizado.copy()

                    var formato_bias = bias_camada.formato.copy()
                    var bias_atualizado = tensor_defs.Tensor(formato_bias^, bias_camada.tipo_computacao)
                    _copiar_device_para_tensor(bias_upd_dev, bias_atualizado, len_grad_b)
                    bias_atualizado.id_pipeline_ultima_operacao = pipeline_id
                    biases[camada] = bias_atualizado.copy()
        except _:
            debug_assert(False, "falha ao executar backward_mlp_cuda_em_tensores_com_sgd")
    else:
        debug_assert(False, "kernels CUDA nao disponiveis nesta compilacao")
