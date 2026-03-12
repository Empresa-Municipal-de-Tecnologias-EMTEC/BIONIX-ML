import src.nucleo.Tensor as tensor_defs
from gpu import global_idx
from gpu.host import DeviceContext, DeviceBuffer
from sys import has_nvidia_gpu_accelerator
from math import exp


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
    var len_w_t = len_peso
    var len_grad_a_prev = len(grad_a_prev.dados)

    @parameter
    if has_nvidia_gpu_accelerator():
        try:
            with DeviceContext(0, api="cuda") as ctx:
                var ativ_prev_dev = ctx.enqueue_create_buffer[DType.float32](len_ativ_prev)
                var grad_z_dev = ctx.enqueue_create_buffer[DType.float32](len_grad_z)
                _copiar_lista_para_device(ativ_prev_dev, ativ_prev.dados, len_ativ_prev)
                _copiar_lista_para_device(grad_z_dev, grad_z_atual.dados, len_grad_z)

                var ativ_prev_t_dev = ctx.enqueue_create_buffer[DType.float32](len_ativ_prev)
                var grad_w_dev = ctx.enqueue_create_buffer[DType.float32](len_grad_w)
                var grad_b_dev = ctx.enqueue_create_buffer[DType.float32](len_grad_b)

                var block_dim = 256

                var grid_t_ativ = (len_ativ_prev + block_dim - 1) // block_dim
                ctx.enqueue_function_experimental[_kernel_transpose](
                    ativ_prev_dev,
                    ativ_prev_t_dev,
                    batch,
                    fan_in,
                    grid_dim=(grid_t_ativ),
                    block_dim=(block_dim),
                )

                var grid_matmul_w = (len_grad_w + block_dim - 1) // block_dim
                ctx.enqueue_function_experimental[_kernel_matmul](
                    ativ_prev_t_dev,
                    grad_z_dev,
                    grad_w_dev,
                    fan_in,
                    batch,
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
                    var peso_dev = ctx.enqueue_create_buffer[DType.float32](len_peso)
                    var w_t_dev = ctx.enqueue_create_buffer[DType.float32](len_w_t)
                    var grad_a_prev_dev = ctx.enqueue_create_buffer[DType.float32](len_grad_a_prev)
                    _copiar_lista_para_device(peso_dev, peso_camada.dados, len_peso)

                    var grid_t_peso = (len_peso + block_dim - 1) // block_dim
                    ctx.enqueue_function_experimental[_kernel_transpose](
                        peso_dev,
                        w_t_dev,
                        fan_in,
                        fan_out,
                        grid_dim=(grid_t_peso),
                        block_dim=(block_dim),
                    )

                    var grid_matmul_a = (len_grad_a_prev + block_dim - 1) // block_dim
                    ctx.enqueue_function_experimental[_kernel_matmul](
                        grad_z_dev,
                        w_t_dev,
                        grad_a_prev_dev,
                        batch,
                        fan_out,
                        fan_in,
                        grid_dim=(grid_matmul_a),
                        block_dim=(block_dim),
                    )

                    if aplicar_derivada_relu:
                        var z_prev_dev = ctx.enqueue_create_buffer[DType.float32](len_grad_a_prev)
                        _copiar_lista_para_device(z_prev_dev, z_derivada_relu.dados, len_grad_a_prev)
                        var grid_relu = (len_grad_a_prev + block_dim - 1) // block_dim
                        ctx.enqueue_function_experimental[_kernel_derivada_relu_local](
                            z_prev_dev,
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
    var len_w_t = len_peso
    var len_grad_a_prev = len(grad_a_prev_out.dados)

    @parameter
    if has_nvidia_gpu_accelerator():
        try:
            with DeviceContext(0, api="cuda") as ctx:
                var ativ_prev_dev = ctx.enqueue_create_buffer[DType.float32](len_ativ_prev)
                var grad_z_dev = ctx.enqueue_create_buffer[DType.float32](len_grad_z)
                _copiar_lista_para_device(ativ_prev_dev, ativ_prev.dados, len_ativ_prev)
                _copiar_lista_para_device(grad_z_dev, grad_z_atual.dados, len_grad_z)

                var ativ_prev_t_dev = ctx.enqueue_create_buffer[DType.float32](len_ativ_prev)
                var grad_w_dev = ctx.enqueue_create_buffer[DType.float32](len_grad_w)
                var grad_b_dev = ctx.enqueue_create_buffer[DType.float32](len_grad_b)

                var block_dim = 256

                var grid_t_ativ = (len_ativ_prev + block_dim - 1) // block_dim
                ctx.enqueue_function_experimental[_kernel_transpose](
                    ativ_prev_dev,
                    ativ_prev_t_dev,
                    batch,
                    fan_in,
                    grid_dim=(grid_t_ativ),
                    block_dim=(block_dim),
                )

                var grid_matmul_w = (len_grad_w + block_dim - 1) // block_dim
                ctx.enqueue_function_experimental[_kernel_matmul](
                    ativ_prev_t_dev,
                    grad_z_dev,
                    grad_w_dev,
                    fan_in,
                    batch,
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
                    var peso_dev = ctx.enqueue_create_buffer[DType.float32](len_peso)
                    var w_t_dev = ctx.enqueue_create_buffer[DType.float32](len_w_t)
                    var grad_a_prev_dev = ctx.enqueue_create_buffer[DType.float32](len_grad_a_prev)
                    _copiar_lista_para_device(peso_dev, peso_camada.dados, len_peso)

                    var grid_t_peso = (len_peso + block_dim - 1) // block_dim
                    ctx.enqueue_function_experimental[_kernel_transpose](
                        peso_dev,
                        w_t_dev,
                        fan_in,
                        fan_out,
                        grid_dim=(grid_t_peso),
                        block_dim=(block_dim),
                    )

                    var grid_matmul_a = (len_grad_a_prev + block_dim - 1) // block_dim
                    ctx.enqueue_function_experimental[_kernel_matmul](
                        grad_z_dev,
                        w_t_dev,
                        grad_a_prev_dev,
                        batch,
                        fan_out,
                        fan_in,
                        grid_dim=(grid_matmul_a),
                        block_dim=(block_dim),
                    )

                    if aplicar_derivada_relu:
                        var z_prev_dev = ctx.enqueue_create_buffer[DType.float32](len_grad_a_prev)
                        _copiar_lista_para_device(z_prev_dev, z_derivada_relu.dados, len_grad_a_prev)
                        var grid_relu = (len_grad_a_prev + block_dim - 1) // block_dim
                        ctx.enqueue_function_experimental[_kernel_derivada_relu_local](
                            z_prev_dev,
                            grad_a_prev_dev,
                            grad_a_prev_dev,
                            len_grad_a_prev,
                            grid_dim=(grid_relu),
                            block_dim=(block_dim),
                        )

                    ctx.synchronize()
                    _copiar_device_para_tensor(grad_a_prev_dev, grad_a_prev_out, len_grad_a_prev)
                else:
                    ctx.synchronize()

                _copiar_device_para_tensor(grad_w_dev, grad_w_out, len_grad_w)
                _copiar_device_para_tensor(grad_b_dev, grad_b_out, len_grad_b)
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
    var pipeline_id: Int,
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

                for passo in range(num_camadas):
                    var camada = num_camadas - 1 - passo
                    var ativ_prev = ativacoes_forward[camada]
                    var peso_camada = pesos[camada]
                    var calcular_grad_a_prev = camada > 0

                    var batch = ativ_prev.formato[0]
                    var fan_in = ativ_prev.formato[1]
                    var fan_out = grad_z_atual.formato[1]

                    var len_ativ_prev = len(ativ_prev.dados)
                    var len_grad_z = len(grad_z_atual.dados)
                    var len_grad_w = len(grad_ws_out[camada].dados)
                    var len_grad_b = len(grad_bs_out[camada].dados)
                    var len_peso = len(peso_camada.dados)
                    var len_grad_a_prev = len(grad_a_prev_out[camada].dados)

                    var ativ_prev_dev = ctx.enqueue_create_buffer[DType.float32](len_ativ_prev)
                    var grad_z_dev = ctx.enqueue_create_buffer[DType.float32](len_grad_z)
                    _copiar_lista_para_device(ativ_prev_dev, ativ_prev.dados, len_ativ_prev)
                    _copiar_lista_para_device(grad_z_dev, grad_z_atual.dados, len_grad_z)

                    var ativ_prev_t_dev = ctx.enqueue_create_buffer[DType.float32](len_ativ_prev)
                    var grad_w_dev = ctx.enqueue_create_buffer[DType.float32](len_grad_w)
                    var grad_b_dev = ctx.enqueue_create_buffer[DType.float32](len_grad_b)

                    var grid_t_ativ = (len_ativ_prev + block_dim - 1) // block_dim
                    ctx.enqueue_function_experimental[_kernel_transpose](
                        ativ_prev_dev,
                        ativ_prev_t_dev,
                        batch,
                        fan_in,
                        grid_dim=(grid_t_ativ),
                        block_dim=(block_dim),
                    )

                    var grid_matmul_w = (len_grad_w + block_dim - 1) // block_dim
                    ctx.enqueue_function_experimental[_kernel_matmul](
                        ativ_prev_t_dev,
                        grad_z_dev,
                        grad_w_dev,
                        fan_in,
                        batch,
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
                        var peso_dev = ctx.enqueue_create_buffer[DType.float32](len_peso)
                        var w_t_dev = ctx.enqueue_create_buffer[DType.float32](len_peso)
                        var grad_a_prev_dev = ctx.enqueue_create_buffer[DType.float32](len_grad_a_prev)
                        _copiar_lista_para_device(peso_dev, peso_camada.dados, len_peso)

                        var grid_t_peso = (len_peso + block_dim - 1) // block_dim
                        ctx.enqueue_function_experimental[_kernel_transpose](
                            peso_dev,
                            w_t_dev,
                            fan_in,
                            fan_out,
                            grid_dim=(grid_t_peso),
                            block_dim=(block_dim),
                        )

                        var grid_matmul_a = (len_grad_a_prev + block_dim - 1) // block_dim
                        ctx.enqueue_function_experimental[_kernel_matmul](
                            grad_z_dev,
                            w_t_dev,
                            grad_a_prev_dev,
                            batch,
                            fan_out,
                            fan_in,
                            grid_dim=(grid_matmul_a),
                            block_dim=(block_dim),
                        )

                        var z_anterior = zs_forward[camada - 1]
                        var z_prev_dev = ctx.enqueue_create_buffer[DType.float32](len_grad_a_prev)
                        _copiar_lista_para_device(z_prev_dev, z_anterior.dados, len_grad_a_prev)
                        var grid_relu = (len_grad_a_prev + block_dim - 1) // block_dim
                        ctx.enqueue_function_experimental[_kernel_derivada_relu_local](
                            z_prev_dev,
                            grad_a_prev_dev,
                            grad_a_prev_dev,
                            len_grad_a_prev,
                            grid_dim=(grid_relu),
                            block_dim=(block_dim),
                        )

                        ctx.synchronize()
                        _copiar_device_para_tensor(grad_a_prev_dev, grad_a_prev_out[camada], len_grad_a_prev)
                        grad_z_atual = grad_a_prev_out[camada].copy()
                    else:
                        ctx.synchronize()

                    _copiar_device_para_tensor(grad_w_dev, grad_ws_out[camada], len_grad_w)
                    _copiar_device_para_tensor(grad_b_dev, grad_bs_out[camada], len_grad_b)
                    grad_ws_out[camada].id_pipeline_ultima_operacao = pipeline_id
                    grad_bs_out[camada].id_pipeline_ultima_operacao = pipeline_id
                    if calcular_grad_a_prev:
                        grad_a_prev_out[camada].id_pipeline_ultima_operacao = pipeline_id
        except _:
            debug_assert(False, "falha ao executar backward_mlp_cuda_em_tensores")
    else:
        debug_assert(False, "kernels CUDA nao disponiveis nesta compilacao")
