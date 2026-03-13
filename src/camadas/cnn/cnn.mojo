fn _conv2d_valid_relu_backward(
    var imagem: List[Float32],
    var altura: Int,
    var largura: Int,
    var kernel: tensor_defs.Tensor,
    var grad_saida: List[Float32],
    var kh: Int,
    var kw: Int
) -> List[Float32]:
    # Calcula gradiente do kernel dado gradiente da saída (dL/dY)
    var out_h = altura - kh + 1
    var out_w = largura - kw + 1
    var grad_kernel = List[Float32]()
    for _ in range(kh * kw):
        grad_kernel.append(0.0)
    for y in range(out_h):
        for x in range(out_w):
            # Só propaga se ativação passou pelo ReLU
            var acc: Float32 = 0.0
            for ky in range(kh):
                for kx in range(kw):
                    var iy = y + ky
                    var ix = x + kx
                    var v = imagem[iy * largura + ix]
                    var w = kernel.dados[ky * kw + kx]
                    acc = acc + v * w
            if acc > 0.0:
                for ky in range(kh):
                    for kx in range(kw):
                        var iy = y + ky
                        var ix = x + kx
                        var v = imagem[iy * largura + ix]
                        grad_kernel[ky * kw + kx] = grad_kernel[ky * kw + kx] + grad_saida[y * out_w + x] * v
    return grad_kernel^
import src.nucleo.Tensor as tensor_defs
import src.computacao.dispatcher_tensor as dispatcher_tensor
import math


struct BlocoCNN(Movable, Copyable):
    var altura: Int
    var largura: Int
    var kernel_h: Int
    var kernel_w: Int
    var num_filtros: Int
    var kernels: List[tensor_defs.Tensor]
    var peso_saida: tensor_defs.Tensor
    var bias_saida: tensor_defs.Tensor
    var tipo_computacao: String

    fn __init__(
        out self,
        var altura_in: Int,
        var largura_in: Int,
        var num_filtros_in: Int = 2,
        var kernel_h_in: Int = 3,
        var kernel_w_in: Int = 3,
        var tipo_computacao_in: String = "cpu",
    ):
        debug_assert(altura_in > 0 and largura_in > 0, "entrada CNN invalida")
        debug_assert(num_filtros_in > 0, "num_filtros deve ser > 0")
        debug_assert(kernel_h_in > 0 and kernel_w_in > 0, "kernel invalido")
        debug_assert(altura_in >= kernel_h_in and largura_in >= kernel_w_in, "kernel maior que entrada")

        self.altura = altura_in
        self.largura = largura_in
        self.kernel_h = kernel_h_in
        self.kernel_w = kernel_w_in
        self.num_filtros = num_filtros_in
        self.tipo_computacao = tipo_computacao_in^
        self.kernels = List[tensor_defs.Tensor]()

        var conv_h = self.altura - self.kernel_h + 1
        var conv_w = self.largura - self.kernel_w + 1
        var pool_h = conv_h // 2
        var pool_w = conv_w // 2
        debug_assert(pool_h > 0 and pool_w > 0, "saida de pooling invalida")
        var feat_dim = self.num_filtros * pool_h * pool_w

        var formato_saida = List[Int]()

            # --- Backpropagação para os kernels convolucionais ---
            # Para cada filtro, computa gradiente do kernel
            var amostras = entradas.formato[0]
            var conv_h = bloco.altura - bloco.kernel_h + 1
            var conv_w = bloco.largura - bloco.kernel_w + 1
            var pool_h = conv_h // 2
            var pool_w = conv_w // 2
            var feat_dim = bloco.num_filtros * pool_h * pool_w

            # grad_feats: [N, feat_dim]
            # grad_feats = grad_z * W^T
            var grad_feats = dispatcher_tensor.multiplicar_matrizes(grad_z, dispatcher_tensor.transpor(bloco.peso_saida))

            for n in range(amostras):
                var img = List[Float32]()
                var base = n * bloco.altura * bloco.largura
                for i in range(bloco.altura * bloco.largura):
                    img.append(entradas.dados[base + i])

                var off = 0
                for f in range(bloco.num_filtros):
                    # grad_feat para este filtro
                    var grad_feat_f = List[Float32]()
                    for i in range(pool_h * pool_w):
                        grad_feat_f.append(grad_feats.dados[n * feat_dim + off + i])
                    # Desfaz pooling (aproximação: distribui gradiente igualmente)
                    var grad_conv = List[Float32]()
                    for y in range(conv_h):
                        for x in range(conv_w):
                            var pool_y = y // 2
                            var pool_x = x // 2
                            grad_conv.append(grad_feat_f[pool_y * pool_w + pool_x] * 0.25)
                    # grad_kernel para este filtro nesta amostra
                    var grad_kernel = _conv2d_valid_relu_backward(img, bloco.altura, bloco.largura, bloco.kernels[f], grad_conv, bloco.kernel_h, bloco.kernel_w)
                    # Atualiza kernel (SGD)
                    for i in range(len(grad_kernel)):
                        bloco.kernels[f].dados[i] = bloco.kernels[f].dados[i] - taxa_aprendizado * grad_kernel[i]
                    off = off + pool_h * pool_w
        formato_saida.append(feat_dim)
        formato_saida.append(1)
        self.peso_saida = tensor_defs.Tensor(formato_saida^, self.tipo_computacao)

        var formato_bias = List[Int]()
        formato_bias.append(1)
        formato_bias.append(1)
        self.bias_saida = tensor_defs.Tensor(formato_bias^, self.tipo_computacao)

        var seed = 991
        for f in range(self.num_filtros):
            var formato_k = List[Int]()
            formato_k.append(self.kernel_h)
            formato_k.append(self.kernel_w)
            var k = tensor_defs.Tensor(formato_k^, self.tipo_computacao)
            for i in range(len(k.dados)):
                seed = (seed * 1664525 + 1013904223 + i + f * 17) % 2147483647
                var u = Float32(seed) / Float32(2147483647)
                k.dados[i] = (u * 2.0 - 1.0) * 0.2
            self.kernels.append(k.copy())

        for i in range(len(self.peso_saida.dados)):
            seed = (seed * 1103515245 + 12345 + i) % 2147483647
            var u = Float32(seed) / Float32(2147483647)
            self.peso_saida.dados[i] = (u * 2.0 - 1.0) * 0.1
        self.bias_saida.dados[0] = 0.0

    fn copy(self) -> BlocoCNN:
        var out = BlocoCNN(
            self.altura,
            self.largura,
            self.num_filtros,
            self.kernel_h,
            self.kernel_w,
            self.tipo_computacao,
        )
        out.kernels = List[tensor_defs.Tensor]()
        for k in self.kernels:
            out.kernels.append(k.copy())
        out.peso_saida = self.peso_saida.copy()
        out.bias_saida = self.bias_saida.copy()
        return out^


fn _relu(var x: Float32) -> Float32:
    if x > 0.0:
        return x
    return 0.0


fn _hard_sigmoid(var x: Float32) -> Float32:
    var y = 0.2 * x + 0.5
    if y < 0.0:
        return 0.0
    if y > 1.0:
        return 1.0
    return y


fn _conv2d_valid_relu(
    var imagem: List[Float32],
    var altura: Int,
    var largura: Int,
    kernel: tensor_defs.Tensor,
    tipo_computacao: String = "cpu",
) -> List[Float32]:
    var kh = kernel.formato[0]
    var kw = kernel.formato[1]
    # Usa dispatcher para ativar CUDA se disponível
    return dispatcher_tensor.conv2d_valid_relu_dispatch(imagem, altura, largura, kernel.dados, kh, kw, tipo_computacao)


fn _avgpool2x2_stride2(var entrada: List[Float32], var h: Int, var w: Int, tipo_computacao: String = "cpu") -> List[Float32]:
    return dispatcher_tensor.avgpool2x2_stride2_dispatch(entrada, h, w, tipo_computacao)


fn extrair_features(bloco: BlocoCNN, entradas: tensor_defs.Tensor) -> tensor_defs.Tensor:
    debug_assert(len(entradas.formato) == 2, "entradas deve ser [N, H*W]")
    debug_assert(entradas.formato[1] == bloco.altura * bloco.largura, "features de entrada incompativeis")

    var amostras = entradas.formato[0]
    var conv_h = bloco.altura - bloco.kernel_h + 1
    var conv_w = bloco.largura - bloco.kernel_w + 1
    var pool_h = conv_h // 2
    var pool_w = conv_w // 2
    var feat_dim = bloco.num_filtros * pool_h * pool_w

    var formato = List[Int]()
    formato.append(amostras)
    formato.append(feat_dim)
    var out = tensor_defs.Tensor(formato^, bloco.tipo_computacao)

    for n in range(amostras):
        var img = List[Float32]()
        var base = n * bloco.altura * bloco.largura
        for i in range(bloco.altura * bloco.largura):
            img.append(entradas.dados[base + i])

        var off = 0
        for f in range(bloco.num_filtros):
            var conv = _conv2d_valid_relu(img, bloco.altura, bloco.largura, bloco.kernels[f], bloco.tipo_computacao)
            var pool = _avgpool2x2_stride2(conv, conv_h, conv_w, bloco.tipo_computacao)
            for i in range(len(pool)):
                out.dados[n * feat_dim + off + i] = pool[i]
            off = off + len(pool)

    return out^


fn prever(bloco: BlocoCNN, entradas: tensor_defs.Tensor) -> tensor_defs.Tensor:
    var feats = extrair_features(bloco, entradas)
    var z = dispatcher_tensor.multiplicar_matrizes(feats, bloco.peso_saida)
    z = dispatcher_tensor.adicionar_bias_coluna(z, bloco.bias_saida)

    var formato = z.formato.copy()
    var out = tensor_defs.Tensor(formato^, bloco.tipo_computacao)
    for i in range(len(z.dados)):
        out.dados[i] = _hard_sigmoid(z.dados[i])
    return out^


fn inferir(bloco: BlocoCNN, entradas: tensor_defs.Tensor) -> tensor_defs.Tensor:
    return prever(bloco, entradas)


fn treinar(
    mut bloco: BlocoCNN,
    entradas: tensor_defs.Tensor,
    alvos: tensor_defs.Tensor,
    var taxa_aprendizado: Float32 = 0.05,
    var epocas: Int = 300,
    var imprimir_cada: Int = 50,
) -> Float32:
    debug_assert(len(alvos.formato) == 2 and alvos.formato[1] == 1, "alvos deve ser [N,1]")
    debug_assert(entradas.formato[0] == alvos.formato[0], "N de entradas e alvos deve bater")

    var loss_final: Float32 = 0.0
    for epoca in range(epocas):
        var feats = extrair_features(bloco, entradas)
        var z = dispatcher_tensor.multiplicar_matrizes(feats, bloco.peso_saida)
        z = dispatcher_tensor.adicionar_bias_coluna(z, bloco.bias_saida)

        var pred = tensor_defs.Tensor(z.formato.copy(), bloco.tipo_computacao)
        for i in range(len(z.dados)):
            pred.dados[i] = _hard_sigmoid(z.dados[i])

        var grad_pred = dispatcher_tensor.gradiente_mse(pred, alvos)
        loss_final = dispatcher_tensor.erro_quadratico_medio_escalar(pred, alvos)

        var grad_z = tensor_defs.Tensor(z.formato.copy(), bloco.tipo_computacao)
        for i in range(len(z.dados)):
            var d: Float32 = 0.0
            if z.dados[i] > -2.5 and z.dados[i] < 2.5:
                d = 0.2
            grad_z.dados[i] = grad_pred.dados[i] * d

        var ft = dispatcher_tensor.transpor(feats)
        var grad_w = dispatcher_tensor.multiplicar_matrizes(ft, grad_z)
        var grad_b = dispatcher_tensor.soma_total(grad_z)

        for i in range(len(bloco.peso_saida.dados)):
            bloco.peso_saida.dados[i] = bloco.peso_saida.dados[i] - taxa_aprendizado * grad_w.dados[i]
        bloco.bias_saida.dados[0] = bloco.bias_saida.dados[0] - taxa_aprendizado * grad_b

        if imprimir_cada > 0 and (epoca % imprimir_cada == 0 or epoca == epocas - 1):
            print("Epoca", epoca, "| MSE:", loss_final)

    return loss_final
