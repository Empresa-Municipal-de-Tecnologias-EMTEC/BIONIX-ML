    var tmp_h_relu: tensor_defs.Tensor
    var tmp_h1: tensor_defs.Tensor
    var tmp_norm1_ffn: tensor_defs.Tensor
    var tmp_norm1_total: tensor_defs.Tensor
    var formato_tmp = List[Int]()
    formato_tmp.append(dim)
    var tmp_h_relu = tensor_defs.Tensor(formato_tmp^, tipo)
    var tmp_h1 = tensor_defs.Tensor(formato_tmp^, tipo)
    var tmp_norm1_ffn = tensor_defs.Tensor(formato_tmp^, tipo)
    var tmp_norm1_total = tensor_defs.Tensor(formato_tmp^, tipo)
import src.computacao.kvcache_sessao as kvcache_sessao
import src.computacao.captura_camadas as captura_camadas
import src.computacao.dispatcher_tensor as dispatcher_tensor
import src.nucleo.Tensor as tensor_defs
import src.camadas.transformer.adaptadores_atencao as adaptadores_atencao
import math


struct BlocoEmbedding(Movable, Copyable):
    var vocab_size: Int
    var dimensao_modelo: Int
    var tabela: tensor_defs.Tensor

    fn __init__(out self, var vocab_size_in: Int, var dimensao_modelo_in: Int, var tipo_computacao: String = "cpu"):
        debug_assert(vocab_size_in > 0 and dimensao_modelo_in > 0, "embedding invalido")
        self.vocab_size = vocab_size_in
        self.dimensao_modelo = dimensao_modelo_in

        var formato = List[Int]()
        formato.append(vocab_size_in)
        formato.append(dimensao_modelo_in)
        self.tabela = tensor_defs.Tensor(formato^, tipo_computacao)

        var seed = 1337
        for i in range(len(self.tabela.dados)):
            seed = (seed * 1103515245 + 12345 + i) % 2147483647
            var u = Float32(seed) / Float32(2147483647)
            self.tabela.dados[i] = (u * 2.0 - 1.0) * 0.02

    fn copy(self) -> BlocoEmbedding:
        var out = BlocoEmbedding(self.vocab_size, self.dimensao_modelo, self.tabela.tipo_computacao)
        out.tabela = self.tabela.copy()
        return out^


fn embedding_forward(bloco: BlocoEmbedding, token_ids: List[Int]) -> tensor_defs.Tensor:
    var seq_len = len(token_ids)
    var formato = List[Int]()
    formato.append(seq_len)
    formato.append(bloco.dimensao_modelo)
    var out = tensor_defs.Tensor(formato^, bloco.tabela.tipo_computacao)

    for t in range(seq_len):
        var token = token_ids[t]
        if token < 0:
            token = 0
        if token >= bloco.vocab_size:
            token = bloco.vocab_size - 1
        var base_src = token * bloco.dimensao_modelo
        var base_dst = t * bloco.dimensao_modelo
        for d in range(bloco.dimensao_modelo):
            out.dados[base_dst + d] = bloco.tabela.dados[base_src + d]

    return out^


fn _linha_para_tensor(var origem: tensor_defs.Tensor, var linha: Int) -> tensor_defs.Tensor:
    var colunas = origem.formato[1]
    var formato = List[Int]()
    formato.append(1)
    formato.append(colunas)
    var out = tensor_defs.Tensor(formato^, origem.tipo_computacao)
    var base = linha * colunas
    for i in range(colunas):
        out.dados[i] = origem.dados[base + i]
    return out^


fn _linha_zeros(var colunas: Int, var tipo_computacao: String) -> tensor_defs.Tensor:
    var formato = List[Int]()
    formato.append(1)
    formato.append(colunas)
    return tensor_defs.Tensor(formato^, tipo_computacao)


fn _dot_head(
    q: tensor_defs.Tensor,
    k: tensor_defs.Tensor,
    var q_linha: Int,
    var inicio_head: Int,
    var head_dim: Int,
) -> Float32:
    var base_q = q_linha * q.formato[1] + inicio_head
    var base_k = inicio_head
    var acc: Float32 = 0.0
    for d in range(head_dim):
        acc = acc + q.dados[base_q + d] * k.dados[base_k + d]
    return acc


fn _acumular_head(
    mut saida: tensor_defs.Tensor,
    v: tensor_defs.Tensor,
    var saida_linha: Int,
    var inicio_head: Int,
    var head_dim: Int,
    var peso: Float32,
):
    var base_out = saida_linha * saida.formato[1] + inicio_head
    var base_v = inicio_head
    for d in range(head_dim):
        saida.dados[base_out + d] = saida.dados[base_out + d] + peso * v.dados[base_v + d]


fn _carregar_ou_fallback(
    provider: kvcache_sessao.KVCacheProvider,
    var camada_idx: Int,
    var posicao: Int,
    fallback: tensor_defs.Tensor,
    var eh_k: Bool,
) -> tensor_defs.Tensor:
    if eh_k:
        return kvcache_sessao.carregar_k_cache(provider, camada_idx, posicao, fallback)
    return kvcache_sessao.carregar_v_cache(provider, camada_idx, posicao, fallback)


fn atencao_causal_multihead_com_kvcache(
    x: tensor_defs.Tensor,
    w_q: tensor_defs.Tensor,
    w_k: tensor_defs.Tensor,
    w_v: tensor_defs.Tensor,
    w_o: tensor_defs.Tensor,
    var num_heads: Int,
    mut kvcache: kvcache_sessao.KVCacheProvider,
    adaptador_atencao: adaptadores_atencao.AdaptadorAtencao,
    var camada_idx: Int,
    var posicao_base: Int,
) -> tensor_defs.Tensor:
    debug_assert(len(x.formato) == 2, "atencao espera tensor 2D")
    var seq_len = x.formato[0]
    var dim = x.formato[1]
    debug_assert(dim > 0, "dimensao do modelo deve ser positiva")
    if num_heads <= 0:
        num_heads = 1
    debug_assert(dim % num_heads == 0, "dimensao deve ser multipla de num_heads")
    var head_dim = dim // num_heads

    var q = dispatcher_tensor.multiplicar_matrizes(x, w_q)
    var k_local = dispatcher_tensor.multiplicar_matrizes(x, w_k)
    var v_local = dispatcher_tensor.multiplicar_matrizes(x, w_v)

    for t in range(seq_len):
        var linha_k = _linha_para_tensor(k_local, t)
        var linha_v = _linha_para_tensor(v_local, t)
        _ = kvcache_sessao.salvar_k_cache(kvcache, camada_idx, posicao_base + t, linha_k)
        _ = kvcache_sessao.salvar_v_cache(kvcache, camada_idx, posicao_base + t, linha_v)

    var formato = List[Int]()
    formato.append(seq_len)
    formato.append(dim)
    var contexto = tensor_defs.Tensor(formato^, x.tipo_computacao)
    var escala = 1.0 / Float32(math.sqrt(Float64(head_dim)))
    var fallback_zeros = _linha_zeros(dim, x.tipo_computacao)

    for t in range(seq_len):
        var pos_abs = posicao_base + t
        for h in range(num_heads):
            var inicio_head = h * head_dim

            var max_score: Float32 = -1e30
            for pos in range(pos_abs + 1):
                var k_pos = _carregar_ou_fallback(kvcache, camada_idx, pos, fallback_zeros, True)
                var s = _dot_head(q, k_pos, t, inicio_head, head_dim) * escala
                if s > max_score:
                    max_score = s

            var soma_exp: Float32 = 0.0
            for pos in range(pos_abs + 1):
                var k_pos = _carregar_ou_fallback(kvcache, camada_idx, pos, fallback_zeros, True)
                var s = _dot_head(q, k_pos, t, inicio_head, head_dim) * escala
                soma_exp = soma_exp + Float32(math.exp(Float64(s - max_score)))

            if soma_exp <= 0.0:
                soma_exp = 1.0

            for pos in range(pos_abs + 1):
                var k_pos = _carregar_ou_fallback(kvcache, camada_idx, pos, fallback_zeros, True)
                var v_pos = _carregar_ou_fallback(kvcache, camada_idx, pos, fallback_zeros, False)
                var s = _dot_head(q, k_pos, t, inicio_head, head_dim) * escala
                var w = Float32(math.exp(Float64(s - max_score))) / soma_exp
                _acumular_head(contexto, v_pos, t, inicio_head, head_dim, w)

    var proj_out = dispatcher_tensor.multiplicar_matrizes(contexto, w_o)
    return adaptadores_atencao.aplicar_adaptador_atencao(proj_out, adaptador_atencao)


struct BlocoTransformerBase(Movable, Copyable):
    var embedding: BlocoEmbedding
    var num_heads: Int
    var w_q: tensor_defs.Tensor
    var w_k: tensor_defs.Tensor
    var w_v: tensor_defs.Tensor
    var w_o: tensor_defs.Tensor
    var w_ffn_1: tensor_defs.Tensor
    var b_ffn_1: tensor_defs.Tensor
    var w_ffn_2: tensor_defs.Tensor
    var b_ffn_2: tensor_defs.Tensor
    var layer_norm_epsilon: Float32

    fn __init__(
        out self,
        var vocab_size: Int,
        var dimensao_modelo: Int,
        var num_heads_in: Int = 4,
        var tipo_computacao: String = "cpu",
    ):
        self.embedding = BlocoEmbedding(vocab_size, dimensao_modelo, tipo_computacao)
        if num_heads_in <= 0:
            num_heads_in = 1
        self.num_heads = num_heads_in

        var formato_w = List[Int]()
        formato_w.append(dimensao_modelo)
        formato_w.append(dimensao_modelo)
        self.w_q = tensor_defs.Tensor(formato_w.copy(), tipo_computacao)
        self.w_k = tensor_defs.Tensor(formato_w.copy(), tipo_computacao)
        self.w_v = tensor_defs.Tensor(formato_w.copy(), tipo_computacao)
        self.w_o = tensor_defs.Tensor(formato_w^, tipo_computacao)

        var formato_bias = List[Int]()
        formato_bias.append(1)
        formato_bias.append(dimensao_modelo)
        self.w_ffn_1 = tensor_defs.Tensor(formato_w.copy(), tipo_computacao)
        self.b_ffn_1 = tensor_defs.Tensor(formato_bias.copy(), tipo_computacao)
        self.w_ffn_2 = tensor_defs.Tensor(formato_w.copy(), tipo_computacao)
        self.b_ffn_2 = tensor_defs.Tensor(formato_bias^, tipo_computacao)
        self.layer_norm_epsilon = 1e-5

        var seed = 2027
        for i in range(len(self.w_q.dados)):
            seed = (seed * 1664525 + 1013904223 + i) % 2147483647
            var u_q = Float32(seed) / Float32(2147483647)
            self.w_q.dados[i] = (u_q * 2.0 - 1.0) * 0.02

            seed = (seed * 1664525 + 1013904223 + i + 17) % 2147483647
            var u_k = Float32(seed) / Float32(2147483647)
            self.w_k.dados[i] = (u_k * 2.0 - 1.0) * 0.02

            seed = (seed * 1664525 + 1013904223 + i + 31) % 2147483647
            var u_v = Float32(seed) / Float32(2147483647)
            self.w_v.dados[i] = (u_v * 2.0 - 1.0) * 0.02

            seed = (seed * 1664525 + 1013904223 + i + 47) % 2147483647
            var u_o = Float32(seed) / Float32(2147483647)
            self.w_o.dados[i] = (u_o * 2.0 - 1.0) * 0.02

            seed = (seed * 1664525 + 1013904223 + i + 59) % 2147483647
            var u_f1 = Float32(seed) / Float32(2147483647)
            self.w_ffn_1.dados[i] = (u_f1 * 2.0 - 1.0) * 0.02

            seed = (seed * 1664525 + 1013904223 + i + 71) % 2147483647
            var u_f2 = Float32(seed) / Float32(2147483647)
            self.w_ffn_2.dados[i] = (u_f2 * 2.0 - 1.0) * 0.02

        for i in range(len(self.b_ffn_1.dados)):
            self.b_ffn_1.dados[i] = 0.0
            self.b_ffn_2.dados[i] = 0.0

    fn copy(self) -> BlocoTransformerBase:
        var out = BlocoTransformerBase(
            self.embedding.vocab_size,
            self.embedding.dimensao_modelo,
            self.num_heads,
            self.embedding.tabela.tipo_computacao,
        )
        out.embedding = self.embedding.copy()
        out.w_q = self.w_q.copy()
        out.w_k = self.w_k.copy()
        out.w_v = self.w_v.copy()
        out.w_o = self.w_o.copy()
        out.w_ffn_1 = self.w_ffn_1.copy()
        out.b_ffn_1 = self.b_ffn_1.copy()
        out.w_ffn_2 = self.w_ffn_2.copy()
        out.b_ffn_2 = self.b_ffn_2.copy()
        out.layer_norm_epsilon = self.layer_norm_epsilon
        return out^


struct ForwardTransformerCache(Movable, Copyable):
    var emb: tensor_defs.Tensor
    var attn: tensor_defs.Tensor
    var resid_1: tensor_defs.Tensor
    var norm_1: tensor_defs.Tensor
    var ffn_h1: tensor_defs.Tensor
    var ffn_h1_relu: tensor_defs.Tensor
    var ffn_out: tensor_defs.Tensor
    var resid_2: tensor_defs.Tensor
    var out: tensor_defs.Tensor

    fn __init__(
        out self,
        emb_in: tensor_defs.Tensor,
        attn_in: tensor_defs.Tensor,
        resid_1_in: tensor_defs.Tensor,
        norm_1_in: tensor_defs.Tensor,
        ffn_h1_in: tensor_defs.Tensor,
        ffn_h1_relu_in: tensor_defs.Tensor,
        ffn_out_in: tensor_defs.Tensor,
        resid_2_in: tensor_defs.Tensor,
        out_in: tensor_defs.Tensor,
    ):
        self.emb = emb_in^
        self.attn = attn_in^
        self.resid_1 = resid_1_in^
        self.norm_1 = norm_1_in^
        self.ffn_h1 = ffn_h1_in^
        self.ffn_h1_relu = ffn_h1_relu_in^
        self.ffn_out = ffn_out_in^
        self.resid_2 = resid_2_in^
        self.out = out_in^

    fn copy(self) -> ForwardTransformerCache:
        return ForwardTransformerCache(
            self.emb.copy(),
            self.attn.copy(),
            self.resid_1.copy(),
            self.norm_1.copy(),
            self.ffn_h1.copy(),
            self.ffn_h1_relu.copy(),
            self.ffn_out.copy(),
            self.resid_2.copy(),
            self.out.copy(),
        )


fn _relu_tensor(x: tensor_defs.Tensor) -> tensor_defs.Tensor:
    var out = x.copy()
    for i in range(len(out.dados)):
        if out.dados[i] < 0.0:
            out.dados[i] = 0.0
    return out^


fn _layer_norm_linhas(x: tensor_defs.Tensor, var epsilon: Float32 = 1e-5) -> tensor_defs.Tensor:
    debug_assert(len(x.formato) == 2, "layer_norm espera tensor 2D")
    var linhas = x.formato[0]
    var cols = x.formato[1]
    var out = x.copy()
    if cols <= 0:
        return out^

    for i in range(linhas):
        var base = i * cols
        var media: Float32 = 0.0
        for j in range(cols):
            media = media + x.dados[base + j]
        media = media / Float32(cols)

        var var_acc: Float32 = 0.0
        for j in range(cols):
            var d = x.dados[base + j] - media
            var_acc = var_acc + d * d
        var_acc = var_acc / Float32(cols)
        var inv_std = 1.0 / Float32(math.sqrt(Float64(var_acc + epsilon)))

        for j in range(cols):
            out.dados[base + j] = (x.dados[base + j] - media) * inv_std

    return out^


fn _ffn_transformer(bloco: BlocoTransformerBase, x: tensor_defs.Tensor) -> tensor_defs.Tensor:
    var h1 = dispatcher_tensor.multiplicar_matrizes(x, bloco.w_ffn_1)
    h1 = dispatcher_tensor.adicionar_bias_coluna(h1, bloco.b_ffn_1)
    h1 = _relu_tensor(h1)
    var h2 = dispatcher_tensor.multiplicar_matrizes(h1, bloco.w_ffn_2)
    h2 = dispatcher_tensor.adicionar_bias_coluna(h2, bloco.b_ffn_2)
    return h2^


fn _linha_tensor_para_lista(t: tensor_defs.Tensor, var linha: Int) -> List[Float32]:
    var cols = t.formato[1]
    var base = linha * cols
    var out = List[Float32]()
    for i in range(cols):
        out.append(t.dados[base + i])
    return out^


fn _escalar_max_1xn(mut t: tensor_defs.Tensor, var n: Int, var escala: Float32) -> Float32:
    if n <= 0:
        return 0.0
    var max_s = t.dados[0] * escala
    for i in range(n):
        t.dados[i] = t.dados[i] * escala
        if t.dados[i] > max_s:
            max_s = t.dados[i]
    return max_s


fn _mascarar_causal_1xn(mut t: tensor_defs.Tensor, var n_total: Int, var n_validos: Int, var valor_mascara: Float32 = -1e30):
    if n_validos < 0:
        n_validos = 0
    if n_validos > n_total:
        n_validos = n_total
    for i in range(n_validos, n_total):
        t.dados[i] = valor_mascara


fn _softmax_1xn_para_probs(mut scores_1xn: tensor_defs.Tensor, mut probs_1xn: tensor_defs.Tensor, var n: Int, var max_s: Float32):
    var soma_exp: Float32 = 0.0
    for i in range(n):
        var e = Float32(math.exp(Float64(scores_1xn.dados[i] - max_s)))
        probs_1xn.dados[i] = e
        soma_exp = soma_exp + e
    if soma_exp <= 0.0:
        soma_exp = 1.0
    for i in range(n):
        probs_1xn.dados[i] = probs_1xn.dados[i] / soma_exp


fn _transpor_para(src: tensor_defs.Tensor, mut dst: tensor_defs.Tensor):
    debug_assert(len(src.formato) == 2 and len(dst.formato) == 2, "_transpor_para espera tensores 2D")
    debug_assert(dst.formato[0] == src.formato[1] and dst.formato[1] == src.formato[0], "formato de destino invalido em _transpor_para")
    var linhas = src.formato[0]
    var cols = src.formato[1]
    for i in range(linhas):
        for j in range(cols):
            dst.dados[j * linhas + i] = src.dados[i * cols + j]


struct BackwardTransformerWorkspace(Movable, Copyable):
    var emb_prefix: tensor_defs.Tensor
    var emb_prefix_t: tensor_defs.Tensor
    var emb_last_col: tensor_defs.Tensor
    var context_last_t: tensor_defs.Tensor
    var grad_context_t: tensor_defs.Tensor
    var grad_emb_atencao: tensor_defs.Tensor
    var k_head: tensor_defs.Tensor
    var v_head: tensor_defs.Tensor
    var q_head_t: tensor_defs.Tensor
    var probs_t: tensor_defs.Tensor
    var grad_ctx_h_t: tensor_defs.Tensor
    var d_q_scaled_t: tensor_defs.Tensor
    var d_s_t: tensor_defs.Tensor
    var w_v_head_old_ht: tensor_defs.Tensor
    var w_k_head_old_ht: tensor_defs.Tensor
    var w_q_head_old_ht: tensor_defs.Tensor
    var k_head_t: tensor_defs.Tensor
    var probs_col: tensor_defs.Tensor
    var grad_ctx_h_col: tensor_defs.Tensor
    var d_s_col: tensor_defs.Tensor
    var w_q_old_t: tensor_defs.Tensor
    var w_k_old_t: tensor_defs.Tensor
    var w_v_old_t: tensor_defs.Tensor

    fn __init__(
        out self,
        emb_prefix_in: tensor_defs.Tensor,
        emb_prefix_t_in: tensor_defs.Tensor,
        emb_last_col_in: tensor_defs.Tensor,
        context_last_t_in: tensor_defs.Tensor,
        grad_context_t_in: tensor_defs.Tensor,
        grad_emb_atencao_in: tensor_defs.Tensor,
        k_head_in: tensor_defs.Tensor,
        v_head_in: tensor_defs.Tensor,
        q_head_t_in: tensor_defs.Tensor,
        probs_t_in: tensor_defs.Tensor,
        grad_ctx_h_t_in: tensor_defs.Tensor,
        d_q_scaled_t_in: tensor_defs.Tensor,
        d_s_t_in: tensor_defs.Tensor,
        w_v_head_old_ht_in: tensor_defs.Tensor,
        w_k_head_old_ht_in: tensor_defs.Tensor,
        w_q_head_old_ht_in: tensor_defs.Tensor,
        k_head_t_in: tensor_defs.Tensor,
        probs_col_in: tensor_defs.Tensor,
        grad_ctx_h_col_in: tensor_defs.Tensor,
        d_s_col_in: tensor_defs.Tensor,
        w_q_old_t_in: tensor_defs.Tensor,
        w_k_old_t_in: tensor_defs.Tensor,
        w_v_old_t_in: tensor_defs.Tensor,
    ):
        self.emb_prefix = emb_prefix_in^
        self.emb_prefix_t = emb_prefix_t_in^
        self.emb_last_col = emb_last_col_in^
        self.context_last_t = context_last_t_in^
        self.grad_context_t = grad_context_t_in^
        self.grad_emb_atencao = grad_emb_atencao_in^
        self.k_head = k_head_in^
        self.v_head = v_head_in^
        self.q_head_t = q_head_t_in^
        self.probs_t = probs_t_in^
        self.grad_ctx_h_t = grad_ctx_h_t_in^
        self.d_q_scaled_t = d_q_scaled_t_in^
        self.d_s_t = d_s_t_in^
        self.w_v_head_old_ht = w_v_head_old_ht_in^
        self.w_k_head_old_ht = w_k_head_old_ht_in^
        self.w_q_head_old_ht = w_q_head_old_ht_in^
        self.k_head_t = k_head_t_in^
        self.probs_col = probs_col_in^
        self.grad_ctx_h_col = grad_ctx_h_col_in^
        self.d_s_col = d_s_col_in^
        self.w_q_old_t = w_q_old_t_in^
        self.w_k_old_t = w_k_old_t_in^
        self.w_v_old_t = w_v_old_t_in^


fn _criar_workspace_backward_transformer(var seq_len: Int, var dim: Int, var head_dim: Int, var tipo: String) -> BackwardTransformerWorkspace:
    var formato_emb = List[Int]()
    formato_emb.append(seq_len)
    formato_emb.append(dim)
    var emb_prefix = tensor_defs.Tensor(formato_emb^, tipo)
    var formato_emb_t = List[Int]()
    formato_emb_t.append(dim)
    formato_emb_t.append(seq_len)
    var emb_prefix_t = tensor_defs.Tensor(formato_emb_t^, tipo)

    var formato_emb_last_col = List[Int]()
    formato_emb_last_col.append(dim)
    formato_emb_last_col.append(1)
    var emb_last_col = tensor_defs.Tensor(formato_emb_last_col^, tipo)

    var formato_ctx = List[Int]()
    formato_ctx.append(1)
    formato_ctx.append(dim)
    var context_last_t = tensor_defs.Tensor(formato_ctx.copy(), tipo)
    var grad_context_t = tensor_defs.Tensor(formato_ctx^, tipo)

    var formato_grad_emb = List[Int]()
    formato_grad_emb.append(seq_len)
    formato_grad_emb.append(dim)
    var grad_emb_atencao = tensor_defs.Tensor(formato_grad_emb^, tipo)

    var formato_head = List[Int]()
    formato_head.append(seq_len)
    formato_head.append(head_dim)
    var k_head = tensor_defs.Tensor(formato_head.copy(), tipo)
    var v_head = tensor_defs.Tensor(formato_head^, tipo)

    var formato_q = List[Int]()
    formato_q.append(1)
    formato_q.append(head_dim)
    var q_head_t = tensor_defs.Tensor(formato_q^, tipo)

    var formato_probs = List[Int]()
    formato_probs.append(1)
    formato_probs.append(seq_len)
    var probs_t = tensor_defs.Tensor(formato_probs^, tipo)

    var formato_grad_ctx = List[Int]()
    formato_grad_ctx.append(1)
    formato_grad_ctx.append(head_dim)
    var grad_ctx_h_t = tensor_defs.Tensor(formato_grad_ctx.copy(), tipo)
    var d_q_scaled_t = tensor_defs.Tensor(formato_grad_ctx.copy(), tipo)

    var formato_ds = List[Int]()
    formato_ds.append(1)
    formato_ds.append(seq_len)
    var d_s_t = tensor_defs.Tensor(formato_ds^, tipo)

    var formato_head_old_ht = List[Int]()
    formato_head_old_ht.append(head_dim)
    formato_head_old_ht.append(dim)
    var w_v_head_old_ht = tensor_defs.Tensor(formato_head_old_ht.copy(), tipo)
    var w_k_head_old_ht = tensor_defs.Tensor(formato_head_old_ht.copy(), tipo)
    var w_q_head_old_ht = tensor_defs.Tensor(formato_head_old_ht^, tipo)

    var formato_k_head_t = List[Int]()
    formato_k_head_t.append(head_dim)
    formato_k_head_t.append(seq_len)
    var k_head_t = tensor_defs.Tensor(formato_k_head_t^, tipo)

    var formato_probs_col = List[Int]()
    formato_probs_col.append(seq_len)
    formato_probs_col.append(1)
    var probs_col = tensor_defs.Tensor(formato_probs_col.copy(), tipo)
    var grad_ctx_h_col = tensor_defs.Tensor(formato_probs_col.copy(), tipo)
    var d_s_col = tensor_defs.Tensor(formato_probs_col^, tipo)

    var formato_w = List[Int]()
    formato_w.append(dim)
    formato_w.append(dim)
    var w_q_old_t = tensor_defs.Tensor(formato_w.copy(), tipo)
    var w_k_old_t = tensor_defs.Tensor(formato_w.copy(), tipo)
    var w_v_old_t = tensor_defs.Tensor(formato_w^, tipo)

    return BackwardTransformerWorkspace(
        emb_prefix,
        emb_prefix_t,
        emb_last_col,
        context_last_t,
        grad_context_t,
        grad_emb_atencao,
        k_head,
        v_head,
        q_head_t,
        probs_t,
        grad_ctx_h_t,
        d_q_scaled_t,
        d_s_t,
        w_v_head_old_ht,
        w_k_head_old_ht,
        w_q_head_old_ht,
        k_head_t,
        probs_col,
        grad_ctx_h_col,
        d_s_col,
        w_q_old_t,
        w_k_old_t,
        w_v_old_t,
        tmp_h_relu,
        tmp_h1,
        tmp_norm1_ffn,
        tmp_norm1_total,
    )


fn _aplicar_gradiente_saida_transformer_analitico_parcial_limite_com_emb_rows(
    mut bloco: BlocoTransformerBase,
    token_ids: List[Int],
    cache: ForwardTransformerCache,
    grad_saida_ultimo_token: List[Float32],
    var seq_limite: Int,
    mut workspace: BackwardTransformerWorkspace,
    var taxa_aprendizado: Float32 = 0.001,
):
    var dim = bloco.embedding.dimensao_modelo
    if dim <= 0 or len(grad_saida_ultimo_token) < dim:
        return
    if cache.out.formato[0] <= 0:
        return

    if seq_limite <= 0:
        return
    if seq_limite > cache.out.formato[0]:
        seq_limite = cache.out.formato[0]

    var linha = seq_limite - 1
    let cols = cache.resid_2.formato[1]
    let base = linha * cols
    let resid2_vec = cache.resid_2.dados[base : base + cols]
    let resid1_vec = cache.resid_1.dados[base : base + cols]
    let norm1_vec = cache.norm_1.dados[base : base + cols]
    let h1_pre_vec = cache.ffn_h1.dados[base : base + cols]
    let h1_relu_vec = cache.ffn_h1_relu.dados[base : base + cols]

    let grad_out = grad_saida_ultimo_token

    var grad_resid2 = _layer_norm_backward_linha(resid2_vec, grad_out, bloco.layer_norm_epsilon)
    var grad_norm1_skip = grad_resid2.copy()
    var grad_ffn_out = grad_resid2.copy()

    let grad_h_relu = workspace.tmp_h_relu.dados
    for i in range(dim):
        grad_h_relu[i] = 0.0

    for i in range(dim):
        var acc: Float32 = 0.0
        for j in range(dim):
            acc = acc + grad_ffn_out[j] * bloco.w_ffn_2.dados[i * dim + j]
        grad_h_relu[i] = acc

    for i in range(dim):
        for j in range(dim):
            var idx = i * dim + j
            bloco.w_ffn_2.dados[idx] = bloco.w_ffn_2.dados[idx] - taxa_aprendizado * h1_relu_vec[i] * grad_ffn_out[j]
    for j in range(dim):
        bloco.b_ffn_2.dados[j] = bloco.b_ffn_2.dados[j] - taxa_aprendizado * grad_ffn_out[j]

    let grad_h1 = workspace.tmp_h1.dados
    for i in range(dim):
        var g = grad_h_relu[i]
        if h1_pre_vec[i] <= 0.0:
            g = 0.0
        grad_h1[i] = g

    let grad_norm1_ffn = workspace.tmp_norm1_ffn.dados
    for i in range(dim):
        grad_norm1_ffn[i] = 0.0

    for i in range(dim):
        var acc: Float32 = 0.0
        for j in range(dim):
            acc = acc + grad_h1[j] * bloco.w_ffn_1.dados[i * dim + j]
        grad_norm1_ffn[i] = acc

    for i in range(dim):
        for j in range(dim):
            var idx = i * dim + j
            bloco.w_ffn_1.dados[idx] = bloco.w_ffn_1.dados[idx] - taxa_aprendizado * norm1_vec[i] * grad_h1[j]
    for j in range(dim):
        bloco.b_ffn_1.dados[j] = bloco.b_ffn_1.dados[j] - taxa_aprendizado * grad_h1[j]

    let grad_norm1_total = workspace.tmp_norm1_total.dados
    for i in range(dim):
        grad_norm1_total[i] = grad_norm1_skip[i] + grad_norm1_ffn[i]

    var grad_resid1 = _layer_norm_backward_linha(resid1_vec, grad_norm1_total, bloco.layer_norm_epsilon)

    # Caminho de atencao (analitico no ultimo token):
    # resid_1 = emb + attn_out, attn_out = context * W_o.
    var grad_attn = grad_resid1.copy()
    var grad_emb_skip = grad_resid1.copy()

    var seq_len = workspace.emb_prefix.formato[0]
    if seq_len <= 0:
        return
    if bloco.num_heads <= 0 or dim % bloco.num_heads != 0:
        return

    var head_dim = dim // bloco.num_heads
    var escala = 1.0 / Float32(math.sqrt(Float64(head_dim)))
    var last = seq_len - 1

    var w_q_old = bloco.w_q.dados.copy()
    var w_k_old = bloco.w_k.dados.copy()
    var w_v_old = bloco.w_v.dados.copy()
    var w_o_old = bloco.w_o.dados.copy()

    # emb_rows_compartilhadas removido, não é mais necessário

    var tipo = cache.emb.tipo_computacao
    for p in range(seq_len):
        let base = p * dim
        for d in range(dim):
            workspace.emb_prefix.dados[base + d] = cache.emb.dados[base + d]

    for i in range(dim * dim):
        workspace.w_q_old_t.dados[i] = w_q_old[i]
        workspace.w_k_old_t.dados[i] = w_k_old[i]
        workspace.w_v_old_t.dados[i] = w_v_old[i]

    var q_all = dispatcher_tensor.multiplicar_matrizes(workspace.emb_prefix, workspace.w_q_old_t)
    var k_all = dispatcher_tensor.multiplicar_matrizes(workspace.emb_prefix, workspace.w_k_old_t)
    var v_all = dispatcher_tensor.multiplicar_matrizes(workspace.emb_prefix, workspace.w_v_old_t)
    _transpor_para(workspace.emb_prefix, workspace.emb_prefix_t)

    let base = last * dim
    for i in range(dim):
        workspace.emb_last_col.dados[i] = cache.emb.dados[base + i]

    for i in range(len(workspace.context_last_t.dados)):
        workspace.context_last_t.dados[i] = 0.0
    for i in range(dim):
        var acc: Float32 = 0.0
        for j in range(dim):
            acc = acc + grad_attn[j] * w_o_old[i * dim + j]
        workspace.grad_context_t.dados[i] = acc

    for i in range(len(workspace.grad_emb_atencao.dados)):
        workspace.grad_emb_atencao.dados[i] = 0.0

    for h in range(bloco.num_heads):
        var hs = h * head_dim

        for p in range(seq_len):
            for d in range(head_dim):
                var col = hs + d
                workspace.k_head.dados[p * head_dim + d] = k_all.dados[p * dim + col]
                workspace.v_head.dados[p * head_dim + d] = v_all.dados[p * dim + col]

        for d in range(head_dim):
            var col = hs + d
            var qd = q_all.dados[last * dim + col]
            workspace.q_head_t.dados[d] = qd

        _transpor_para(workspace.k_head, workspace.k_head_t)
        var scores_t = dispatcher_tensor.multiplicar_matrizes(workspace.q_head_t, workspace.k_head_t)

        var max_s = _escalar_max_1xn(scores_t, seq_len, escala)
        _mascarar_causal_1xn(scores_t, seq_len, seq_limite)
        if seq_limite > 0:
            max_s = scores_t.dados[0]
            for p in range(1, seq_limite):
                if scores_t.dados[p] > max_s:
                    max_s = scores_t.dados[p]
        _softmax_1xn_para_probs(scores_t, workspace.probs_t, seq_len, max_s)
        _transpor_para(workspace.probs_t, workspace.probs_col)

        var context_h_t = dispatcher_tensor.multiplicar_matrizes(workspace.probs_t, workspace.v_head)

        for d in range(head_dim):
            workspace.context_last_t.dados[hs + d] = context_h_t.dados[d]

        for d in range(head_dim):
            workspace.grad_ctx_h_t.dados[d] = workspace.grad_context_t.dados[hs + d]

        _transpor_para(workspace.grad_ctx_h_t, workspace.grad_ctx_h_col)
        var d_p_t = dispatcher_tensor.multiplicar_matrizes(workspace.v_head, workspace.grad_ctx_h_col)
        var d_v_t = dispatcher_tensor.multiplicar_matrizes(workspace.probs_col, workspace.grad_ctx_h_t)

        var soma_pdp: Float32 = 0.0
        for p in range(seq_len):
            soma_pdp = soma_pdp + workspace.probs_t.dados[p] * d_p_t.dados[p]

        for p in range(seq_len):
            workspace.d_s_t.dados[p] = workspace.probs_t.dados[p] * (d_p_t.dados[p] - soma_pdp)
        _transpor_para(workspace.d_s_t, workspace.d_s_col)

        var d_q_t = dispatcher_tensor.multiplicar_matrizes(workspace.d_s_t, workspace.k_head)
        for d in range(head_dim):
            workspace.d_q_scaled_t.dados[d] = d_q_t.dados[d] * escala

        var d_k_t = dispatcher_tensor.multiplicar_matrizes(workspace.d_s_col, workspace.q_head_t)
        for i in range(len(d_k_t.dados)):
            d_k_t.dados[i] = d_k_t.dados[i] * escala

        var grad_w_v_t = dispatcher_tensor.multiplicar_matrizes(workspace.emb_prefix_t, d_v_t)
        var grad_w_k_t = dispatcher_tensor.multiplicar_matrizes(workspace.emb_prefix_t, d_k_t)
        var grad_w_q_t = dispatcher_tensor.multiplicar_matrizes(workspace.emb_last_col, workspace.d_q_scaled_t)

        for d in range(head_dim):
            var col = hs + d
            for i in range(dim):
                workspace.w_v_head_old_ht.dados[d * dim + i] = w_v_old[i * dim + col]
                workspace.w_k_head_old_ht.dados[d * dim + i] = w_k_old[i * dim + col]
                workspace.w_q_head_old_ht.dados[d * dim + i] = w_q_old[i * dim + col]

        var grad_emb_v_t = dispatcher_tensor.multiplicar_matrizes(d_v_t, workspace.w_v_head_old_ht)
        var grad_emb_k_t = dispatcher_tensor.multiplicar_matrizes(d_k_t, workspace.w_k_head_old_ht)
        var grad_emb_q_last_t = dispatcher_tensor.multiplicar_matrizes(workspace.d_q_scaled_t, workspace.w_q_head_old_ht)

        for i in range(dim):
            for d in range(head_dim):
                var col = hs + d
                bloco.w_v.dados[i * dim + col] = bloco.w_v.dados[i * dim + col] - taxa_aprendizado * grad_w_v_t.dados[i * head_dim + d]
                bloco.w_k.dados[i * dim + col] = bloco.w_k.dados[i * dim + col] - taxa_aprendizado * grad_w_k_t.dados[i * head_dim + d]
                bloco.w_q.dados[i * dim + col] = bloco.w_q.dados[i * dim + col] - taxa_aprendizado * grad_w_q_t.dados[i * head_dim + d]

        for p in range(seq_len):
            for i in range(dim):
                var g = grad_emb_v_t.dados[p * dim + i] + grad_emb_k_t.dados[p * dim + i]
                if p == last:
                    g = g + grad_emb_q_last_t.dados[i]
                workspace.grad_emb_atencao.dados[p * dim + i] = workspace.grad_emb_atencao.dados[p * dim + i] + g

    for i in range(dim):
        for j in range(dim):
            bloco.w_o.dados[i * dim + j] = bloco.w_o.dados[i * dim + j] - taxa_aprendizado * workspace.context_last_t.dados[i] * grad_attn[j]

    var max_pos = len(token_ids)
    if max_pos > seq_limite:
        max_pos = seq_limite
    for p in range(max_pos):
        var tok = token_ids[p]
        if tok < 0:
            tok = 0
        if tok >= bloco.embedding.vocab_size:
            tok = bloco.embedding.vocab_size - 1
        var base = tok * dim
        for d in range(dim):
            var g_total = workspace.grad_emb_atencao.dados[p * dim + d]
            if p == last:
                g_total = g_total + grad_emb_skip[d]
            bloco.embedding.tabela.dados[base + d] = bloco.embedding.tabela.dados[base + d] - taxa_aprendizado * g_total


fn _aplicar_gradiente_saida_transformer_analitico_parcial_limite(
    mut bloco: BlocoTransformerBase,
    token_ids: List[Int],
    cache: ForwardTransformerCache,
    grad_saida_ultimo_token: List[Float32],
    var seq_limite: Int,
    var taxa_aprendizado: Float32 = 0.001,
):
    if seq_limite <= 0:
        return
    if seq_limite > cache.out.formato[0]:
        seq_limite = cache.out.formato[0]

    # emb_rows_compartilhadas removido, acesso direto ao tensor

    var dim = bloco.embedding.dimensao_modelo
    var head_dim = dim
    if bloco.num_heads > 0:
        head_dim = dim // bloco.num_heads
    var workspace = _criar_workspace_backward_transformer(seq_limite, dim, head_dim, cache.emb.tipo_computacao)

    _aplicar_gradiente_saida_transformer_analitico_parcial_limite_com_emb_rows(
        bloco,
        token_ids,
        cache,
        grad_saida_ultimo_token,
        seq_limite,
        workspace,
        taxa_aprendizado,
    )


fn _layer_norm_backward_linha(x: Float32[], grad_y: Float32[], var epsilon: Float32 = 1e-5) -> List[Float32]:
    var n = len(x)
    var out = List[Float32]()
    if n <= 0:
        return out^

    var media: Float32 = 0.0
    for i in range(n):
        media = media + x[i]
    media = media / Float32(n)

    var var_acc: Float32 = 0.0
    for i in range(n):
        var d = x[i] - media
        var_acc = var_acc + d * d
    var_acc = var_acc / Float32(n)

    var inv_std = 1.0 / Float32(math.sqrt(Float64(var_acc + epsilon)))
    var inv_var = inv_std * inv_std

    var soma_dy: Float32 = 0.0
    var soma_dy_xmu: Float32 = 0.0
    for i in range(n):
        var dy = grad_y[i]
        var xmu = x[i] - media
        soma_dy = soma_dy + dy
        soma_dy_xmu = soma_dy_xmu + dy * xmu

    for i in range(n):
        var dy = grad_y[i]
        var xmu = x[i] - media
        var dx = inv_std * (dy - soma_dy / Float32(n) - xmu * inv_var * (soma_dy_xmu / Float32(n)))
        out.append(dx)

    return out^


fn forward_transformer_base_com_cache(
    bloco: BlocoTransformerBase,
    token_ids: List[Int],
    mut kvcache: kvcache_sessao.KVCacheProvider,
    adaptador_atencao: adaptadores_atencao.AdaptadorAtencao = adaptadores_atencao.criar_adaptador_atencao_nenhum(),
    mut captura_adaptador: captura_camadas.CapturaCamadasAdaptador = captura_camadas.criar_captura_camadas_desativado(),
    var camada_idx: Int = 0,
    var posicao_base: Int = 0,
) -> ForwardTransformerCache:
    var emb = embedding_forward(bloco.embedding, token_ids)
    var attn = atencao_causal_multihead_com_kvcache(
        emb,
        bloco.w_q,
        bloco.w_k,
        bloco.w_v,
        bloco.w_o,
        bloco.num_heads,
        kvcache,
        adaptador_atencao,
        camada_idx,
        posicao_base,
    )

    var resid_1 = dispatcher_tensor.somar_elemento_a_elemento(emb, attn)
    var norm_1 = _layer_norm_linhas(resid_1, bloco.layer_norm_epsilon)

    var ffn_h1 = dispatcher_tensor.multiplicar_matrizes(norm_1, bloco.w_ffn_1)
    ffn_h1 = dispatcher_tensor.adicionar_bias_coluna(ffn_h1, bloco.b_ffn_1)
    var ffn_h1_relu = _relu_tensor(ffn_h1)
    var ffn_out = dispatcher_tensor.multiplicar_matrizes(ffn_h1_relu, bloco.w_ffn_2)
    ffn_out = dispatcher_tensor.adicionar_bias_coluna(ffn_out, bloco.b_ffn_2)

    var resid_2 = dispatcher_tensor.somar_elemento_a_elemento(norm_1, ffn_out)
    var out = _layer_norm_linhas(resid_2, bloco.layer_norm_epsilon)

    _ = captura_camadas.capturar_io_camada(captura_adaptador, "infer", "transformer/bloco_" + String(camada_idx), emb, out)
    return ForwardTransformerCache(emb, attn, resid_1, norm_1, ffn_h1, ffn_h1_relu, ffn_out, resid_2, out)


fn forward_transformer_base(
    bloco: BlocoTransformerBase,
    token_ids: List[Int],
    mut kvcache: kvcache_sessao.KVCacheProvider,
    adaptador_atencao: adaptadores_atencao.AdaptadorAtencao = adaptadores_atencao.criar_adaptador_atencao_nenhum(),
    mut captura_adaptador: captura_camadas.CapturaCamadasAdaptador = captura_camadas.criar_captura_camadas_desativado(),
    var camada_idx: Int = 0,
    var posicao_base: Int = 0,
) -> tensor_defs.Tensor:
    var cache = forward_transformer_base_com_cache(
        bloco,
        token_ids,
        kvcache,
        adaptador_atencao,
        captura_adaptador,
        camada_idx,
        posicao_base,
    )
    return cache.out.copy()


fn aplicar_gradiente_saida_transformer_analitico_parcial(
    mut bloco: BlocoTransformerBase,
    token_ids: List[Int],
    cache: ForwardTransformerCache,
    grad_saida_ultimo_token: List[Float32],
    var taxa_aprendizado: Float32 = 0.001,
):
    _aplicar_gradiente_saida_transformer_analitico_parcial_limite(
        bloco,
        token_ids,
        cache,
        grad_saida_ultimo_token,
        cache.out.formato[0],
        taxa_aprendizado,
    )


fn aplicar_gradiente_saida_transformer_analitico_todos_tokens(
    mut bloco: BlocoTransformerBase,
    token_ids: List[Int],
    cache: ForwardTransformerCache,
    grads_saida_por_token: List[List[Float32]],
    var taxa_aprendizado: Float32 = 0.001,
):
    var seq_len = cache.out.formato[0]
    if seq_len <= 0:
        return

    var n = len(grads_saida_por_token)
    if n > seq_len:
        n = seq_len
    if n > len(token_ids):
        n = len(token_ids)

    # Reusa rows de embedding para todas as posicoes, reduzindo custo host-side.
    # emb_rows_compartilhadas removido, acesso direto ao tensor

    var dim = bloco.embedding.dimensao_modelo
    var head_dim = dim
    if bloco.num_heads > 0:
        head_dim = dim // bloco.num_heads
    var workspace = _criar_workspace_backward_transformer(seq_len, dim, head_dim, cache.emb.tipo_computacao)

    # Atualizacao causal por prefixo, evitando copia de tensores por token.
    for pos in range(n):
        var grad_pos = grads_saida_por_token[pos]
        if len(grad_pos) == 0:
            continue

        _aplicar_gradiente_saida_transformer_analitico_parcial_limite_com_emb_rows(
            bloco,
            token_ids,
            cache,
            grad_pos,
            pos + 1,
            workspace,
            taxa_aprendizado,
        )


fn aplicar_gradiente_saida_transformer(
    mut bloco: BlocoTransformerBase,
    token_ids: List[Int],
    saida_transformer: tensor_defs.Tensor,
    grad_saida_ultimo_token: List[Float32],
    var taxa_aprendizado: Float32 = 0.001,
):
    if len(saida_transformer.formato) != 2:
        return
    if len(saida_transformer.formato) < 2 or saida_transformer.formato[0] <= 0:
        return

    var dim = bloco.embedding.dimensao_modelo
    if dim <= 0:
        return
    if len(grad_saida_ultimo_token) < dim:
        return

    var ultima_linha = saida_transformer.formato[0] - 1
    var base = ultima_linha * dim
    var feat = List[Float32]()
    for d in range(dim):
        feat.append(saida_transformer.dados[base + d])

    # Atualizacao surrogate: injeta gradiente do ultimo token nas projecoes de atencao e FFN.
    var escala_attn: Float32 = 0.25
    for i in range(dim):
        for j in range(dim):
            var g = feat[i] * grad_saida_ultimo_token[j]
            var idx = i * dim + j

            bloco.w_o.dados[idx] = bloco.w_o.dados[idx] - taxa_aprendizado * g
            bloco.w_ffn_2.dados[idx] = bloco.w_ffn_2.dados[idx] - taxa_aprendizado * g
            bloco.w_q.dados[idx] = bloco.w_q.dados[idx] - taxa_aprendizado * escala_attn * g
            bloco.w_k.dados[idx] = bloco.w_k.dados[idx] - taxa_aprendizado * escala_attn * g
            bloco.w_v.dados[idx] = bloco.w_v.dados[idx] - taxa_aprendizado * escala_attn * g

    for j in range(dim):
        bloco.b_ffn_2.dados[j] = bloco.b_ffn_2.dados[j] - taxa_aprendizado * grad_saida_ultimo_token[j]

    # Pequena realimentacao no embedding dos tokens de contexto usados na janela.
    if len(token_ids) > 0:
        var escala_emb = taxa_aprendizado / Float32(len(token_ids))
        for t in token_ids:
            var tok = t
            if tok < 0:
                tok = 0
            if tok >= bloco.embedding.vocab_size:
                tok = bloco.embedding.vocab_size - 1
            var emb_base = tok * dim
            for d in range(dim):
                bloco.embedding.tabela.dados[emb_base + d] = bloco.embedding.tabela.dados[emb_base + d] - escala_emb * grad_saida_ultimo_token[d]
