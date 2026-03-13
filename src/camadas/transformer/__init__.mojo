import src.camadas.transformer.bloco_transformer as bloco
import src.computacao.kvcache_sessao as kvcache
import src.computacao.captura_camadas as captura_camadas
import src.camadas.transformer.adaptadores_atencao as adaptadores

alias BlocoEmbedding = bloco.BlocoEmbedding
alias BlocoTransformerBase = bloco.BlocoTransformerBase
alias ForwardTransformerCache = bloco.ForwardTransformerCache
alias AdaptadorAtencao = adaptadores.AdaptadorAtencao


def criar_bloco_transformer_base(
    var vocab_size: Int,
    var dimensao_modelo: Int,
    var num_heads: Int = 4,
    var tipo_computacao: String = "cpu",
) -> bloco.BlocoTransformerBase:
    return bloco.BlocoTransformerBase(vocab_size, dimensao_modelo, num_heads, tipo_computacao)


def embedding_forward(bloco_embedding: bloco.BlocoEmbedding, token_ids: List[Int]):
    return bloco.embedding_forward(bloco_embedding, token_ids)


def forward_transformer_base(
    bloco_transformer: bloco.BlocoTransformerBase,
    token_ids: List[Int],
    mut kvcache_provider: kvcache.KVCacheProvider,
    adaptador_atencao: adaptadores.AdaptadorAtencao = adaptadores.criar_adaptador_atencao_nenhum(),
    mut captura_adaptador: captura_camadas.CapturaCamadasAdaptador = captura_camadas.criar_captura_camadas_desativado(),
    var camada_idx: Int = 0,
    var posicao_base: Int = 0,
):
    return bloco.forward_transformer_base(
        bloco_transformer,
        token_ids,
        kvcache_provider,
        adaptador_atencao,
        captura_adaptador,
        camada_idx,
        posicao_base,
    )


def forward_transformer_base_com_cache(
    bloco_transformer: bloco.BlocoTransformerBase,
    token_ids: List[Int],
    mut kvcache_provider: kvcache.KVCacheProvider,
    adaptador_atencao: adaptadores.AdaptadorAtencao = adaptadores.criar_adaptador_atencao_nenhum(),
    mut captura_adaptador: captura_camadas.CapturaCamadasAdaptador = captura_camadas.criar_captura_camadas_desativado(),
    var camada_idx: Int = 0,
    var posicao_base: Int = 0,
) -> bloco.ForwardTransformerCache:
    return bloco.forward_transformer_base_com_cache(
        bloco_transformer,
        token_ids,
        kvcache_provider,
        adaptador_atencao,
        captura_adaptador,
        camada_idx,
        posicao_base,
    )


def aplicar_gradiente_saida_transformer(
    mut bloco_transformer: bloco.BlocoTransformerBase,
    token_ids: List[Int],
    saida_transformer,
    grad_saida_ultimo_token: List[Float32],
    var taxa_aprendizado: Float32 = 0.001,
):
    bloco.aplicar_gradiente_saida_transformer(
        bloco_transformer,
        token_ids,
        saida_transformer,
        grad_saida_ultimo_token,
        taxa_aprendizado,
    )


def aplicar_gradiente_saida_transformer_analitico_parcial(
    mut bloco_transformer: bloco.BlocoTransformerBase,
    token_ids: List[Int],
    cache_forward: bloco.ForwardTransformerCache,
    grad_saida_ultimo_token: List[Float32],
    var taxa_aprendizado: Float32 = 0.001,
):
    bloco.aplicar_gradiente_saida_transformer_analitico_parcial(
        bloco_transformer,
        token_ids,
        cache_forward,
        grad_saida_ultimo_token,
        taxa_aprendizado,
    )


def aplicar_gradiente_saida_transformer_analitico_todos_tokens(
    mut bloco_transformer: bloco.BlocoTransformerBase,
    token_ids: List[Int],
    cache_forward: bloco.ForwardTransformerCache,
    grads_saida_por_token: List[List[Float32]],
    var taxa_aprendizado: Float32 = 0.001,
):
    bloco.aplicar_gradiente_saida_transformer_analitico_todos_tokens(
        bloco_transformer,
        token_ids,
        cache_forward,
        grads_saida_por_token,
        taxa_aprendizado,
    )


def criar_adaptador_atencao_nenhum() -> adaptadores.AdaptadorAtencao:
    return adaptadores.criar_adaptador_atencao_nenhum()


def criar_adaptador_atencao_svd(var rank: Int = 8) -> adaptadores.AdaptadorAtencao:
    return adaptadores.criar_adaptador_atencao_svd(rank)


def criar_adaptador_atencao_lora(var rank: Int = 8, var alpha: Float32 = 16.0) -> adaptadores.AdaptadorAtencao:
    return adaptadores.criar_adaptador_atencao_lora(rank, alpha)


def criar_adaptador_atencao_flash(var bloco_flash: Int = 128) -> adaptadores.AdaptadorAtencao:
    return adaptadores.criar_adaptador_atencao_flash(bloco_flash)


def criar_adaptador_atencao_combinado(var pilha_modos: List[String]) -> adaptadores.AdaptadorAtencao:
    return adaptadores.criar_adaptador_atencao_combinado(pilha_modos)
