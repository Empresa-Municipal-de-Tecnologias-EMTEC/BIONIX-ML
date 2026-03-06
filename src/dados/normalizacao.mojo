# Normalizações: Min-Max e Z-Score (padronização)
import src.uteis as uteis

struct MinMaxResult(Movable, Copyable):
    var dados_normalizados: List[List[Float32]]
    var minimo_por_coluna: List[Float32]
    var maximo_por_coluna: List[Float32]

    fn __init__(out self, var dados_norm: List[List[Float32]], var min_col: List[Float32], var max_col: List[Float32]):
        self.dados_normalizados = dados_norm^
        self.minimo_por_coluna = min_col^
        self.maximo_por_coluna = max_col^

struct ZScoreResult(Movable, Copyable):
    var dados_normalizados: List[List[Float32]]
    var media_por_coluna: List[Float32]
    var desvio_por_coluna: List[Float32]

    fn __init__(out self, var dados_norm: List[List[Float32]], var medias: List[Float32], var desvios: List[Float32]):
        self.dados_normalizados = dados_norm^
        self.media_por_coluna = medias^
        self.desvio_por_coluna = desvios^


struct NormalizacaoPersistida(Movable, Copyable):
    var tipo_entradas: String
    var media_entradas: List[Float32]
    var desvio_entradas: List[Float32]
    var tipo_alvo: String
    var media_alvo: Float32
    var desvio_alvo: Float32

    fn __init__(
        out self,
        var tipo_entradas_in: String,
        var media_entradas_in: List[Float32],
        var desvio_entradas_in: List[Float32],
        var tipo_alvo_in: String,
        var media_alvo_in: Float32,
        var desvio_alvo_in: Float32,
    ):
        self.tipo_entradas = tipo_entradas_in^
        self.media_entradas = media_entradas_in^
        self.desvio_entradas = desvio_entradas_in^
        self.tipo_alvo = tipo_alvo_in^
        self.media_alvo = media_alvo_in
        self.desvio_alvo = desvio_alvo_in

    fn copy(self) -> NormalizacaoPersistida:
        return NormalizacaoPersistida(
            self.tipo_entradas,
            self.media_entradas.copy(),
            self.desvio_entradas.copy(),
            self.tipo_alvo,
            self.media_alvo,
            self.desvio_alvo,
        )

import math

fn criar_normalizacao_persistida(
    var tipo_entradas: String,
    var media_entradas: List[Float32],
    var desvio_entradas: List[Float32],
    var tipo_alvo: String,
    var media_alvo: Float32,
    var desvio_alvo: Float32,
) -> NormalizacaoPersistida:
    return NormalizacaoPersistida(
        tipo_entradas,
        media_entradas^,
        desvio_entradas^,
        tipo_alvo,
        media_alvo,
        desvio_alvo,
    )


fn salvar_normalizacao_persistida(norm: NormalizacaoPersistida, var caminho: String):
    var chaves = List[String]()
    var valores = List[String]()
    chaves.append("tipo_entradas")
    valores.append(norm.tipo_entradas)
    chaves.append("media_entradas")
    valores.append(uteis.float_list_para_csv(norm.media_entradas.copy()))
    chaves.append("desvio_entradas")
    valores.append(uteis.float_list_para_csv(norm.desvio_entradas.copy()))
    chaves.append("tipo_alvo")
    valores.append(norm.tipo_alvo)
    chaves.append("media_alvo")
    valores.append(String(norm.media_alvo))
    chaves.append("desvio_alvo")
    valores.append(String(norm.desvio_alvo))
    _ = uteis.salvar_kv_arquivo_seguro(caminho, chaves, valores)


fn carregar_normalizacao_persistida(var caminho: String) -> NormalizacaoPersistida:
    var kv = uteis.carregar_kv_arquivo_seguro(caminho)
    if len(kv.chaves) == 0:
        return NormalizacaoPersistida("nenhuma", List[Float32](), List[Float32](), "nenhuma", 0.0, 1.0)

    var tipo_entradas = uteis.obter_valor_ou_padrao(kv, "tipo_entradas", "nenhuma")
    var medias = List[Float32]()
    var desvios = List[Float32]()
    var tipo_alvo = uteis.obter_valor_ou_padrao(kv, "tipo_alvo", "nenhuma")
    var media_alvo = uteis.parse_float_ascii(uteis.obter_valor_ou_padrao(kv, "media_alvo", "0"))
    var desvio_alvo = uteis.parse_float_ascii(uteis.obter_valor_ou_padrao(kv, "desvio_alvo", "1"))

    var itens_m = uteis.split_csv_simples(uteis.obter_valor_ou_padrao(kv, "media_entradas", ""))
    for it in itens_m:
        if len(it.strip()) > 0:
            medias.append(uteis.parse_float_ascii(it))

    var itens_d = uteis.split_csv_simples(uteis.obter_valor_ou_padrao(kv, "desvio_entradas", ""))
    for it in itens_d:
        if len(it.strip()) > 0:
            desvios.append(uteis.parse_float_ascii(it))

    return NormalizacaoPersistida(tipo_entradas, medias^, desvios^, tipo_alvo, media_alvo, desvio_alvo)


fn normalizar_amostra_entradas(norm: NormalizacaoPersistida, amostra: List[Float32]) -> List[Float32]:
    if norm.tipo_entradas != "zscore":
        return amostra.copy()
    var out = List[Float32]()
    for i in range(len(amostra)):
        if i >= len(norm.media_entradas) or i >= len(norm.desvio_entradas):
            out.append(amostra[i])
            continue
        var d = norm.desvio_entradas[i]
        if d == 0.0:
            out.append(0.0)
        else:
            out.append((amostra[i] - norm.media_entradas[i]) / d)
    return out^


fn desnormalizar_valor_alvo(norm: NormalizacaoPersistida, valor_normalizado: Float32) -> Float32:
    if norm.tipo_alvo != "zscore":
        return valor_normalizado
    return valor_normalizado * norm.desvio_alvo + norm.media_alvo

fn min_max_normalize(var dados: List[List[Float32]]) -> MinMaxResult:
    if len(dados) == 0:
        return MinMaxResult(List[List[Float32]](), List[Float32](), List[Float32]())
    var n_linhas = len(dados)
    var n_cols = len(dados[0])
    var min_col = List[Float32]()
    var max_col = List[Float32]()
    for j in range(n_cols):
        min_col.append(dados[0][j])
        max_col.append(dados[0][j])
    for i in range(n_linhas):
        for j in range(n_cols):
            if dados[i][j] < min_col[j]:
                min_col[j] = dados[i][j]
            if dados[i][j] > max_col[j]:
                max_col[j] = dados[i][j]

    var resultado = List[List[Float32]]()
    for i in range(n_linhas):
        var linha_norm = List[Float32]()
        for j in range(n_cols):
            var denom: Float32 = max_col[j] - min_col[j]
            if denom == 0.0:
                linha_norm.append(0.0)
            else:
                linha_norm.append((dados[i][j] - min_col[j]) / denom)
        resultado.append(linha_norm^)
    return MinMaxResult(resultado^, min_col^, max_col^)

fn z_score_normalize(var dados: List[List[Float32]]) -> ZScoreResult:
    if len(dados) == 0:
        return ZScoreResult(List[List[Float32]](), List[Float32](), List[Float32]())
    var n_linhas = len(dados)
    var n_cols = len(dados[0])
    var medias = List[Float32]()
    var variancias = List[Float32]()
    for j in range(n_cols):
        medias.append(0.0)
        variancias.append(0.0)

    for i in range(n_linhas):
        for j in range(n_cols):
            medias[j] = medias[j] + dados[i][j]
    for j in range(n_cols):
        medias[j] = medias[j] / Float32(n_linhas)

    for i in range(n_linhas):
        for j in range(n_cols):
            var diff = dados[i][j] - medias[j]
            variancias[j] = variancias[j] + diff * diff
    for j in range(n_cols):
        variancias[j] = variancias[j] / Float32(n_linhas)

    var desvios = List[Float32]()
    for j in range(n_cols):
        desvios.append(Float32(math.sqrt(variancias[j])))

    var resultado = List[List[Float32]]()
    for i in range(n_linhas):
        var linha_norm = List[Float32]()
        for j in range(n_cols):
            if desvios[j] == 0.0:
                linha_norm.append(0.0)
            else:
                linha_norm.append((dados[i][j] - medias[j]) / desvios[j])
        resultado.append(linha_norm^)

    return ZScoreResult(resultado^, medias^, desvios^)
