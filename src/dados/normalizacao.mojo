# Normalizações: Min-Max e Z-Score (padronização)

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

import math

fn min_max_normalize(var dados: List[List[Float32]]) -> MinMaxResult:
    if len(dados) == 0:
        return MinMaxResult(List[List[Float32]](), List[Float32](), List[Float32]())
    var n_linhas = len(dados)
    var n_cols = len(dados[0])
    var min_col = List[Float32](n_cols)
    var max_col = List[Float32](n_cols)
    for j in range(n_cols):
        min_col[j] = dados[0][j]
        max_col[j] = dados[0][j]
    for i in range(n_linhas):
        for j in range(n_cols):
            if dados[i][j] < min_col[j]:
                min_col[j] = dados[i][j]
            if dados[i][j] > max_col[j]:
                max_col[j] = dados[i][j]

    var resultado = List[List[Float32]]()
    for i in range(n_linhas):
        var linha_norm = List[Float32](n_cols)
        for j in range(n_cols):
            var denom: Float32 = max_col[j] - min_col[j]
            if denom == 0.0:
                linha_norm[j] = 0.0
            else:
                linha_norm[j] = (dados[i][j] - min_col[j]) / denom
        resultado.append(linha_norm^)
    return MinMaxResult(resultado^, min_col^, max_col^)

fn z_score_normalize(var dados: List[List[Float32]]) -> ZScoreResult:
    if len(dados) == 0:
        return ZScoreResult(List[List[Float32]](), List[Float32](), List[Float32]())
    var n_linhas = len(dados)
    var n_cols = len(dados[0])
    var medias = List[Float32](n_cols)
    var variancias = List[Float32](n_cols)
    for j in range(n_cols):
        medias[j] = 0.0
        variancias[j] = 0.0

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

    var desvios = List[Float32](n_cols)
    for j in range(n_cols):
        desvios[j] = Float32(math.sqrt(variancias[j]))

    var resultado = List[List[Float32]]()
    for i in range(n_linhas):
        var linha_norm = List[Float32](n_cols)
        for j in range(n_cols):
            if desvios[j] == 0.0:
                linha_norm[j] = 0.0
            else:
                linha_norm[j] = (dados[i][j] - medias[j]) / desvios[j]
        resultado.append(linha_norm^)

    return ZScoreResult(resultado^, medias^, desvios^)
