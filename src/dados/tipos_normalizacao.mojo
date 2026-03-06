# Tipos de normalização (enum-like) e utilitários de conversão/validação

fn normalizacao_nenhuma_id() -> Int:
    return 0


fn normalizacao_minmax_id() -> Int:
    return 1


fn normalizacao_zscore_id() -> Int:
    return 2


fn _normalizar_nome_interno(var nome: String) -> String:
    var n = nome.strip()
    if n == "nenhuma" or n == "NENHUMA" or n == "none" or n == "NONE":
        return "nenhuma"
    if n == "minmax" or n == "MINMAX" or n == "min_max":
        return "minmax"
    if n == "zscore" or n == "ZSCORE" or n == "z_score":
        return "zscore"
    var out = ""
    for i in range(len(n)):
        out = out + n[i:i+1]
    return out


fn normalizacao_id_de_nome(var nome: String) -> Int:
    var n = _normalizar_nome_interno(nome)
    if n == "nenhuma":
        return normalizacao_nenhuma_id()
    if n == "minmax":
        return normalizacao_minmax_id()
    if n == "zscore":
        return normalizacao_zscore_id()
    return -1


fn normalizacao_nome_de_id(var normalizacao_id: Int) -> String:
    if normalizacao_id == normalizacao_nenhuma_id():
        return "nenhuma"
    if normalizacao_id == normalizacao_minmax_id():
        return "minmax"
    if normalizacao_id == normalizacao_zscore_id():
        return "zscore"
    return "desconhecida"


fn normalizacao_id_valido(var normalizacao_id: Int) -> Bool:
    return normalizacao_id == normalizacao_nenhuma_id() or normalizacao_id == normalizacao_minmax_id() or normalizacao_id == normalizacao_zscore_id()


fn normalizacao_nome_valido(var nome: String) -> Bool:
    return normalizacao_id_de_nome(nome) >= 0