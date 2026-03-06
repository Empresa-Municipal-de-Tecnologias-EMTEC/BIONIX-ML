import src.conjuntos.csv_supervisionado as csv_supervisionado

def carregar_csv_supervisionado(
    var caminho: String,
    var indice_coluna_alvo: Int = -1,
    var delimitador: String = ",",
    var detectar_cabecalho: Bool = True,
    var tipo_computacao: String = "cpu",
    var normalizacao_entradas: String = "zscore",
    var normalizacao_alvo: String = "zscore",
) -> csv_supervisionado.ConjuntoSupervisionado:
    return csv_supervisionado.carregar_csv_supervisionado(
        caminho,
        indice_coluna_alvo,
        delimitador,
        detectar_cabecalho,
        tipo_computacao,
        normalizacao_entradas,
        normalizacao_alvo,
    )


def normalizar_amostra_entradas(conjunto: csv_supervisionado.ConjuntoSupervisionado, amostra: List[Float32]) -> List[Float32]:
    return csv_supervisionado.normalizar_amostra_entradas(conjunto, amostra)


def desnormalizar_valor_alvo(conjunto: csv_supervisionado.ConjuntoSupervisionado, valor_normalizado: Float32) -> Float32:
    return csv_supervisionado.desnormalizar_valor_alvo(conjunto, valor_normalizado)