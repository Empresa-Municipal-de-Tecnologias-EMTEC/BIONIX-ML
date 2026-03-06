import src.conjuntos.csv_supervisionado as csv_supervisionado

def carregar_csv_supervisionado(
    var caminho: String,
    var indice_coluna_alvo: Int = -1,
    var delimitador: String = ",",
    var detectar_cabecalho: Bool = True,
    var tipo_computacao: String = "cpu",
) -> csv_supervisionado.ConjuntoSupervisionado:
    return csv_supervisionado.carregar_csv_supervisionado(
        caminho,
        indice_coluna_alvo,
        delimitador,
        detectar_cabecalho,
        tipo_computacao,
    )