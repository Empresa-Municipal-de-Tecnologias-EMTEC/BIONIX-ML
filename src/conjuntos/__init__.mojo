import src.conjuntos.csv_supervisionado as csv_supervisionado
import src.conjuntos.bitmap_supervisionado as bmp_supervisionado
import src.conjuntos.wav_supervisionado as wav_supervisionado
import src.conjuntos.lotes_supervisionados as lotes_supervisionados
import src.dados.tipos_normalizacao as norm_tipos

alias LoteSupervisionado = lotes_supervisionados.LoteSupervisionado
alias LoteEpocaSupervisionado = lotes_supervisionados.LoteEpocaSupervisionado
alias PreparacaoTreinoValidacaoLotes = lotes_supervisionados.PreparacaoTreinoValidacaoLotes

def carregar_csv_supervisionado(
    var caminho: String,
    var indice_coluna_alvo: Int = -1,
    var delimitador: String = ",",
    var detectar_cabecalho: Bool = True,
    var tipo_computacao: String = "cpu",
    var normalizacao_entradas_id: Int = norm_tipos.normalizacao_zscore_id(),
    var normalizacao_alvo_id: Int = norm_tipos.normalizacao_zscore_id(),
) -> csv_supervisionado.ConjuntoSupervisionado:
    return csv_supervisionado.carregar_csv_supervisionado(
        caminho,
        indice_coluna_alvo,
        delimitador,
        detectar_cabecalho,
        tipo_computacao,
        normalizacao_entradas_id,
        normalizacao_alvo_id,
    )


def normalizar_amostra_entradas(conjunto: csv_supervisionado.ConjuntoSupervisionado, amostra: List[Float32]) -> List[Float32]:
    return csv_supervisionado.normalizar_amostra_entradas(conjunto, amostra)


def desnormalizar_valor_alvo(conjunto: csv_supervisionado.ConjuntoSupervisionado, valor_normalizado: Float32) -> Float32:
    return csv_supervisionado.desnormalizar_valor_alvo(conjunto, valor_normalizado)


def carregar_bitmap_supervisionado(
    var caminho_ou_diretorio: String,
    var tipo_computacao: String = "cpu",
    var stride: Int = 2,
    var limiar_amostra: Float32 = 0.05,
    var limiar_classe: Float32 = 0.6,
) -> csv_supervisionado.ConjuntoSupervisionado:
    return bmp_supervisionado.carregar_bitmap_supervisionado(
        caminho_ou_diretorio,
        tipo_computacao,
        stride,
        limiar_amostra,
        limiar_classe,
    )


def carregar_wav_supervisionado(
    var caminho_ou_diretorio: String,
    var tipo_computacao: String = "cpu",
    var stride: Int = 4,
    var limiar_amostra: Float32 = 0.05,
    var limiar_classe: Float32 = 0.35,
) -> csv_supervisionado.ConjuntoSupervisionado:
    return wav_supervisionado.carregar_wav_supervisionado(
        caminho_ou_diretorio,
        tipo_computacao,
        stride,
        limiar_amostra,
        limiar_classe,
    )


def quebrar_em_lotes(conjunto: csv_supervisionado.ConjuntoSupervisionado, var tamanho_lote: Int) -> List[LoteSupervisionado]:
    return lotes_supervisionados.quebrar_em_lotes(conjunto, tamanho_lote)


def dividir_treino_validacao(
    conjunto: csv_supervisionado.ConjuntoSupervisionado,
    var proporcao_validacao: Float32 = 0.2,
) -> List[csv_supervisionado.ConjuntoSupervisionado]:
    return lotes_supervisionados.dividir_treino_validacao(conjunto, proporcao_validacao)


def preparar_treino_validacao_em_lotes(
    conjunto: csv_supervisionado.ConjuntoSupervisionado,
    var tamanho_lote: Int,
    var epocas_treino: Int = 1,
    var proporcao_validacao: Float32 = 0.2,
) -> PreparacaoTreinoValidacaoLotes:
    return lotes_supervisionados.preparar_treino_validacao_em_lotes(
        conjunto,
        tamanho_lote,
        epocas_treino,
        proporcao_validacao,
    )