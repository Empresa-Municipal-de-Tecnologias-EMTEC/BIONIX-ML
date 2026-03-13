import src.conjuntos.csv_supervisionado as csv_supervisionado
import src.conjuntos.txt_supervisionado as txt_supervisionado
import src.conjuntos.gpt_txt_supervisionado as gpt_txt_supervisionado
import src.conjuntos.bitmap_supervisionado as bmp_supervisionado
import src.conjuntos.wav_supervisionado as wav_supervisionado
import src.conjuntos.lotes_supervisionados as lotes_supervisionados
import src.dados.tipos_normalizacao as norm_tipos

alias LoteSupervisionado = lotes_supervisionados.LoteSupervisionado
alias LoteEpocaSupervisionado = lotes_supervisionados.LoteEpocaSupervisionado
alias PreparacaoTreinoValidacaoLotes = lotes_supervisionados.PreparacaoTreinoValidacaoLotes
alias ConjuntoTXTSupervisionado = txt_supervisionado.ConjuntoTXTSupervisionado
alias AmostraGPTSupervisionada = gpt_txt_supervisionado.AmostraGPTSupervisionada
alias LoteGPTSupervisionado = gpt_txt_supervisionado.LoteGPTSupervisionado
alias AmostraGPTJanelaToken = gpt_txt_supervisionado.AmostraGPTJanelaToken
alias LoteGPTJanelaToken = gpt_txt_supervisionado.LoteGPTJanelaToken

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


def carregar_txt_supervisionado(
    var caminho: String,
    var largura: Int,
    var altura: Int,
    var tipo_computacao: String = "cpu",
) -> txt_supervisionado.ConjuntoTXTSupervisionado:
    return txt_supervisionado.carregar_txt_supervisionado(caminho, largura, altura, tipo_computacao)


def carregar_amostras_gpt_txt(
    var caminho_arquivo: String,
    var max_tokens_inicio: Int = 128,
    var max_tokens_completar: Int = 128,
) -> List[gpt_txt_supervisionado.AmostraGPTSupervisionada]:
    return gpt_txt_supervisionado.carregar_amostras_gpt_txt(caminho_arquivo, max_tokens_inicio, max_tokens_completar)


def carregar_amostras_gpt_txt_diretorio(
    var diretorio: String,
    var max_tokens_inicio: Int = 128,
    var max_tokens_completar: Int = 128,
) -> List[gpt_txt_supervisionado.AmostraGPTSupervisionada]:
    return gpt_txt_supervisionado.carregar_amostras_gpt_txt_diretorio(diretorio, max_tokens_inicio, max_tokens_completar)


def gerar_lotes_gpt_supervisionado(
    amostras: List[gpt_txt_supervisionado.AmostraGPTSupervisionada],
    var tamanho_lote: Int = 4,
) -> List[gpt_txt_supervisionado.LoteGPTSupervisionado]:
    return gpt_txt_supervisionado.gerar_lotes_gpt_supervisionado(amostras, tamanho_lote)


def carregar_lotes_gpt_supervisionado_txt(
    var caminho_arquivo: String,
    var tamanho_lote: Int = 4,
    var max_tokens_inicio: Int = 128,
    var max_tokens_completar: Int = 128,
) -> List[gpt_txt_supervisionado.LoteGPTSupervisionado]:
    return gpt_txt_supervisionado.carregar_lotes_gpt_supervisionado_txt(
        caminho_arquivo,
        tamanho_lote,
        max_tokens_inicio,
        max_tokens_completar,
    )


def carregar_lotes_gpt_supervisionado_diretorio(
    var diretorio: String,
    var tamanho_lote: Int = 4,
    var max_tokens_inicio: Int = 128,
    var max_tokens_completar: Int = 128,
) -> List[gpt_txt_supervisionado.LoteGPTSupervisionado]:
    return gpt_txt_supervisionado.carregar_lotes_gpt_supervisionado_diretorio(
        diretorio,
        tamanho_lote,
        max_tokens_inicio,
        max_tokens_completar,
    )


def gerar_amostras_janela_token_gpt(
    amostras: List[gpt_txt_supervisionado.AmostraGPTSupervisionada],
    var tamanho_contexto: Int = 24,
    var passo: Int = 1,
) -> List[gpt_txt_supervisionado.AmostraGPTJanelaToken]:
    return gpt_txt_supervisionado.gerar_amostras_janela_token_gpt(amostras, tamanho_contexto, passo)


def gerar_lotes_janela_token_gpt(
    amostras_janela: List[gpt_txt_supervisionado.AmostraGPTJanelaToken],
    var tamanho_lote: Int = 8,
) -> List[gpt_txt_supervisionado.LoteGPTJanelaToken]:
    return gpt_txt_supervisionado.gerar_lotes_janela_token_gpt(amostras_janela, tamanho_lote)


def carregar_lotes_janela_token_gpt_txt(
    var caminho_arquivo: String,
    var tamanho_contexto: Int = 24,
    var passo: Int = 1,
    var tamanho_lote: Int = 8,
    var max_tokens_inicio: Int = 256,
    var max_tokens_completar: Int = 256,
) -> List[gpt_txt_supervisionado.LoteGPTJanelaToken]:
    return gpt_txt_supervisionado.carregar_lotes_janela_token_gpt_txt(
        caminho_arquivo,
        tamanho_contexto,
        passo,
        tamanho_lote,
        max_tokens_inicio,
        max_tokens_completar,
    )


def carregar_lotes_janela_token_gpt_diretorio(
    var diretorio: String,
    var tamanho_contexto: Int = 24,
    var passo: Int = 1,
    var tamanho_lote: Int = 8,
    var max_tokens_inicio: Int = 256,
    var max_tokens_completar: Int = 256,
) -> List[gpt_txt_supervisionado.LoteGPTJanelaToken]:
    return gpt_txt_supervisionado.carregar_lotes_janela_token_gpt_diretorio(
        diretorio,
        tamanho_contexto,
        passo,
        tamanho_lote,
        max_tokens_inicio,
        max_tokens_completar,
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