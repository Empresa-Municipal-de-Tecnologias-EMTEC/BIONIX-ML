# Pacote `dados` (lowercase) - utilitários para leitura e normalização de dados
import src.dados.csv as csv
import src.dados.arquivo as arquivo
import src.dados.wav as wav
import src.dados.bmp as bmp
import src.dados.normalizacao as normalizacao
import src.dados.tipos_normalizacao as tipos_normalizacao
import src.dados.conversao as conversao

alias BMP_MODO_RGB = bmp.BMP_MODO_RGB
alias BMP_MODO_PRETO_BRANCO = bmp.BMP_MODO_PRETO_BRANCO
alias BMP_MODO_GRAYSCALE = bmp.BMP_MODO_GRAYSCALE
alias NormalizacaoPersistida = normalizacao.NormalizacaoPersistida

def carregar_csv_de_texto(var texto: String, var delimitador: String = ",", var detectar_cabecalho: Bool = True) -> csv.CSVData:
    return csv.parse_csv(texto, delimitador, detectar_cabecalho)

def carregar_csv(var caminho: String, var delimitador: String = ",", var detectar_cabecalho: Bool = True) -> csv.CSVData:
    var texto = arquivo.ler_arquivo_texto(caminho)
    return csv.parse_csv(texto, delimitador, detectar_cabecalho)


fn gravar_arquivo_binario(var caminho: String, var dados: List[Int]) -> Bool:
    return arquivo.gravar_arquivo_binario(caminho, dados)

def carregar_wav(var caminho: String) -> wav.WAVInfo:
    return wav.parse_wav(caminho)^

def carregar_bmp(var caminho: String, var modo: Int = BMP_MODO_GRAYSCALE) -> bmp.BMPInfo:
    return bmp.parse_bmp(caminho, modo)^

def carregar_bmp_rgb(var caminho: String) -> bmp.BMPInfo:
    return bmp.parse_bmp(caminho, BMP_MODO_RGB)^

def carregar_bmp_preto_branco(var caminho: String) -> bmp.BMPInfo:
    return bmp.parse_bmp(caminho, BMP_MODO_PRETO_BRANCO)^

def carregar_bmp_grayscale(var caminho: String) -> bmp.BMPInfo:
    return bmp.parse_bmp(caminho, BMP_MODO_GRAYSCALE)^

def carregar_bmp_grayscale_matriz(var caminho: String) -> List[List[Float32]]:
    return bmp.parse_bmp_grayscale_matrix(caminho)

def diagnosticar_wav(var caminho: String) -> Bool:
    return wav.diagnosticar_wav(caminho)

def diagnosticar_bmp(var caminho: String) -> Bool:
    return bmp.diagnosticar_bmp(caminho)

def normalizar_min_max(var dados_numericos: List[List[Float32]]) -> normalizacao.MinMaxResult:
    return normalizacao.min_max_normalize(dados_numericos^)

def normalizar_zscore(var dados_numericos: List[List[Float32]]) -> normalizacao.ZScoreResult:
    return normalizacao.z_score_normalize(dados_numericos^)

def criar_normalizacao_persistida(
    var tipo_entradas_id: Int,
    var media_entradas: List[Float32],
    var desvio_entradas: List[Float32],
    var tipo_alvo_id: Int,
    var media_alvo: Float32,
    var desvio_alvo: Float32,
) -> normalizacao.NormalizacaoPersistida:
    return normalizacao.criar_normalizacao_persistida(
        tipo_entradas_id,
        media_entradas,
        desvio_entradas,
        tipo_alvo_id,
        media_alvo,
        desvio_alvo,
    )

def normalizacao_nenhuma_id() -> Int:
    return tipos_normalizacao.normalizacao_nenhuma_id()

def normalizacao_minmax_id() -> Int:
    return tipos_normalizacao.normalizacao_minmax_id()

def normalizacao_zscore_id() -> Int:
    return tipos_normalizacao.normalizacao_zscore_id()

def normalizacao_nome_de_id(var normalizacao_id: Int) -> String:
    return tipos_normalizacao.normalizacao_nome_de_id(normalizacao_id)

def normalizacao_id_de_nome(var nome: String) -> Int:
    return tipos_normalizacao.normalizacao_id_de_nome(nome)

def normalizacao_id_valido(var normalizacao_id: Int) -> Bool:
    return tipos_normalizacao.normalizacao_id_valido(normalizacao_id)

def salvar_normalizacao_persistida(norm: normalizacao.NormalizacaoPersistida, var caminho: String):
    normalizacao.salvar_normalizacao_persistida(norm, caminho)

def carregar_normalizacao_persistida(var caminho: String) -> normalizacao.NormalizacaoPersistida:
    return normalizacao.carregar_normalizacao_persistida(caminho)

def normalizar_amostra_entradas(norm: normalizacao.NormalizacaoPersistida, amostra: List[Float32]) -> List[Float32]:
    return normalizacao.normalizar_amostra_entradas(norm, amostra)

def desnormalizar_valor_alvo(norm: normalizacao.NormalizacaoPersistida, valor_normalizado: Float32) -> Float32:
    return normalizacao.desnormalizar_valor_alvo(norm, valor_normalizado)

def bmp_para_tensor(var bmp_info):
    return conversao.bmp_to_tensor(bmp_info)

def wav_para_tensor(var wav_info, var mixdown: Bool = true):
    return conversao.wav_to_tensor(wav_info, mixdown)
