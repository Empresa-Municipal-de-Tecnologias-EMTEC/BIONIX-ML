# Pacote `dados` (lowercase) - utilitários para leitura e normalização de dados
import src.dados.csv as csv
import src.dados.arquivo as arquivo
import src.dados.wav as wav
import src.dados.bmp as bmp
import src.dados.normalizacao as normalizacao
import src.dados.conversao as conversao

def carregar_csv_de_texto(var texto: String, var delimitador: String = ",", var detectar_cabecalho: Bool = True) -> csv.CSVData:
    return csv.parse_csv(texto, delimitador, detectar_cabecalho)

def carregar_csv(var caminho: String, var delimitador: String = ",", var detectar_cabecalho: Bool = True) -> csv.CSVData:
    var texto = arquivo.ler_arquivo_texto(caminho)
    return csv.parse_csv(texto, delimitador, detectar_cabecalho)

def carregar_wav(var caminho: String) -> wav.WAVInfo:
    return wav.parse_wav(caminho)^

def carregar_bmp(var caminho: String) -> bmp.BMPInfo:
    return bmp.parse_bmp(caminho)^

def normalizar_min_max(var dados_numericos: List[List[Float32]]) -> normalizacao.MinMaxResult:
    return normalizacao.min_max_normalize(dados_numericos.copy())

def normalizar_zscore(var dados_numericos: List[List[Float32]]) -> normalizacao.ZScoreResult:
    return normalizacao.z_score_normalize(dados_numericos.copy())

def bmp_para_tensor(var bmp_info):
    return conversao.bmp_to_tensor(bmp_info)

def wav_para_tensor(var wav_info, var mixdown: Bool = true):
    return conversao.wav_to_tensor(wav_info, mixdown)
