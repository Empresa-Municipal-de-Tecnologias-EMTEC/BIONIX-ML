# Pacote `dados` (lowercase) - utilitários para leitura e normalização de dados
import src.dados.csv as csv
import src.dados.wav as wav
import src.dados.bmp as bmp
import src.dados.normalizacao as normalizacao
import src.dados.conversao as conversao

def carregar_csv_de_texto(var texto: String, var delimitador: String = ",", var detectar_cabecalho: Bool = True):
    return csv.parse_csv(texto, delimitador, detectar_cabecalho)

def carregar_csv(var caminho: String, var delimitador: String = ",", var detectar_cabecalho: Bool = True):
    try:
        var texto = csv.ler_arquivo_texto(caminho)
        return csv.parse_csv(texto, delimitador, detectar_cabecalho)
    except Exception:
        return csv.parse_csv("", delimitador, detectar_cabecalho)

def carregar_wav(var caminho: String):
    try:
        return wav.parse_wav(caminho)
    except Exception:
        return None

def carregar_bmp(var caminho: String):
    try:
        return bmp.parse_bmp(caminho)
    except Exception:
        return None

def normalizar_min_max(var dados_numericos: List[List[Float32]]):
    return normalizacao.min_max_normalize(dados_numericos)

def normalizar_zscore(var dados_numericos: List[List[Float32]]):
    return normalizacao.z_score_normalize(dados_numericos)

def bmp_para_tensor(var bmp_info):
    return conversao.bmp_to_tensor(bmp_info)

def wav_para_tensor(var wav_info, var mixdown: Bool = true):
    return conversao.wav_to_tensor(wav_info, mixdown)
