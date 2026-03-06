import src.uteis.texto as texto
import src.uteis.arquivo as arquivo
import src.uteis.kv as kv

fn parse_float_ascii(var texto_in: String) -> Float32:
    return texto.parse_float_ascii(texto_in)

fn float_list_para_csv(valores: List[Float32]) -> String:
    return texto.float_list_para_csv(valores)

fn split_csv_simples(var texto_in: String) -> List[String]:
    return texto.split_csv_simples(texto_in)

fn parse_linha_chave_valor(var linha: String) -> List[String]:
    return texto.parse_linha_chave_valor(linha)

fn ler_texto_seguro(var caminho: String) -> String:
    return arquivo.ler_texto_seguro(caminho)

fn gravar_texto_seguro(var caminho: String, var conteudo: String) -> Bool:
    return arquivo.gravar_texto_seguro(caminho, conteudo)


alias KVData = kv.KVData

fn parse_kv_texto(var conteudo: String) -> KVData:
    return kv.parse_kv_texto(conteudo)

fn kv_para_texto(chaves: List[String], valores: List[String]) -> String:
    return kv.kv_para_texto(chaves, valores)

fn carregar_kv_arquivo_seguro(var caminho: String) -> KVData:
    return kv.carregar_kv_arquivo_seguro(caminho)

fn salvar_kv_arquivo_seguro(var caminho: String, chaves: List[String], valores: List[String]) -> Bool:
    return kv.salvar_kv_arquivo_seguro(caminho, chaves, valores)

fn obter_valor_ou_padrao(kv_data: KVData, var chave: String, var padrao: String) -> String:
    return kv.obter_valor_ou_padrao(kv_data, chave, padrao)