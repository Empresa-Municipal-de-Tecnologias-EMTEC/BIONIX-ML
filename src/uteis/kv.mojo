import src.uteis.texto as texto
import src.uteis.arquivo as arquivo

# Estrutura simples para pares chave=valor em texto
struct KVData(Movable, Copyable):
    var chaves: List[String]
    var valores: List[String]

    fn __init__(out self, var chaves_in: List[String], var valores_in: List[String]):
        self.chaves = chaves_in^
        self.valores = valores_in^

    fn copy(self) -> KVData:
        return KVData(self.chaves.copy(), self.valores.copy())


fn parse_kv_texto(var conteudo: String) -> KVData:
    var chaves = List[String]()
    var valores = List[String]()

    var linha = ""
    for i in range(len(conteudo)):
        var c = conteudo[i:i+1]
        if c == "\n":
            var kv = texto.parse_linha_chave_valor(linha)
            if len(kv) == 2 and len(kv[0].strip()) > 0:
                chaves.append(kv[0])
                valores.append(kv[1])
            linha = ""
        else:
            linha = linha + c

    if len(linha.strip()) > 0:
        var kv_last = texto.parse_linha_chave_valor(linha)
        if len(kv_last) == 2 and len(kv_last[0].strip()) > 0:
            chaves.append(kv_last[0])
            valores.append(kv_last[1])

    return KVData(chaves^, valores^)


fn kv_para_texto(chaves: List[String], valores: List[String]) -> String:
    var out = ""
    var n = len(chaves)
    if len(valores) < n:
        n = len(valores)
    for i in range(n):
        out = out + chaves[i] + "=" + valores[i] + "\n"
    return out


fn carregar_kv_arquivo_seguro(var caminho: String) -> KVData:
    var conteudo = arquivo.ler_texto_seguro(caminho)
    if len(conteudo) == 0:
        return KVData(List[String](), List[String]())
    return parse_kv_texto(conteudo)


fn salvar_kv_arquivo_seguro(var caminho: String, chaves: List[String], valores: List[String]) -> Bool:
    var conteudo = kv_para_texto(chaves, valores)
    return arquivo.gravar_texto_seguro(caminho, conteudo)


fn obter_valor_ou_padrao(kv: KVData, var chave: String, var padrao: String) -> String:
    for i in range(len(kv.chaves)):
        if kv.chaves[i] == chave:
            return kv.valores[i]
    return padrao