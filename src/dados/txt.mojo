# Parser TXT simples para carregar texto bruto e linhas.
import src.dados.arquivo as arquivo


struct TXTData(Movable, Copyable):
    var texto_completo: String
    var linhas: List[String]

    fn __init__(out self, var texto_in: String, var linhas_in: List[String]):
        self.texto_completo = texto_in
        self.linhas = linhas_in^

    fn copy(self) -> TXTData:
        return TXTData(self.texto_completo, self.linhas.copy())


fn _split_linhas(var texto: String) -> List[String]:
    var out = List[String]()
    var buffer = ""
    for i in range(len(texto)):
        var c = texto[i:i+1]
        if c == "\n":
            if len(buffer.strip()) > 0:
                out.append(buffer)
            buffer = ""
        else:
            buffer = buffer + c
    if len(buffer.strip()) > 0:
        out.append(buffer)
    return out^


fn carregar_txt(var caminho: String) -> TXTData:
    var texto = arquivo.ler_arquivo_texto(caminho)
    var linhas = _split_linhas(texto)
    return TXTData(texto, linhas)^


fn carregar_txt_texto(var caminho: String) -> String:
    return arquivo.ler_arquivo_texto(caminho)


fn carregar_txt_linhas(var caminho: String) -> List[String]:
    return _split_linhas(arquivo.ler_arquivo_texto(caminho))
