# Utilitários de I/O de texto com tratamento seguro de exceções

fn ler_texto_seguro(var caminho: String) -> String:
    try:
        var f = open(caminho, "r")
        var conteudo = f.read()
        f.close()
        return conteudo
    except Exception:
        return ""


fn gravar_texto_seguro(var caminho: String, var conteudo: String) -> Bool:
    try:
        var f = open(caminho, "w")
        f.write(conteudo)
        f.close()
        return True
    except Exception:
        return False