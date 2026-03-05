# Utilitários de I/O de arquivos (texto e binário)

fn ler_arquivo_texto(var caminho: String) -> String:
    try:
        var f = open(caminho, "r")
        var conteudo = f.read()
        f.close()
        return conteudo
    except Exception:
        return ""

fn ler_arquivo_binario(var caminho: String) -> List[Int]:
    try:
        var f = open(caminho, "rb")
        var raw = f.read()
        f.close()
        var out = List[Int]()
        for ch in raw:
            try:
                out.append(Int(ch) & 0xFF)
            except Exception:
                out.append(ord(ch) & 0xFF)
        return out^
    except Exception:
        return List[Int]()
