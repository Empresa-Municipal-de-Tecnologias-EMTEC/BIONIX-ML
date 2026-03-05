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
        var f = open(caminho, "r")
        var raw = f.read_bytes(-1)
        f.close()
        var out = List[Int]()
        for by in raw:
            out.append(Int(by) & 0xFF)
        return out^
    except Exception:
        return List[Int]()
