# Utilitários de I/O de arquivos (texto e binário)
import src.uteis.arquivo as uteis_arquivo

fn ler_arquivo_texto(var caminho: String) -> String:
    return uteis_arquivo.ler_texto_seguro(caminho)

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
