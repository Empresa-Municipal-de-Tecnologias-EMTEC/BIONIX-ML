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


fn gravar_arquivo_binario(var caminho: String, var dados: List[Int]) -> Bool:
    try:
        var f = open(caminho, "w")
        var bytes_out = List[UInt8]()
        for v in dados:
            bytes_out.append(UInt8(v & 0xFF))
        f.write_bytes(bytes_out)
        f.close()
        return True
    except Exception:
        return False
