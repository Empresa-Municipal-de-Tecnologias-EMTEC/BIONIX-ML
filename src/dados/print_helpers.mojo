from .csv import CSVData
from .normalizacao import MinMaxResult, ZScoreResult

fn imprimir_cabecalho(var cabecalho: List[String]):
    if len(cabecalho) == 0:
        print("  (nenhum cabeçalho)")
        return
    for i in range(len(cabecalho)):
        print("  ", cabecalho[i])

fn imprimir_linhas_raw(var linhas: List[List[String]], var max_linhas: Int = -1):
    var count = 0
    for r in linhas:
        if max_linhas != -1 and count >= max_linhas:
            break
        var s = ""
        for f in r:
            s = s + f + " "
        print("  ", s)
        count = count + 1

fn imprimir_matriz_float(var matriz: List[List[Float32]], var max_linhas: Int = -1):
    var count = 0
    for i in range(len(matriz)):
        if max_linhas != -1 and count >= max_linhas:
            break
        var l = matriz[i].copy()
        # Print values separated by space
        var linha_prefix = "  "
        var first = True
        for j in range(len(l)):
            if first:
                print(l[j])
                first = False
            else:
                print(" ", l[j])
        count = count + 1

fn imprimir_min_max(var mm: MinMaxResult):
    print("Min-Max: valores normalizados (primeiras linhas):")
    imprimir_matriz_float(mm.dados_normalizados, 20)
    print("Min por coluna:")
    for v in mm.minimo_por_coluna:
        print("  ", v)
    print("Max por coluna:")
    for v in mm.maximo_por_coluna:
        print("  ", v)

fn imprimir_zscore(var zs: ZScoreResult):
    print("Z-Score: valores normalizados (primeiras linhas):")
    imprimir_matriz_float(zs.dados_normalizados, 20)
    print("Médias por coluna:")
    for v in zs.media_por_coluna:
        print("  ", v)
    print("Desvios por coluna:")
    for v in zs.desvio_por_coluna:
        print("  ", v)
