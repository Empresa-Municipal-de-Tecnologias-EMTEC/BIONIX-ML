import src.dados as dados_pkg
import src.dados.print_helpers as print_helpers
import src.nucleo.nucleo as nucleo

fn _digit_value(var ch: String) -> Int:
    if ch == "0":
        return 0
    if ch == "1":
        return 1
    if ch == "2":
        return 2
    if ch == "3":
        return 3
    if ch == "4":
        return 4
    if ch == "5":
        return 5
    if ch == "6":
        return 6
    if ch == "7":
        return 7
    if ch == "8":
        return 8
    if ch == "9":
        return 9
    return -1

fn _parse_float_ascii(var texto: String) -> Float32:
    var s = texto.strip().replace(",", ".")
    if len(s) == 0:
        return 0.0

    var sinal: Float32 = 1.0
    var i: Int = 0
    if s[0:1] == "-":
        sinal = -1.0
        i = 1
    elif s[0:1] == "+":
        i = 1

    var inteiro: Float32 = 0.0
    while i < len(s):
        var ch = s[i:i+1]
        if ch == ".":
            i = i + 1
            break
        var d = _digit_value(ch)
        if d < 0:
            return sinal * inteiro
        inteiro = inteiro * 10.0 + Float32(d)
        i = i + 1

    var frac: Float32 = 0.0
    var base: Float32 = 1.0
    while i < len(s):
        var d = _digit_value(s[i:i+1])
        if d < 0:
            break
        frac = frac * 10.0 + Float32(d)
        base = base * 10.0
        i = i + 1

    return sinal * (inteiro + (frac / base))

def executar_exemplo():
    print("--- Exemplo e000001: leitura e normalização de dados ---")

    # Tenta carregar CSV do diretório do exemplo; se falhar, usa CSV embutido
    var caminho_csv = "exemplos/e000001_exemplo/dados.csv"
    var parsed = dados_pkg.carregar_csv(caminho_csv, ",", True)
    var usado_arquivo = True
    if len(parsed.linhas) == 0:
        usado_arquivo = False
        var csv_text = "x,y\n1.0,2.0\n2.0,4.1\n3.0,6.0\n4.0,8.1\n5.0,10.2\n"
        parsed = dados_pkg.carregar_csv_de_texto(csv_text, ",", True)

    print("Fonte CSV:", ("arquivo: " + caminho_csv) if usado_arquivo else "embutido")
    print("Cabeçalho detectado:")
    print_helpers.imprimir_cabecalho(parsed.cabecalho.copy())

    print("Linhas (raw):")
    print_helpers.imprimir_linhas_raw(parsed.linhas.copy(), 50)

    # Converter colunas para Float32 (assume todas as colunas numéricas neste exemplo)
    var dados_numericos = List[List[Float32]]()
    for r in parsed.linhas:
        var linha_numerica = True
        for j in range(len(r)):
            var campo_check = r[j].strip().replace(",", ".")
            if campo_check == "":
                linha_numerica = False
                break
            var i: Int = 0
            var zero = "0"[0:1]
            var nine = "9"[0:1]
            var dot = "."[0:1]
            var minus = "-"[0:1]
            var plus = "+"[0:1]
            var e_l = "e"[0:1]
            var e_U = "E"[0:1]
            while i < len(campo_check):
                var ch = campo_check[i:i+1]
                if not (ch >= zero and ch <= nine) and ch != dot and ch != minus and ch != plus and ch != e_l and ch != e_U:
                    linha_numerica = False
                    break
                i = i + 1
            if not linha_numerica:
                break
        if not linha_numerica:
            continue

        var linha = List[Float32](len(r))
        for j in range(len(r)):
            var campo = r[j].strip()
            var campo_clean = campo.replace(",", ".")
            linha[j] = _parse_float_ascii(campo_clean)
        dados_numericos.append(linha.copy())

    print("Matriz numérica (primeiras linhas):")
    print_helpers.imprimir_matriz_float(dados_numericos.copy(), 50)

    # Normalização Min-Max
    var mm = dados_pkg.normalizar_min_max(dados_numericos.copy())
    print("\n")
    print_helpers.imprimir_min_max(mm.copy())

    # Normalização Z-Score
    var zs = dados_pkg.normalizar_zscore(dados_numericos.copy())
    print("\n")
    print_helpers.imprimir_zscore(zs.copy())

    # --- Exemplo de imagem ---
    print("\nExemplo de processamento de imagem:")
    # Tenta carregar BMP a partir do pacote `dados`
    var caminho_bmp = "exemplos/e000001_exemplo/dados.BMP"
    var bmp_info = dados_pkg.carregar_bmp(caminho_bmp)
    if bmp_info.width != 0:
        print("Arquivo BMP encontrado:", caminho_bmp, "w=", bmp_info.width, "h=", bmp_info.height, "bpp=", bmp_info.bits_per_pixel)
    else:
        print("Arquivo BMP não encontrado ou inválido — usando imagem simulada")

    var imagem_simulada = List[List[Float32]]()
    var row1 = List[Float32](2)
    row1[0] = 0.0
    row1[1] = 128.0
    var row2 = List[Float32](2)
    row2[0] = 255.0
    row2[1] = 64.0
    imagem_simulada.append(row1.copy())
    imagem_simulada.append(row2.copy())

    var flat = List[List[Float32]]()
    for i in range(len(imagem_simulada)):
        var linha = List[Float32](len(imagem_simulada[i]))
        for j in range(len(imagem_simulada[i])):
            linha[j] = imagem_simulada[i][j]
        flat.append(linha.copy())

    var img_mm = dados_pkg.normalizar_min_max(flat.copy())
    print("Imagem normalizada (Min-Max):")
    print_helpers.imprimir_matriz_float(img_mm.dados_normalizados.copy())

    # --- Exemplo de áudio ---
    print("\nExemplo de processamento de áudio:")
    var caminho_wav = "exemplos/e000001_exemplo/dados.wav"
    var wav_info = dados_pkg.carregar_wav(caminho_wav)
    if wav_info.sample_rate != 0:
        print("Arquivo WAV encontrado:", caminho_wav, "sr=", wav_info.sample_rate, "ch=", wav_info.num_channels, "bps=", wav_info.bits_per_sample)
    else:
        print("Arquivo WAV não encontrado ou inválido — usando áudio simulado")

    var audio_simulado = List[List[Float32]]()
    audio_simulado.append(List[Float32](3))
    audio_simulado[0][0] = 0.1
    audio_simulado[0][1] = -0.2
    audio_simulado[0][2] = 0.3
    audio_simulado.append(List[Float32](3))
    audio_simulado[1][0] = -0.1
    audio_simulado[1][1] = 0.2
    audio_simulado[1][2] = -0.05

    var audio_zs = dados_pkg.normalizar_zscore(audio_simulado.copy())
    print("Áudio normalizado (Z-Score):")
    print_helpers.imprimir_matriz_float(audio_zs.dados_normalizados.copy())

    print("\n--- Fim do exemplo e000001 ---")

    # --- Exemplo do núcleo (operação tensorial) ---
    print("\n--- Exemplo do núcleo Bionix (operações tensoriais) ---")
    var formato_a = List[Int]()
    formato_a.append(2)
    formato_a.append(2)
    var formato_b = List[Int]()
    formato_b.append(2)
    formato_b.append(2)
    var a = nucleo.Tensor(formato_a^)
    var b = nucleo.Tensor(formato_b^)
    a.dados[0] = 1.0
    a.dados[1] = 2.0
    a.dados[2] = 3.0
    a.dados[3] = 4.0
    b.dados[0] = 0.5
    b.dados[1] = 1.5
    b.dados[2] = -1.0
    b.dados[3] = 2.0

    var soma = nucleo.somar(a, b)
    print("resultado da soma no exemplo:")
    for i in range(len(soma.dados)):
        print("  ", soma.dados[i])

    print("--- Fim do exemplo ---")
