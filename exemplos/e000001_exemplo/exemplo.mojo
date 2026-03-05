import src.dados as dados_pkg
import src.dados.arquivo as dados_io
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
    var caminho_bmp = "exemplos/e000001_exemplo/dados.bmp"
    var bmp_bytes = dados_io.ler_arquivo_binario(caminho_bmp)
    print("Diagnóstico BMP: bytes lidos=", len(bmp_bytes))
    if len(bmp_bytes) >= 2:
        print("Diagnóstico BMP: assinatura=", bmp_bytes[0], bmp_bytes[1])
    var bmp_info = dados_pkg.carregar_bmp(caminho_bmp)

    var imagem_para_normalizar = List[List[Float32]]()
    if bmp_info.width != 0:
        print("Arquivo BMP encontrado:", caminho_bmp, "w=", bmp_info.width, "h=", bmp_info.height, "bpp=", bmp_info.bits_per_pixel)
        for y in range(bmp_info.height):
            var linha = List[Float32]()
            for x in range(bmp_info.width):
                var px = bmp_info.pixels[y][x]
                var gray = (px[0] + px[1] + px[2]) / 3.0
                linha.append(gray)
            imagem_para_normalizar.append(linha)
    else:
        print("Arquivo BMP não encontrado ou inválido — usando imagem simulada")
        var row1 = List[Float32]()
        row1.append(0.0)
        row1.append(128.0)
        var row2 = List[Float32]()
        row2.append(255.0)
        row2.append(64.0)
        imagem_para_normalizar.append(row1)
        imagem_para_normalizar.append(row2)

    var img_mm = dados_pkg.normalizar_min_max(imagem_para_normalizar.copy())
    print("Imagem normalizada (Min-Max):")
    print_helpers.imprimir_matriz_float(img_mm.dados_normalizados.copy())

    # --- Exemplo de áudio ---
    print("\nExemplo de processamento de áudio:")
    var caminho_wav = "exemplos/e000001_exemplo/dados.wav"
    var wav_bytes = dados_io.ler_arquivo_binario(caminho_wav)
    print("Diagnóstico WAV: bytes lidos=", len(wav_bytes))
    if len(wav_bytes) >= 12:
        print("Diagnóstico WAV: assinatura=", wav_bytes[0], wav_bytes[1], wav_bytes[2], wav_bytes[3], "/", wav_bytes[8], wav_bytes[9], wav_bytes[10], wav_bytes[11])
    var wav_info = dados_pkg.carregar_wav(caminho_wav)

    var audio_para_normalizar = List[List[Float32]]()
    if wav_info.sample_rate != 0:
        print("Arquivo WAV encontrado:", caminho_wav, "sr=", wav_info.sample_rate, "ch=", wav_info.num_channels, "bps=", wav_info.bits_per_sample)
        for i in range(len(wav_info.samples)):
            var frame = List[Float32]()
            if len(wav_info.samples[i]) > 0:
                frame.append(wav_info.samples[i][0])
            else:
                frame.append(0.0)
            audio_para_normalizar.append(frame)
    else:
        print("Arquivo WAV não encontrado ou inválido — usando áudio simulado")
        var frame0 = List[Float32]()
        frame0.append(0.1)
        frame0.append(-0.2)
        frame0.append(0.3)
        audio_para_normalizar.append(frame0)
        var frame1 = List[Float32]()
        frame1.append(-0.1)
        frame1.append(0.2)
        frame1.append(-0.05)
        audio_para_normalizar.append(frame1)

    var audio_zs = dados_pkg.normalizar_zscore(audio_para_normalizar.copy())
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
