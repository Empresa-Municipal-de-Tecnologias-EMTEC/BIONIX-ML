import src.camadas.linear as linear_pkg
import src.conjuntos as conjuntos_pkg
import src.conjuntos.csv_supervisionado as conjuntos_csv
import src.nucleo.Tensor as tensor_defs

struct NormalizacaoPersistida(Movable, Copyable):
    var tipo_entradas: String
    var media_entradas: List[Float32]
    var desvio_entradas: List[Float32]
    var tipo_alvo: String
    var media_alvo: Float32
    var desvio_alvo: Float32

    fn __init__(
        out self,
        var tipo_entradas_in: String,
        var media_entradas_in: List[Float32],
        var desvio_entradas_in: List[Float32],
        var tipo_alvo_in: String,
        var media_alvo_in: Float32,
        var desvio_alvo_in: Float32,
    ):
        self.tipo_entradas = tipo_entradas_in^
        self.media_entradas = media_entradas_in^
        self.desvio_entradas = desvio_entradas_in^
        self.tipo_alvo = tipo_alvo_in^
        self.media_alvo = media_alvo_in
        self.desvio_alvo = desvio_alvo_in


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


fn _float_list_para_texto(valores: List[Float32]) -> String:
    var out = ""
    for i in range(len(valores)):
        out = out + String(valores[i])
        if i < len(valores) - 1:
            out = out + ","
    return out


fn _split_csv_simples(var texto: String) -> List[String]:
    var itens = List[String]()
    var buffer = ""
    for i in range(len(texto)):
        var c = texto[i:i+1]
        if c == ",":
            itens.append(buffer)
            buffer = ""
        else:
            buffer = buffer + c
    itens.append(buffer)
    return itens^


fn _parse_linha_chave_valor(var linha: String) -> List[String]:
    var idx: Int = -1
    for i in range(len(linha)):
        if linha[i:i+1] == "=":
            idx = i
            break
    var out = List[String]()
    if idx < 0:
        out.append(linha)
        out.append("")
        return out^
    out.append(linha[0:idx])
    out.append(linha[idx + 1:len(linha)])
    return out^


fn salvar_normalizacao(conjunto: conjuntos_csv.ConjuntoSupervisionado, var caminho: String):
    try:
        var f = open(caminho, "w")
        f.write("tipo_entradas=" + conjunto.tipo_normalizacao_entradas + "\n")
        f.write("media_entradas=" + _float_list_para_texto(conjunto.media_entradas.copy()) + "\n")
        f.write("desvio_entradas=" + _float_list_para_texto(conjunto.desvio_entradas.copy()) + "\n")
        f.write("tipo_alvo=" + conjunto.tipo_normalizacao_alvo + "\n")
        f.write("media_alvo=" + String(conjunto.media_alvo) + "\n")
        f.write("desvio_alvo=" + String(conjunto.desvio_alvo) + "\n")
        f.close()
    except Exception:
        pass


fn carregar_normalizacao(var caminho: String) -> NormalizacaoPersistida:
    var conteudo = ""
    try:
        var f = open(caminho, "r")
        conteudo = f.read()
        f.close()
    except Exception:
        return NormalizacaoPersistida("nenhuma", List[Float32](), List[Float32](), "nenhuma", 0.0, 1.0)^

    var tipo_entradas = "nenhuma"
    var medias = List[Float32]()
    var desvios = List[Float32]()
    var tipo_alvo = "nenhuma"
    var media_alvo: Float32 = 0.0
    var desvio_alvo: Float32 = 1.0

    var linha = ""
    for i in range(len(conteudo)):
        var c = conteudo[i:i+1]
        if c == "\n":
            var kv = _parse_linha_chave_valor(linha)
            if len(kv) == 2:
                if kv[0] == "tipo_entradas":
                    tipo_entradas = kv[1]
                elif kv[0] == "media_entradas":
                    var itens = _split_csv_simples(kv[1])
                    for it in itens:
                        if len(it.strip()) > 0:
                            medias.append(_parse_float_ascii(it))
                elif kv[0] == "desvio_entradas":
                    var itens_d = _split_csv_simples(kv[1])
                    for it in itens_d:
                        if len(it.strip()) > 0:
                            desvios.append(_parse_float_ascii(it))
                elif kv[0] == "tipo_alvo":
                    tipo_alvo = kv[1]
                elif kv[0] == "media_alvo":
                    media_alvo = _parse_float_ascii(kv[1])
                elif kv[0] == "desvio_alvo":
                    desvio_alvo = _parse_float_ascii(kv[1])
            linha = ""
        else:
            linha = linha + c

    return NormalizacaoPersistida(tipo_entradas, medias^, desvios^, tipo_alvo, media_alvo, desvio_alvo)^


fn normalizar_amostra_com_persistencia(norm: NormalizacaoPersistida, amostra: List[Float32]) -> List[Float32]:
    if norm.tipo_entradas != "zscore":
        return amostra.copy()
    var out = List[Float32]()
    for i in range(len(amostra)):
        if i >= len(norm.media_entradas) or i >= len(norm.desvio_entradas):
            out.append(amostra[i])
            continue
        var d = norm.desvio_entradas[i]
        if d == 0.0:
            out.append(0.0)
        else:
            out.append((amostra[i] - norm.media_entradas[i]) / d)
    return out^


fn desnormalizar_alvo_com_persistencia(norm: NormalizacaoPersistida, valor_normalizado: Float32) -> Float32:
    if norm.tipo_alvo != "zscore":
        return valor_normalizado
    return valor_normalizado * norm.desvio_alvo + norm.media_alvo

fn _criar_tensor_uma_linha(var valores: List[Float32], var tipo_computacao: String) -> tensor_defs.Tensor:
    var formato = List[Int]()
    formato.append(1)
    formato.append(len(valores))
    var t = tensor_defs.Tensor(formato^, tipo_computacao)
    for i in range(len(valores)):
        t.dados[i] = valores[i]
    return t^


def executar_exemplo():
    print("--- Exemplo e000002: modelo linear com Tensor parametrizável ---")

    var tipo_computacao = "cpu"
    var caminho_csv = "exemplos/e000002_modelo_linear/dados.csv"
    var caminho_pesos = "exemplos/e000002_modelo_linear/pesos_linear.txt"
    var caminho_norm = "exemplos/e000002_modelo_linear/normalizacao_linear.txt"

    var conjunto = conjuntos_pkg.carregar_csv_supervisionado(caminho_csv, -1, ",", True, tipo_computacao, "zscore", "zscore")
    if len(conjunto.entradas.dados) == 0 or conjunto.entradas.formato[1] == 0:
        print("Falha ao carregar conjunto supervisionado do CSV:", caminho_csv)
        return

    print("Tipo de computação do tensor:", conjunto.entradas.tipo_computacao)
    print("Amostras:", conjunto.entradas.formato[0], "| Features:", conjunto.entradas.formato[1])
    print("Normalização das entradas:", conjunto.tipo_normalizacao_entradas)
    print("Normalização do alvo:", conjunto.tipo_normalizacao_alvo)

    print("[debug] Criando camada linear...")
    var camada = linear_pkg.CamadaLinear(conjunto.entradas.formato[1], tipo_computacao)
    print("[debug] Camada criada. Iniciando treinamento...")

    print("Treinando modelo linear...")
    var loss_final = linear_pkg.treinar(camada, conjunto.entradas, conjunto.alvos, 0.0001, 4000, 500)
    print("Loss final:", loss_final)

    linear_pkg.salvar_pesos(camada, caminho_pesos)
    print("Pesos salvos em:", caminho_pesos)
    salvar_normalizacao(conjunto, caminho_norm)
    print("Normalização salva em:", caminho_norm)

    var modelo_carregado = linear_pkg.carregar_pesos(caminho_pesos, tipo_computacao)
    var norm_carregada = carregar_normalizacao(caminho_norm)
    print("Modelo recarregado. Bias:", modelo_carregado.bias.dados[0])
    print("Normalização recarregada. Tipo entradas:", norm_carregada.tipo_entradas, "| Tipo alvo:", norm_carregada.tipo_alvo)

    var amostra = List[Float32]()
    amostra.append(78.0)
    amostra.append(3.0)
    amostra.append(2.0)
    var amostra_norm = normalizar_amostra_com_persistencia(norm_carregada, amostra)
    var entrada_nova = _criar_tensor_uma_linha(amostra_norm^, tipo_computacao)
    var pred = linear_pkg.inferir(modelo_carregado, entrada_nova)
    var pred_escala_original = desnormalizar_alvo_com_persistencia(norm_carregada, pred.dados[0])

    print("Inferência normalizada para [area=78, quartos=3, idade=2]:", pred.dados[0])
    print("Inferência em escala original (preço):", pred_escala_original)
    print("--- Fim do exemplo e000002 ---")