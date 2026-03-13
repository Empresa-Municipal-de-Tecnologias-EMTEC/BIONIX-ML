import src.dados as dados_pkg
import os


struct AmostraGPTSupervisionada(Movable, Copyable):
    var modo: String
    var inicio: String
    var completar: String
    var arquivo_origem: String
    var tokens_inicio: List[Int]
    var tokens_completar: List[Int]

    fn __init__(
        out self,
        var modo_in: String,
        var inicio_in: String,
        var completar_in: String,
        var arquivo_origem_in: String,
        var tokens_inicio_in: List[Int],
        var tokens_completar_in: List[Int],
    ):
        self.modo = modo_in^
        self.inicio = inicio_in^
        self.completar = completar_in^
        self.arquivo_origem = arquivo_origem_in^
        self.tokens_inicio = tokens_inicio_in^
        self.tokens_completar = tokens_completar_in^

    fn copy(self) -> AmostraGPTSupervisionada:
        return AmostraGPTSupervisionada(
            self.modo,
            self.inicio,
            self.completar,
            self.arquivo_origem,
            self.tokens_inicio.copy(),
            self.tokens_completar.copy(),
        )


struct LoteGPTSupervisionado(Movable, Copyable):
    var amostras: List[AmostraGPTSupervisionada]

    fn __init__(out self, var amostras_in: List[AmostraGPTSupervisionada]):
        self.amostras = amostras_in^

    fn copy(self) -> LoteGPTSupervisionado:
        var copia = List[AmostraGPTSupervisionada]()
        for a in self.amostras:
            copia.append(a.copy())
        return LoteGPTSupervisionado(copia)


struct AmostraGPTJanelaToken(Movable, Copyable):
    var modo: String
    var arquivo_origem: String
    var contexto_tokens: List[Int]
    var alvo_token: Int

    fn __init__(
        out self,
        var modo_in: String,
        var arquivo_origem_in: String,
        var contexto_tokens_in: List[Int],
        var alvo_token_in: Int,
    ):
        self.modo = modo_in^
        self.arquivo_origem = arquivo_origem_in^
        self.contexto_tokens = contexto_tokens_in^
        self.alvo_token = alvo_token_in

    fn copy(self) -> AmostraGPTJanelaToken:
        return AmostraGPTJanelaToken(
            self.modo,
            self.arquivo_origem,
            self.contexto_tokens.copy(),
            self.alvo_token,
        )


struct LoteGPTJanelaToken(Movable, Copyable):
    var amostras: List[AmostraGPTJanelaToken]

    fn __init__(out self, var amostras_in: List[AmostraGPTJanelaToken]):
        self.amostras = amostras_in^

    fn copy(self) -> LoteGPTJanelaToken:
        var copia = List[AmostraGPTJanelaToken]()
        for a in self.amostras:
            copia.append(a.copy())
        return LoteGPTJanelaToken(copia)


fn _starts_with(var s: String, var prefixo: String) -> Bool:
    if len(prefixo) > len(s):
        return False
    return s[0:len(prefixo)] == prefixo


fn _tokenizar_ascii(var texto: String, var limite: Int) -> List[Int]:
    var out = List[Int]()
    var n = len(texto)
    if limite > 0 and n > limite:
        n = limite
    for i in range(n):
        out.append(Int(ord(texto[i:i+1])) & 0xFF)
    return out^


fn _limpar_campos(mut modo: String, mut inicio: String, mut completar: String, mut campo_atual: String):
    modo = ""
    inicio = ""
    completar = ""
    campo_atual = ""


fn _anexar_amostra_se_valida(
    mut out: List[AmostraGPTSupervisionada],
    var modo: String,
    var inicio: String,
    var completar: String,
    var arquivo_origem: String,
    var max_tokens_inicio: Int,
    var max_tokens_completar: Int,
):
    var m = modo.strip()
    var i = inicio.strip()
    var c = completar.strip()
    if len(m) == 0 or len(i) == 0 or len(c) == 0:
        return

    var tk_i = _tokenizar_ascii(i, max_tokens_inicio)
    var tk_c = _tokenizar_ascii(c, max_tokens_completar)
    out.append(AmostraGPTSupervisionada(m, i, c, arquivo_origem, tk_i, tk_c))


fn carregar_amostras_gpt_txt(
    var caminho_arquivo: String,
    var max_tokens_inicio: Int = 128,
    var max_tokens_completar: Int = 128,
) -> List[AmostraGPTSupervisionada]:
    var linhas = dados_pkg.carregar_txt_linhas(caminho_arquivo)
    var out = List[AmostraGPTSupervisionada]()

    var modo = ""
    var inicio = ""
    var completar = ""
    var campo_atual = ""

    for linha in linhas:
        var l = linha.strip()

        if len(l) == 0 or l == "---":
            _anexar_amostra_se_valida(out, modo, inicio, completar, caminho_arquivo, max_tokens_inicio, max_tokens_completar)
            _limpar_campos(modo, inicio, completar, campo_atual)
            continue

        if _starts_with(l, "modo:"):
            modo = l[5:len(l)].strip()
            campo_atual = ""
            continue

        if _starts_with(l, "inicio:"):
            inicio = l[7:len(l)].strip()
            campo_atual = "inicio"
            continue

        if _starts_with(l, "completar:"):
            completar = l[10:len(l)].strip()
            campo_atual = "completar"
            continue

        # Continuidade de texto: sempre fica no mesmo arquivo, preservando pareamento local.
        if campo_atual == "inicio":
            if len(inicio) > 0:
                inicio = inicio + " " + l
            else:
                inicio = l
        elif campo_atual == "completar":
            if len(completar) > 0:
                completar = completar + " " + l
            else:
                completar = l

    _anexar_amostra_se_valida(out, modo, inicio, completar, caminho_arquivo, max_tokens_inicio, max_tokens_completar)
    return out^


fn carregar_amostras_gpt_txt_diretorio(
    var diretorio: String,
    var max_tokens_inicio: Int = 128,
    var max_tokens_completar: Int = 128,
) -> List[AmostraGPTSupervisionada]:
    var out = List[AmostraGPTSupervisionada]()

    try:
        if not os.path.isdir(diretorio):
            return out^

        var nomes = os.listdir(diretorio)
        for nome in nomes:
            if not nome.endswith(".txt"):
                continue
            var caminho = os.path.join(diretorio, nome)
            if not os.path.isfile(caminho):
                continue

            var locais = carregar_amostras_gpt_txt(caminho, max_tokens_inicio, max_tokens_completar)
            for a in locais:
                out.append(a.copy())
    except Exception:
        return List[AmostraGPTSupervisionada]()

    return out^


fn gerar_lotes_gpt_supervisionado(
    amostras: List[AmostraGPTSupervisionada],
    var tamanho_lote: Int = 4,
) -> List[LoteGPTSupervisionado]:
    var lotes = List[LoteGPTSupervisionado]()
    var total = len(amostras)
    if total <= 0:
        return lotes^

    if tamanho_lote <= 0:
        tamanho_lote = total

    var inicio = 0
    while inicio < total:
        var fim = inicio + tamanho_lote
        if fim > total:
            fim = total

        var bloco = List[AmostraGPTSupervisionada]()
        for i in range(inicio, fim):
            bloco.append(amostras[i].copy())
        lotes.append(LoteGPTSupervisionado(bloco))
        inicio = fim

    return lotes^


fn carregar_lotes_gpt_supervisionado_txt(
    var caminho_arquivo: String,
    var tamanho_lote: Int = 4,
    var max_tokens_inicio: Int = 128,
    var max_tokens_completar: Int = 128,
) -> List[LoteGPTSupervisionado]:
    var amostras = carregar_amostras_gpt_txt(caminho_arquivo, max_tokens_inicio, max_tokens_completar)
    return gerar_lotes_gpt_supervisionado(amostras, tamanho_lote)


fn carregar_lotes_gpt_supervisionado_diretorio(
    var diretorio: String,
    var tamanho_lote: Int = 4,
    var max_tokens_inicio: Int = 128,
    var max_tokens_completar: Int = 128,
) -> List[LoteGPTSupervisionado]:
    var amostras = carregar_amostras_gpt_txt_diretorio(diretorio, max_tokens_inicio, max_tokens_completar)
    return gerar_lotes_gpt_supervisionado(amostras, tamanho_lote)


fn _concatenar_tokens_amostra(a: AmostraGPTSupervisionada) -> List[Int]:
    var seq = List[Int]()
    for t in a.tokens_inicio:
        seq.append(t)
    for t in a.tokens_completar:
        seq.append(t)
    return seq^


fn gerar_amostras_janela_token_gpt(
    amostras: List[AmostraGPTSupervisionada],
    var tamanho_contexto: Int = 24,
    var passo: Int = 1,
) -> List[AmostraGPTJanelaToken]:
    var out = List[AmostraGPTJanelaToken]()
    if tamanho_contexto <= 0:
        tamanho_contexto = 24
    if passo <= 0:
        passo = 1

    # A janela e o alvo sao sempre gerados dentro da mesma amostra,
    # garantindo que contexto/completar permanecem no mesmo arquivo.
    for a in amostras:
        var seq = _concatenar_tokens_amostra(a)
        if len(seq) <= tamanho_contexto:
            continue

        var i = 0
        while i + tamanho_contexto < len(seq):
            var contexto = List[Int]()
            for j in range(i, i + tamanho_contexto):
                contexto.append(seq[j])
            var alvo = seq[i + tamanho_contexto]
            out.append(AmostraGPTJanelaToken(a.modo, a.arquivo_origem, contexto, alvo))
            i = i + passo

    return out^


fn gerar_lotes_janela_token_gpt(
    amostras_janela: List[AmostraGPTJanelaToken],
    var tamanho_lote: Int = 8,
) -> List[LoteGPTJanelaToken]:
    var lotes = List[LoteGPTJanelaToken]()
    if len(amostras_janela) == 0:
        return lotes^
    if tamanho_lote <= 0:
        tamanho_lote = len(amostras_janela)

    var inicio = 0
    while inicio < len(amostras_janela):
        var fim = inicio + tamanho_lote
        if fim > len(amostras_janela):
            fim = len(amostras_janela)

        var bloco = List[AmostraGPTJanelaToken]()
        for i in range(inicio, fim):
            bloco.append(amostras_janela[i].copy())
        lotes.append(LoteGPTJanelaToken(bloco))
        inicio = fim

    return lotes^


fn carregar_lotes_janela_token_gpt_txt(
    var caminho_arquivo: String,
    var tamanho_contexto: Int = 24,
    var passo: Int = 1,
    var tamanho_lote: Int = 8,
    var max_tokens_inicio: Int = 256,
    var max_tokens_completar: Int = 256,
) -> List[LoteGPTJanelaToken]:
    var amostras = carregar_amostras_gpt_txt(caminho_arquivo, max_tokens_inicio, max_tokens_completar)
    var janelas = gerar_amostras_janela_token_gpt(amostras, tamanho_contexto, passo)
    return gerar_lotes_janela_token_gpt(janelas, tamanho_lote)


fn carregar_lotes_janela_token_gpt_diretorio(
    var diretorio: String,
    var tamanho_contexto: Int = 24,
    var passo: Int = 1,
    var tamanho_lote: Int = 8,
    var max_tokens_inicio: Int = 256,
    var max_tokens_completar: Int = 256,
) -> List[LoteGPTJanelaToken]:
    var amostras = carregar_amostras_gpt_txt_diretorio(diretorio, max_tokens_inicio, max_tokens_completar)
    var janelas = gerar_amostras_janela_token_gpt(amostras, tamanho_contexto, passo)
    return gerar_lotes_janela_token_gpt(janelas, tamanho_lote)
