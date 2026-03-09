import os
import src.conjuntos.csv_supervisionado as csv_sup
import src.dados as dados_pkg
import src.dados.tipos_normalizacao as norm_tipos
import src.nucleo.Tensor as tensor_defs


fn _termina_com_wav(var nome: String) -> Bool:
    if len(nome) < 4:
        return False
    var sufixo = nome[len(nome)-4:len(nome)]
    return sufixo == ".wav" or sufixo == ".WAV"


fn _conjunto_vazio(var tipo_computacao: String) -> csv_sup.ConjuntoSupervisionado:
    var formato_x_vazio = List[Int]()
    formato_x_vazio.append(0)
    formato_x_vazio.append(2)
    var formato_y_vazio = List[Int]()
    formato_y_vazio.append(0)
    formato_y_vazio.append(1)

    var cab = List[String]()
    cab.append("tempo")
    cab.append("amplitude")
    cab.append("classe")

    return csv_sup.ConjuntoSupervisionado(
        tensor_defs.Tensor(formato_x_vazio^, tipo_computacao),
        tensor_defs.Tensor(formato_y_vazio^, tipo_computacao),
        cab^,
        2,
        norm_tipos.normalizacao_nenhuma_id(),
        List[Float32](),
        List[Float32](),
        norm_tipos.normalizacao_nenhuma_id(),
        0.0,
        1.0,
    )^


fn _coletar_caminhos_wav(var caminho_ou_diretorio: String) -> List[String]:
    var caminhos = List[String]()
    try:
        if os.path.isdir(caminho_ou_diretorio):
            var nomes = os.listdir(caminho_ou_diretorio)
            for nome in nomes:
                var nome_str = String(nome)
                if _termina_com_wav(nome_str):
                    caminhos.append(os.path.join(caminho_ou_diretorio, nome_str))
        else:
            caminhos.append(caminho_ou_diretorio)
    except Exception:
        caminhos.append(caminho_ou_diretorio)

    return caminhos^


fn carregar_wav_supervisionado(
    var caminho_ou_diretorio: String,
    var tipo_computacao: String = "cpu",
    var stride: Int = 4,
    var limiar_amostra: Float32 = 0.05,
    var limiar_classe: Float32 = 0.35,
) -> csv_sup.ConjuntoSupervisionado:
    if stride <= 0:
        stride = 1

    var caminhos = _coletar_caminhos_wav(caminho_ou_diretorio)
    if len(caminhos) == 0:
        return _conjunto_vazio(tipo_computacao)

    var entradas_flat = List[Float32]()
    var alvos_flat = List[Float32]()
    var n = 0

    for caminho_wav in caminhos:
        var wav = dados_pkg.carregar_wav(caminho_wav)
        if wav.sample_rate <= 0 or len(wav.samples) == 0:
            continue

        var total_frames = len(wav.samples)
        for i in range(0, total_frames, stride):
            if len(wav.samples[i]) == 0:
                continue

            var amp = wav.samples[i][0]
            var amp_abs = amp if amp >= 0.0 else Float32(-amp)
            if amp_abs <= limiar_amostra:
                continue

            var t_norm = (Float32(i) / Float32(total_frames - 1)) * 2.0 - 1.0 if total_frames > 1 else 0.0
            entradas_flat.append(t_norm)
            entradas_flat.append(amp)
            alvos_flat.append(1.0 if amp_abs > limiar_classe else Float32(0.0))
            n = n + 1

    if n == 0:
        return _conjunto_vazio(tipo_computacao)

    var formato_x = List[Int]()
    formato_x.append(n)
    formato_x.append(2)
    var formato_y = List[Int]()
    formato_y.append(n)
    formato_y.append(1)

    var x_t = tensor_defs.Tensor(formato_x^, tipo_computacao)
    var y_t = tensor_defs.Tensor(formato_y^, tipo_computacao)

    for i in range(len(entradas_flat)):
        x_t.dados[i] = entradas_flat[i]
    for i in range(len(alvos_flat)):
        y_t.dados[i] = alvos_flat[i]

    var cab = List[String]()
    cab.append("tempo")
    cab.append("amplitude")
    cab.append("classe")

    return csv_sup.ConjuntoSupervisionado(
        x_t^,
        y_t^,
        cab^,
        2,
        norm_tipos.normalizacao_nenhuma_id(),
        List[Float32](),
        List[Float32](),
        norm_tipos.normalizacao_nenhuma_id(),
        0.0,
        1.0,
    )^
