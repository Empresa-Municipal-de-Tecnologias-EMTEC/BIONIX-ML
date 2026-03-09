import src.conjuntos.csv_supervisionado as csv_sup
import src.dados.tipos_normalizacao as norm_tipos
import src.nucleo.Tensor as tensor_defs

struct LoteSupervisionado(Movable, Copyable):
    var entradas: tensor_defs.Tensor
    var alvos: tensor_defs.Tensor

    fn __init__(out self, entradas_in: tensor_defs.Tensor, alvos_in: tensor_defs.Tensor):
        self.entradas = entradas_in.copy()
        self.alvos = alvos_in.copy()


struct LoteEpocaSupervisionado(Movable, Copyable):
    var epoca: Int
    var indice_lote: Int
    var lote: LoteSupervisionado

    fn __init__(out self, var epoca_in: Int, var indice_lote_in: Int, lote_in: LoteSupervisionado):
        self.epoca = epoca_in
        self.indice_lote = indice_lote_in
        self.lote = lote_in.copy()


struct PreparacaoTreinoValidacaoLotes(Movable, Copyable):
    var treino_por_epoca: List[LoteEpocaSupervisionado]
    var validacao: List[LoteSupervisionado]

    fn __init__(out self, var treino_in: List[LoteEpocaSupervisionado], var validacao_in: List[LoteSupervisionado]):
        self.treino_por_epoca = treino_in^
        self.validacao = validacao_in^


fn _conjunto_vazio(var tipo_computacao: String, var num_features: Int = 2) -> csv_sup.ConjuntoSupervisionado:
    var formato_x = List[Int]()
    formato_x.append(0)
    formato_x.append(num_features)
    var formato_y = List[Int]()
    formato_y.append(0)
    formato_y.append(1)

    return csv_sup.ConjuntoSupervisionado(
        tensor_defs.Tensor(formato_x^, tipo_computacao),
        tensor_defs.Tensor(formato_y^, tipo_computacao),
        List[String](),
        num_features,
        norm_tipos.normalizacao_nenhuma_id(),
        List[Float32](),
        List[Float32](),
        norm_tipos.normalizacao_nenhuma_id(),
        0.0,
        1.0,
    )^


fn _fatiar_conjunto(conjunto: csv_sup.ConjuntoSupervisionado, var inicio: Int, var fim: Int) -> csv_sup.ConjuntoSupervisionado:
    var total = conjunto.entradas.formato[0]
    var features = conjunto.entradas.formato[1]

    if inicio < 0:
        inicio = 0
    if fim > total:
        fim = total
    if fim <= inicio:
        return _conjunto_vazio(conjunto.entradas.tipo_computacao, features)

    var n = fim - inicio
    var formato_x = List[Int]()
    formato_x.append(n)
    formato_x.append(features)
    var formato_y = List[Int]()
    formato_y.append(n)
    formato_y.append(1)

    var x_t = tensor_defs.Tensor(formato_x^, conjunto.entradas.tipo_computacao)
    var y_t = tensor_defs.Tensor(formato_y^, conjunto.alvos.tipo_computacao)

    for i in range(n):
        var src_i = inicio + i
        for j in range(features):
            x_t.dados[i * features + j] = conjunto.entradas.dados[src_i * features + j]
        y_t.dados[i] = conjunto.alvos.dados[src_i]

    return csv_sup.ConjuntoSupervisionado(
        x_t^,
        y_t^,
        conjunto.cabecalho.copy(),
        conjunto.indice_alvo,
        conjunto.tipo_normalizacao_entradas_id,
        conjunto.media_entradas.copy(),
        conjunto.desvio_entradas.copy(),
        conjunto.tipo_normalizacao_alvo_id,
        conjunto.media_alvo,
        conjunto.desvio_alvo,
    )^


fn quebrar_em_lotes(conjunto: csv_sup.ConjuntoSupervisionado, var tamanho_lote: Int) -> List[LoteSupervisionado]:
    var lotes = List[LoteSupervisionado]()
    var total = conjunto.entradas.formato[0]
    if total <= 0:
        return lotes^
    if tamanho_lote <= 0:
        tamanho_lote = total

    var inicio = 0
    while inicio < total:
        var fim = inicio + tamanho_lote
        if fim > total:
            fim = total
        var sub = _fatiar_conjunto(conjunto, inicio, fim)
        lotes.append(LoteSupervisionado(sub.entradas, sub.alvos))
        inicio = fim

    return lotes^


fn dividir_treino_validacao(
    conjunto: csv_sup.ConjuntoSupervisionado,
    var proporcao_validacao: Float32 = 0.2,
) -> List[csv_sup.ConjuntoSupervisionado]:
    var partes = List[csv_sup.ConjuntoSupervisionado]()
    var total = conjunto.entradas.formato[0]
    if total <= 0:
        partes.append(_conjunto_vazio(conjunto.entradas.tipo_computacao, conjunto.entradas.formato[1]))
        partes.append(_conjunto_vazio(conjunto.entradas.tipo_computacao, conjunto.entradas.formato[1]))
        return partes^

    var p = proporcao_validacao
    if p < 0.0:
        p = 0.0
    if p > 0.9:
        p = 0.9

    var n_valid = Int(Float32(total) * p)
    if n_valid < 1 and total > 1 and p > 0.0:
        n_valid = 1
    if n_valid >= total:
        n_valid = total - 1

    var n_treino = total - n_valid

    partes.append(_fatiar_conjunto(conjunto, 0, n_treino))
    partes.append(_fatiar_conjunto(conjunto, n_treino, total))
    return partes^


fn preparar_treino_validacao_em_lotes(
    conjunto: csv_sup.ConjuntoSupervisionado,
    var tamanho_lote: Int,
    var epocas_treino: Int = 1,
    var proporcao_validacao: Float32 = 0.2,
) -> PreparacaoTreinoValidacaoLotes:
    if epocas_treino <= 0:
        epocas_treino = 1

    var partes = dividir_treino_validacao(conjunto, proporcao_validacao)
    var treino = partes[0].copy()
    var valid = partes[1].copy()

    var lotes_treino_base = quebrar_em_lotes(treino, tamanho_lote)
    var lotes_valid = quebrar_em_lotes(valid, tamanho_lote)

    var treino_por_epoca = List[LoteEpocaSupervisionado]()
    for ep in range(epocas_treino):
        for i in range(len(lotes_treino_base)):
            treino_por_epoca.append(LoteEpocaSupervisionado(ep, i, lotes_treino_base[i]))

    return PreparacaoTreinoValidacaoLotes(treino_por_epoca^, lotes_valid^)
