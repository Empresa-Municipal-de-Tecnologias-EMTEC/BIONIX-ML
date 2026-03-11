import src.camadas.mlp.mlp as mlp_impl
import src.autograd.tipos_mlp as tipos_mlp
import src.conjuntos.lotes_supervisionados as lotes_sup
import src.nucleo.Tensor as tensor_defs

alias BlocoMLP = mlp_impl.BlocoMLP


fn ativacao_saida_hard_sigmoid_id() -> Int:
    return tipos_mlp.ativacao_saida_hard_sigmoid_id()


fn ativacao_saida_linear_id() -> Int:
    return tipos_mlp.ativacao_saida_linear_id()


fn ativacao_saida_softmax_id() -> Int:
    return tipos_mlp.ativacao_saida_softmax_id()


fn perda_mse_id() -> Int:
    return tipos_mlp.perda_mse_id()


fn perda_cross_entropy_id() -> Int:
    return tipos_mlp.perda_cross_entropy_id()


fn ativacao_saida_id_valido(var ativacao_saida_id: Int) -> Bool:
    return tipos_mlp.ativacao_saida_id_valido(ativacao_saida_id)


fn perda_id_valido(var perda_id: Int) -> Bool:
    return tipos_mlp.perda_id_valido(perda_id)


fn ativacao_saida_nome_de_id(var ativacao_saida_id: Int) -> String:
    return tipos_mlp.ativacao_saida_nome_de_id(ativacao_saida_id)


fn perda_nome_de_id(var perda_id: Int) -> String:
    return tipos_mlp.perda_nome_de_id(perda_id)


def prever(bloco: BlocoMLP, entradas: tensor_defs.Tensor) -> tensor_defs.Tensor:
    return mlp_impl.prever(bloco, entradas)


def inferir(bloco: BlocoMLP, entradas: tensor_defs.Tensor) -> tensor_defs.Tensor:
    return mlp_impl.inferir(bloco, entradas)


def treinar(
    mut bloco: BlocoMLP,
    entradas: tensor_defs.Tensor,
    alvos: tensor_defs.Tensor,
    var taxa_aprendizado: Float32 = 0.03,
    var epocas: Int = 1200,
    var imprimir_cada: Int = 200,
    var manter_gradientes_na_ram_principal: Bool = False,
) -> Float32:
    return mlp_impl.treinar(
        bloco,
        entradas,
        alvos,
        taxa_aprendizado,
        epocas,
        imprimir_cada,
        manter_gradientes_na_ram_principal,
    )


def treinar_por_lotes(
    mut bloco: BlocoMLP,
    lotes_treino_por_epoca: List[lotes_sup.LoteEpocaSupervisionado],
    lotes_validacao: List[lotes_sup.LoteSupervisionado],
    var taxa_aprendizado: Float32 = 0.03,
    var imprimir_cada_epoca: Int = 1,
    var manter_gradientes_na_ram_principal: Bool = False,
) -> Float32:
    return mlp_impl.treinar_por_lotes(
        bloco,
        lotes_treino_por_epoca,
        lotes_validacao,
        taxa_aprendizado,
        imprimir_cada_epoca,
        manter_gradientes_na_ram_principal,
    )
