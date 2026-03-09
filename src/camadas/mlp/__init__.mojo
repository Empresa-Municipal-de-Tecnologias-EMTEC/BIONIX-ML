import src.camadas.mlp.mlp as mlp_impl
import src.conjuntos.lotes_supervisionados as lotes_sup
import src.nucleo.Tensor as tensor_defs

alias BlocoMLP = mlp_impl.BlocoMLP


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
    var manter_gradientes_na_ram_principal: Bool = True,
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
    var manter_gradientes_na_ram_principal: Bool = True,
) -> Float32:
    return mlp_impl.treinar_por_lotes(
        bloco,
        lotes_treino_por_epoca,
        lotes_validacao,
        taxa_aprendizado,
        imprimir_cada_epoca,
        manter_gradientes_na_ram_principal,
    )
