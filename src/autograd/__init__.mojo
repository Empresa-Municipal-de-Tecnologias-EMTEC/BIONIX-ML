import src.autograd.mlp as mlp
import src.autograd.grafo as grafo
import src.autograd.tipos_mlp as tipos_mlp
import src.nucleo.Tensor as tensor_defs

alias MLPForwardContext = mlp.MLPForwardContext
alias MLPGradientes = mlp.MLPGradientes
alias GrafoComputacao = grafo.GrafoComputacao


fn construir_contexto_mlp(
    entradas: tensor_defs.Tensor,
    alvos: tensor_defs.Tensor,
    pesos: List[tensor_defs.Tensor],
    biases: List[tensor_defs.Tensor],
    var ativacao_saida_id: Int = -1,
    var perda_id: Int = -1,
) -> MLPForwardContext:
    return mlp.construir_contexto(entradas, alvos, pesos, biases, ativacao_saida_id, perda_id)


fn calcular_gradientes_mlp(ctx: MLPForwardContext, pesos: List[tensor_defs.Tensor]) -> MLPGradientes:
    return mlp.calcular_gradientes(ctx, pesos)


fn calcular_loss_mlp(pred: tensor_defs.Tensor, alvos: tensor_defs.Tensor, var perda_id: Int) -> Float32:
    return mlp.calcular_loss_configurado(pred, alvos, perda_id)


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


fn adicionar_bias_vetor_coluna(a: tensor_defs.Tensor, b: tensor_defs.Tensor) -> tensor_defs.Tensor:
    return mlp.adicionar_bias_vetor_coluna(a, b)


fn criar_grafo_mlp_forward() -> GrafoComputacao:
    return grafo.criar_grafo_mlp_forward()
