import src.autograd as autograd
import src.autograd.tipos_mlp as tipos_mlp
import src.computacao.dispatcher_gradiente as dispatcher_gradiente
import src.conjuntos.lotes_supervisionados as lotes_sup
import src.nucleo.Tensor as tensor_defs
import math

struct BlocoMLP(Movable, Copyable):
    var topologia: List[Int]
    var pesos: List[tensor_defs.Tensor]
    var biases: List[tensor_defs.Tensor]
    var tipo_computacao: String
    var ativacao_saida_id: Int
    var perda_id: Int

    fn __init__(out self, var num_entradas: Int, var num_ocultas: Int = 16, var tipo_computacao_in: String = "cpu"):
        var topologia_local = List[Int]()
        topologia_local.append(num_entradas)
        topologia_local.append(num_ocultas)
        topologia_local.append(num_ocultas)
        topologia_local.append(1)

        _ = self.__init__(topologia_local^, tipo_computacao_in, -1, -1)

    fn __init__(
        out self,
        var topologia_in: List[Int],
        var tipo_computacao_in: String = "cpu",
        var ativacao_saida_id_in: Int = -1,
        var perda_id_in: Int = -1,
    ):
        debug_assert(len(topologia_in) >= 2, "topologia deve ter ao menos entrada e saída")

        self.topologia = topologia_in^
        self.tipo_computacao = tipo_computacao_in^
        self.pesos = List[tensor_defs.Tensor]()
        self.biases = List[tensor_defs.Tensor]()

        var num_saidas = self.topologia[len(self.topologia) - 1]
        self.ativacao_saida_id = ativacao_saida_id_in
        self.perda_id = perda_id_in
        if not tipos_mlp.ativacao_saida_id_valido(self.ativacao_saida_id):
            self.ativacao_saida_id = tipos_mlp.ativacao_saida_padrao_id(num_saidas)
        if not tipos_mlp.perda_id_valido(self.perda_id):
            self.perda_id = tipos_mlp.perda_padrao_id(num_saidas)

        debug_assert(num_saidas > 1 or self.ativacao_saida_id != tipos_mlp.ativacao_saida_softmax_id(), "softmax requer pelo menos 2 saídas")
        debug_assert(self.perda_id != tipos_mlp.perda_cross_entropy_id() or self.ativacao_saida_id == tipos_mlp.ativacao_saida_softmax_id(), "cross-entropy requer ativação de saída softmax")

        var num_camadas = len(self.topologia) - 1
        for camada in range(num_camadas):
            var formato_w = List[Int]()
            formato_w.append(self.topologia[camada])
            formato_w.append(self.topologia[camada + 1])
            var w = tensor_defs.Tensor(formato_w^, self.tipo_computacao)

            var formato_b = List[Int]()
            formato_b.append(1)
            formato_b.append(self.topologia[camada + 1])
            var b = tensor_defs.Tensor(formato_b^, self.tipo_computacao)

            var fan_in = self.topologia[camada]
            var fan_out = self.topologia[camada + 1]
            var limite = Float32(math.sqrt(6.0 / Float64(fan_in + fan_out)))

            var seed = (fan_in * 1103515245 + fan_out * 12345 + (camada + 1) * 97) % 2147483647
            if seed <= 0:
                seed = 1234567 + camada

            for i in range(len(w.dados)):
                seed = (seed * 1664525 + 1013904223 + i) % 2147483647
                var u01 = Float32(seed) / Float32(2147483647)
                var normalizado = u01 * 2.0 - 1.0
                w.dados[i] = normalizado * limite
            for i in range(len(b.dados)):
                b.dados[i] = 0.01 if camada < num_camadas - 1 else Float32(0.0)

            self.pesos.append(w.copy())
            self.biases.append(b.copy())

    fn copy(self) -> BlocoMLP:
        var copia = BlocoMLP(self.topologia.copy(), self.tipo_computacao, self.ativacao_saida_id, self.perda_id)
        copia.pesos = List[tensor_defs.Tensor]()
        for w in self.pesos:
            copia.pesos.append(w.copy())
        copia.biases = List[tensor_defs.Tensor]()
        for b in self.biases:
            copia.biases.append(b.copy())
        return copia^


fn prever(bloco: BlocoMLP, entradas: tensor_defs.Tensor) -> tensor_defs.Tensor:
    debug_assert(len(entradas.formato) == 2, "entradas deve ser tensor 2D")
    var saidas = bloco.topologia[len(bloco.topologia) - 1]
    var formato_y = List[Int]()
    formato_y.append(entradas.formato[0])
    formato_y.append(saidas)
    var alvos_dummy = tensor_defs.Tensor(formato_y^, entradas.tipo_computacao)
    var ctx = autograd.construir_contexto_mlp(entradas, alvos_dummy, bloco.pesos, bloco.biases, bloco.ativacao_saida_id, bloco.perda_id)
    return ctx.pred.copy()


fn inferir(bloco: BlocoMLP, entradas: tensor_defs.Tensor) -> tensor_defs.Tensor:
    return prever(bloco, entradas)


fn _loss_medio_lotes_validacao(bloco: BlocoMLP, lotes_validacao: List[lotes_sup.LoteSupervisionado]) -> Float32:
    if len(lotes_validacao) == 0:
        return 0.0

    var soma: Float32 = 0.0
    for lote in lotes_validacao:
        var pred = inferir(bloco, lote.entradas)
        soma = soma + autograd.calcular_loss_mlp(pred, lote.alvos, bloco.perda_id)

    return soma / Float32(len(lotes_validacao))


fn treinar_por_lotes(
    mut bloco: BlocoMLP,
    lotes_treino_por_epoca: List[lotes_sup.LoteEpocaSupervisionado],
    lotes_validacao: List[lotes_sup.LoteSupervisionado],
    var taxa_aprendizado: Float32 = 0.03,
    var imprimir_cada_epoca: Int = 1,
    var manter_gradientes_na_ram_principal: Bool = False,
) -> Float32:
    if len(lotes_treino_por_epoca) == 0:
        return 0.0

    var loss_lote_final: Float32 = 0.0
    var epoca_atual = lotes_treino_por_epoca[0].epoca
    var soma_loss_epoca: Float32 = 0.0
    var quantidade_lotes_epoca: Int = 0

    for item in lotes_treino_por_epoca:
        if item.epoca != epoca_atual:
            var loss_medio_epoca = soma_loss_epoca / Float32(quantidade_lotes_epoca) if quantidade_lotes_epoca > 0 else 0.0
            var loss_validacao = _loss_medio_lotes_validacao(bloco, lotes_validacao)
            if imprimir_cada_epoca > 0 and (epoca_atual % imprimir_cada_epoca == 0):
                print("Época", epoca_atual, "| Loss treino médio:", loss_medio_epoca, "| Loss validação:", loss_validacao)

            epoca_atual = item.epoca
            soma_loss_epoca = 0.0
            quantidade_lotes_epoca = 0

        var entradas = item.lote.entradas.copy()
        var alvos = item.lote.alvos.copy()

        var ctx = autograd.construir_contexto_mlp(entradas, alvos, bloco.pesos, bloco.biases, bloco.ativacao_saida_id, bloco.perda_id)
        var grads = dispatcher_gradiente.calcular_gradientes_mlp(ctx, bloco.pesos, manter_gradientes_na_ram_principal)
        loss_lote_final = grads.loss

        for camada in range(len(bloco.pesos)):
            for i in range(len(bloco.pesos[camada].dados)):
                bloco.pesos[camada].dados[i] = bloco.pesos[camada].dados[i] - taxa_aprendizado * grads.grad_ws[camada].dados[i]
            for j in range(len(bloco.biases[camada].dados)):
                bloco.biases[camada].dados[j] = bloco.biases[camada].dados[j] - taxa_aprendizado * grads.grad_bs[camada].dados[j]

        soma_loss_epoca = soma_loss_epoca + grads.loss
        quantidade_lotes_epoca = quantidade_lotes_epoca + 1

    var loss_medio_epoca_final = soma_loss_epoca / Float32(quantidade_lotes_epoca) if quantidade_lotes_epoca > 0 else 0.0
    var loss_validacao_final = _loss_medio_lotes_validacao(bloco, lotes_validacao)
    if imprimir_cada_epoca > 0 and (epoca_atual % imprimir_cada_epoca == 0):
        print("Época", epoca_atual, "| Loss treino médio:", loss_medio_epoca_final, "| Loss validação:", loss_validacao_final)

    return loss_lote_final


fn treinar(
    mut bloco: BlocoMLP,
    entradas: tensor_defs.Tensor,
    alvos: tensor_defs.Tensor,
    var taxa_aprendizado: Float32 = 0.03,
    var epocas: Int = 1200,
    var imprimir_cada: Int = 200,
    var manter_gradientes_na_ram_principal: Bool = False,
) -> Float32:
    debug_assert(len(entradas.formato) == 2, "entradas deve ser tensor 2D")
    debug_assert(len(alvos.formato) == 2 and alvos.formato[1] == 1, "alvos deve ser tensor 2D [N,1]")
    debug_assert(entradas.formato[0] == alvos.formato[0], "entradas/alvos com número de linhas diferente")
    debug_assert(entradas.formato[1] == bloco.topologia[0], "número de features incompatível")

    var loss_final: Float32 = 0.0
    for epoca in range(epocas):
        var ctx = autograd.construir_contexto_mlp(entradas, alvos, bloco.pesos, bloco.biases, bloco.ativacao_saida_id, bloco.perda_id)
        var grads = dispatcher_gradiente.calcular_gradientes_mlp(ctx, bloco.pesos, manter_gradientes_na_ram_principal)
        loss_final = grads.loss

        for camada in range(len(bloco.pesos)):
            for i in range(len(bloco.pesos[camada].dados)):
                bloco.pesos[camada].dados[i] = bloco.pesos[camada].dados[i] - taxa_aprendizado * grads.grad_ws[camada].dados[i]
            for j in range(len(bloco.biases[camada].dados)):
                bloco.biases[camada].dados[j] = bloco.biases[camada].dados[j] - taxa_aprendizado * grads.grad_bs[camada].dados[j]

        if imprimir_cada > 0 and (epoca % imprimir_cada == 0 or epoca == epocas - 1):
            print("Época", epoca, "| Loss:", loss_final)
            if epoca == 0:
                print("Grafo de computação (forward):")
                for op in ctx.operacoes:
                    print("  -", op)
                print("Nós do grafo:")
                for no in ctx.grafo.nos:
                    print("  *", no)
                print("Arestas do grafo:")
                for ar in ctx.grafo.arestas:
                    print("  ->", ar)

    return loss_final
