import src.autograd as autograd
import src.computacao.dispatcher_gradiente as dispatcher_gradiente
import src.nucleo.Tensor as tensor_defs

struct BlocoMLP(Movable, Copyable):
    var w1: tensor_defs.Tensor
    var b1: tensor_defs.Tensor
    var w2: tensor_defs.Tensor
    var b2: tensor_defs.Tensor
    var tipo_computacao: String

    fn __init__(out self, var num_entradas: Int, var num_ocultas: Int = 16, var tipo_computacao_in: String = "cpu"):
        self.tipo_computacao = tipo_computacao_in^

        var formato_w1 = List[Int]()
        formato_w1.append(num_entradas)
        formato_w1.append(num_ocultas)
        self.w1 = tensor_defs.Tensor(formato_w1^, self.tipo_computacao)

        var formato_b1 = List[Int]()
        formato_b1.append(1)
        formato_b1.append(num_ocultas)
        self.b1 = tensor_defs.Tensor(formato_b1^, self.tipo_computacao)

        var formato_w2 = List[Int]()
        formato_w2.append(num_ocultas)
        formato_w2.append(1)
        self.w2 = tensor_defs.Tensor(formato_w2^, self.tipo_computacao)

        var formato_b2 = List[Int]()
        formato_b2.append(1)
        formato_b2.append(1)
        self.b2 = tensor_defs.Tensor(formato_b2^, self.tipo_computacao)

        var escala: Float32 = 0.05
        for i in range(len(self.w1.dados)):
            var sinal: Float32 = 1.0 if (i % 2 == 0) else Float32(-1.0)
            self.w1.dados[i] = Float32(i + 1) * escala * sinal
        for i in range(len(self.b1.dados)):
            self.b1.dados[i] = 0.0
        for i in range(len(self.w2.dados)):
            var sinal2: Float32 = 1.0 if (i % 2 == 0) else Float32(-1.0)
            self.w2.dados[i] = Float32(i + 1) * escala * sinal2
        self.b2.dados[0] = 0.0

    fn copy(self) -> BlocoMLP:
        var copia = BlocoMLP(self.w1.formato[0], self.w1.formato[1], self.tipo_computacao)
        copia.w1 = self.w1.copy()
        copia.b1 = self.b1.copy()
        copia.w2 = self.w2.copy()
        copia.b2 = self.b2.copy()
        return copia^


fn prever(bloco: BlocoMLP, entradas: tensor_defs.Tensor) -> tensor_defs.Tensor:
    debug_assert(len(entradas.formato) == 2, "entradas deve ser tensor 2D")
    var formato_y = List[Int]()
    formato_y.append(entradas.formato[0])
    formato_y.append(1)
    var alvos_dummy = tensor_defs.Tensor(formato_y^, entradas.tipo_computacao)
    var ctx = autograd.construir_contexto_mlp(entradas, alvos_dummy, bloco.w1, bloco.b1, bloco.w2, bloco.b2)
    return ctx.pred


fn inferir(bloco: BlocoMLP, entradas: tensor_defs.Tensor) -> tensor_defs.Tensor:
    return prever(bloco, entradas)


fn treinar(
    mut bloco: BlocoMLP,
    entradas: tensor_defs.Tensor,
    alvos: tensor_defs.Tensor,
    var taxa_aprendizado: Float32 = 0.03,
    var epocas: Int = 1200,
    var imprimir_cada: Int = 200,
    var manter_gradientes_na_ram_principal: Bool = True,
) -> Float32:
    debug_assert(len(entradas.formato) == 2, "entradas deve ser tensor 2D")
    debug_assert(len(alvos.formato) == 2 and alvos.formato[1] == 1, "alvos deve ser tensor 2D [N,1]")
    debug_assert(entradas.formato[0] == alvos.formato[0], "entradas/alvos com número de linhas diferente")
    debug_assert(entradas.formato[1] == bloco.w1.formato[0], "número de features incompatível")

    var loss_final: Float32 = 0.0
    for epoca in range(epocas):
        var ctx = autograd.construir_contexto_mlp(entradas, alvos, bloco.w1, bloco.b1, bloco.w2, bloco.b2)
        var grads = dispatcher_gradiente.calcular_gradientes_mlp(ctx, bloco.w2, manter_gradientes_na_ram_principal)
        loss_final = grads.loss

        for i in range(len(bloco.w1.dados)):
            bloco.w1.dados[i] = bloco.w1.dados[i] - taxa_aprendizado * grads.grad_w1.dados[i]
        for i in range(len(bloco.b1.dados)):
            bloco.b1.dados[i] = bloco.b1.dados[i] - taxa_aprendizado * grads.grad_b1.dados[i]
        for i in range(len(bloco.w2.dados)):
            bloco.w2.dados[i] = bloco.w2.dados[i] - taxa_aprendizado * grads.grad_w2.dados[i]
        bloco.b2.dados[0] = bloco.b2.dados[0] - taxa_aprendizado * grads.grad_b2

        if imprimir_cada > 0 and (epoca % imprimir_cada == 0 or epoca == epocas - 1):
            print("Época", epoca, "| Loss MSE:", loss_final)
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
