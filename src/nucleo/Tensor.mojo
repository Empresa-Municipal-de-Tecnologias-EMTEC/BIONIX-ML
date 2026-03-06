# Núcleo - Tensor separado para suportar múltiplos backends
# Contém a definição de `Tensor` e auxiliares de formato/passos.
import src.computacao.tipos as backend_tipos

struct Tensor(Movable, Copyable):
    var dados: List[Float32]
    var formato: List[Int]
    var passos: List[Int]
    var tipo_computacao: String
    var id_backend: Int
    var gradiente: List[Float32]
    var tem_gradiente: Bool

    fn __init__(out self, var formato: List[Int], var tipo_computacao: String = "cpu"):
        debug_assert(backend_tipos.backend_nome_valido(tipo_computacao), "tipo de backend inválido")
        self.formato = formato^
        self.passos = calcular_passos(self.formato)
        self.tipo_computacao = backend_tipos.backend_nome_normalizado(tipo_computacao)
        self.id_backend = backend_tipos.backend_id_de_nome(self.tipo_computacao)
        var total: Int = 1
        for i in range(len(self.formato)):
            total = total * self.formato[i]
        self.dados = List[Float32](capacity=total)
        self.gradiente = List[Float32](capacity=total)
        for _ in range(total):
            self.dados.append(0.0)
            self.gradiente.append(0.0)
        self.tem_gradiente = False

    fn copy(self) -> Tensor:
        var novo_formato = self.formato.copy()
        var novo_tensor = Tensor(novo_formato^, self.tipo_computacao)
        for i in range(len(self.dados)):
            novo_tensor.dados[i] = self.dados[i]
            novo_tensor.gradiente[i] = self.gradiente[i]
        novo_tensor.tem_gradiente = self.tem_gradiente
        return novo_tensor^

#É possível calcular os passos a partir do formato, o que é útil para indexação eficiente.
#Se ainda não entendeu o que são passos, pense neles como o número de elementos que você precisa pular para ir para a próxima posição em cada dimensão.
#Essa estrutura de tensor controla os dados e o formato, mas os passos são calculados dinamicamente para garantir que estejam sempre corretos, mesmo se o formato for alterado.
#Dessa forma, caso seja necessário trabalhar com múltiplos backends (Plataforma de hardware), a estrutura do tensor permanece consistente.
#É possivel inclusive criar parâmetros para controlar paralelismo de tensores (Dividir o tensor em blocos localizados na meória de diferentes dispositivos), como blocos de processamento, mas isso fica para uma implementação futura.
fn calcular_passos(formato: List[Int]) -> List[Int]:
    var passos = List[Int](len(formato))
    var acumulador: Int = 1
    for i in reversed(range(len(formato))):
        passos[i] = acumulador
        acumulador = acumulador * formato[i]
    return passos^


fn preenchido_como(t: Tensor, v: Float32) -> Tensor:
    var copia_formato = t.formato.copy()
    var saida = Tensor(copia_formato^, t.tipo_computacao)
    for i in range(len(saida.dados)):
        saida.dados[i] = v
    return saida^


fn zerar_gradiente(mut t: Tensor):
    for i in range(len(t.gradiente)):
        t.gradiente[i] = 0.0
    t.tem_gradiente = False


fn acumular_gradiente(mut t: Tensor, g: Tensor):
    debug_assert(len(t.gradiente) == len(g.dados), "gradiente incompatível")
    for i in range(len(t.gradiente)):
        t.gradiente[i] = t.gradiente[i] + g.dados[i]
    t.tem_gradiente = True


fn somar_elementwise(a: Tensor, b: Tensor) -> Tensor:
    debug_assert(len(a.dados) == len(b.dados), "tensores devem ter mesmo tamanho")
    var formato = a.formato.copy()
    var saida = Tensor(formato^, a.tipo_computacao)
    for i in range(len(a.dados)):
        saida.dados[i] = a.dados[i] + b.dados[i]
    return saida^


fn subtrair_elementwise(a: Tensor, b: Tensor) -> Tensor:
    debug_assert(len(a.dados) == len(b.dados), "tensores devem ter mesmo tamanho")
    var formato = a.formato.copy()
    var saida = Tensor(formato^, a.tipo_computacao)
    for i in range(len(a.dados)):
        saida.dados[i] = a.dados[i] - b.dados[i]
    return saida^


fn multiplicar_elementwise(a: Tensor, b: Tensor) -> Tensor:
    debug_assert(len(a.dados) == len(b.dados), "tensores devem ter mesmo tamanho")
    var formato = a.formato.copy()
    var saida = Tensor(formato^, a.tipo_computacao)
    for i in range(len(a.dados)):
        saida.dados[i] = a.dados[i] * b.dados[i]
    return saida^


fn transpor(a: Tensor) -> Tensor:
    debug_assert(len(a.formato) == 2, "transpor requer tensor 2D")
    var linhas = a.formato[0]
    var colunas = a.formato[1]
    var formato = List[Int]()
    formato.append(colunas)
    formato.append(linhas)
    var out = Tensor(formato^, a.tipo_computacao)
    for i in range(linhas):
        for j in range(colunas):
            out.dados[j * linhas + i] = a.dados[i * colunas + j]
    return out^


fn multiplicar_matrizes(a: Tensor, b: Tensor) -> Tensor:
    debug_assert(len(a.formato) == 2 and len(b.formato) == 2, "matmul requer tensores 2D")
    var m = a.formato[0]
    var n = a.formato[1]
    debug_assert(n == b.formato[0], "dimensões incompatíveis para matmul")
    var p = b.formato[1]
    var formato = List[Int]()
    formato.append(m)
    formato.append(p)
    var out = Tensor(formato^, a.tipo_computacao)
    for i in range(m):
        for j in range(p):
            var acc: Float32 = 0.0
            for k in range(n):
                acc = acc + a.dados[i * n + k] * b.dados[k * p + j]
            out.dados[i * p + j] = acc
    return out^


fn adicionar_bias_coluna(a: Tensor, b: Tensor) -> Tensor:
    debug_assert(len(a.formato) == 2, "entrada deve ser 2D")
    debug_assert(len(b.formato) == 2 and b.formato[0] == 1 and b.formato[1] == 1, "bias deve ter formato [1,1]")
    var formato = a.formato.copy()
    var out = Tensor(formato^, a.tipo_computacao)
    var valor_bias = b.dados[0]
    for i in range(len(a.dados)):
        out.dados[i] = a.dados[i] + valor_bias
    return out^


fn soma_total(a: Tensor) -> Float32:
    var s: Float32 = 0.0
    for i in range(len(a.dados)):
        s = s + a.dados[i]
    return s


fn erro_quadratico_medio_escalar(pred: Tensor, alvo: Tensor) -> Float32:
    debug_assert(len(pred.dados) == len(alvo.dados), "pred e alvo devem ter mesmo tamanho")
    if len(pred.dados) == 0:
        return 0.0
    var soma: Float32 = 0.0
    for i in range(len(pred.dados)):
        var d = pred.dados[i] - alvo.dados[i]
        soma = soma + d * d
    return soma / Float32(len(pred.dados))


fn gradiente_mse(pred: Tensor, alvo: Tensor) -> Tensor:
    debug_assert(len(pred.dados) == len(alvo.dados), "pred e alvo devem ter mesmo tamanho")
    var formato = pred.formato.copy()
    var out = Tensor(formato^, pred.tipo_computacao)
    if len(pred.dados) == 0:
        return out^
    var n = Float32(len(pred.dados))
    for i in range(len(pred.dados)):
        out.dados[i] = 2.0 * (pred.dados[i] - alvo.dados[i]) / n
    return out^
