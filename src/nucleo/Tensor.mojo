# Núcleo - Tensor separado para suportar múltiplos backends
# Contém a definição de `Tensor` e auxiliares de formato/passos.
struct Tensor(Movable, Copyable):
    var dados: List[Float32]
    var formato: List[Int]
    var passos: List[Int]

    fn __init__(out self, var formato: List[Int]):
        self.formato = formato^
        self.passos = calcular_passos(self.formato)
        var total: Int = 1
        for i in range(len(self.formato)):
            total = total * self.formato[i]
        self.dados = List[Float32](capacity=total)
        for _ in range(total):
            self.dados.append(0.0)

    fn copy(self) -> Tensor:
        var novo_formato = self.formato.copy()
        var novo_tensor = Tensor(novo_formato^)
        for i in range(len(self.dados)):
            novo_tensor.dados[i] = self.dados[i]
        return novo_tensor^


fn calcular_passos(formato: List[Int]) -> List[Int]:
    var passos = List[Int](len(formato))
    var acumulador: Int = 1
    for i in reversed(range(len(formato))):
        passos[i] = acumulador
        acumulador = acumulador * formato[i]
    return passos^


fn preenchido_como(t: Tensor, v: Float32) -> Tensor:
    var copia_formato = t.formato.copy()
    var saida = Tensor(copia_formato^)
    for i in range(len(saida.dados)):
        saida.dados[i] = v
    return saida^
