struct GrafoComputacao(Movable, Copyable):
    var nos: List[String]
    var arestas: List[String]

    fn __init__(out self, var nos_in: List[String], var arestas_in: List[String]):
        self.nos = nos_in^
        self.arestas = arestas_in^


fn criar_grafo_mlp_forward() -> GrafoComputacao:
    var nos = List[String]()
    nos.append("x")
    nos.append("w1")
    nos.append("b1")
    nos.append("z1")
    nos.append("a1")
    nos.append("w2")
    nos.append("b2")
    nos.append("z2")
    nos.append("pred")

    var arestas = List[String]()
    arestas.append("x -> matmul(x,w1)")
    arestas.append("w1 -> matmul(x,w1)")
    arestas.append("matmul(x,w1) -> add_bias_hidden")
    arestas.append("b1 -> add_bias_hidden")
    arestas.append("add_bias_hidden -> relu")
    arestas.append("relu -> matmul(a1,w2)")
    arestas.append("w2 -> matmul(a1,w2)")
    arestas.append("matmul(a1,w2) -> add_bias_out")
    arestas.append("b2 -> add_bias_out")
    arestas.append("add_bias_out -> hard_sigmoid")
    arestas.append("hard_sigmoid -> pred")

    return GrafoComputacao(nos^, arestas^)
