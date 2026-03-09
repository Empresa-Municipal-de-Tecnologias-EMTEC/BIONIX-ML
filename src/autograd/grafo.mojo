struct GrafoComputacao(Movable, Copyable):
    var nos: List[String]
    var arestas: List[String]

    fn __init__(out self, var nos_in: List[String], var arestas_in: List[String]):
        self.nos = nos_in^
        self.arestas = arestas_in^


fn criar_grafo_mlp_forward_topologia(topologia: List[Int]) -> GrafoComputacao:
    debug_assert(len(topologia) >= 2, "topologia deve ter ao menos entrada e saída")

    var nos = List[String]()
    nos.append("a0")

    var num_camadas = len(topologia) - 1
    for camada in range(num_camadas):
        nos.append("w" + String(camada + 1))
        nos.append("b" + String(camada + 1))
        nos.append("z" + String(camada + 1))
        nos.append("a" + String(camada + 1))

    nos.append("pred")

    var arestas = List[String]()
    for camada in range(num_camadas):
        var atual = camada + 1
        arestas.append("a" + String(camada) + " -> matmul(a" + String(camada) + ",w" + String(atual) + ")")
        arestas.append("w" + String(atual) + " -> matmul(a" + String(camada) + ",w" + String(atual) + ")")
        arestas.append("matmul(a" + String(camada) + ",w" + String(atual) + ") -> add_bias_" + String(atual))
        arestas.append("b" + String(atual) + " -> add_bias_" + String(atual))

        if camada < num_camadas - 1:
            arestas.append("add_bias_" + String(atual) + " -> relu_" + String(atual))
            arestas.append("relu_" + String(atual) + " -> a" + String(atual))
        else:
            arestas.append("add_bias_" + String(atual) + " -> hard_sigmoid_out")
            arestas.append("hard_sigmoid_out -> a" + String(atual))

    arestas.append("a" + String(num_camadas) + " -> pred")

    return GrafoComputacao(nos^, arestas^)


fn criar_grafo_mlp_forward() -> GrafoComputacao:
    var topologia_padrao = List[Int]()
    topologia_padrao.append(2)
    topologia_padrao.append(16)
    topologia_padrao.append(16)
    topologia_padrao.append(1)

    return criar_grafo_mlp_forward_topologia(topologia_padrao^)
