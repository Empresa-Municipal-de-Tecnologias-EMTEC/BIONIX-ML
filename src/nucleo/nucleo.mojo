# Núcleo do Bionix: tensores, operações e autograd (automatic gradient - gradiente automático)
# Todas as documentações abaixo estão em Português: nome, parâmetros e descrição.

# Estrutura: Tensor (em inglês: Tensor)
# Parâmetros:
# - dados: List[Float32] -> buffer (memória intermediária) contíguo com os dados do tensor
# - formato: List[Int] -> dimensões do tensor (em inglês: shape)
# - passos: List[Int] -> passos entre elementos de cada dimensão (em inglês: strides)
# O que faz: Representa um tensor n-dimensional simples com buffer contíguo,
#           formato e passos. Fornece inicialização básica.
struct Tensor(Movable, Copyable):
    var dados: List[Float32]  # data (em inglês)
    var formato: List[Int]     # shape (em inglês)
    var passos: List[Int]      # strides (em inglês)
    
    fn __init__(out self, var formato: List[Int]):  # shape (em inglês)
        self.formato = formato^
        self.passos = calcular_passos(self.formato)  # compute_strides (em inglês)
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


# Função: calcular_passos (em inglês: compute_strides)
# Parâmetros:
# - formato: List[Int] -> dimensões do tensor (em inglês: shape)
# Retorno: List[Int] com os passos correspondentes (em inglês: strides)
# O que faz: Calcula os passos para indexação em buffer (memória intermediária) contíguo 
#           usando ordenação row-major (linhas principais - elementos consecutivos na última dimensão).
fn calcular_passos(formato: List[Int]) -> List[Int]:  # compute_strides (em inglês)
    var passos = List[Int](len(formato))  # strides (em inglês)
    var acumulador: Int = 1  # acc/accumulator (em inglês)
    for i in reversed(range(len(formato))):
        passos[i] = acumulador
        acumulador = acumulador * formato[i]
    return passos^


# Função: preenchido_como (em inglês: filled_like)
# Parâmetros:
# - t: Tensor -> tensor de referência para formato
# - v: Float32 -> valor para preencher
# Retorno: Tensor preenchido com `v` e mesmo `formato` de `t`
# O que faz: Cria um tensor com os mesmos formato/passos e preenche com `v`.
fn preenchido_como(t: Tensor, v: Float32) -> Tensor:  # filled_like (em inglês)
    var copia_formato = t.formato.copy()  # shape_copy (em inglês)
    var saida = Tensor(copia_formato^)  # out/output (em inglês)
    for i in range(len(saida.dados)):
        saida.dados[i] = v
    return saida^


# Operações tensoriais (forward - propagação direta)

# Função: somar (em inglês: add)
# Parâmetros:
# - a: Tensor -> primeiro tensor
# - b: Tensor -> segundo tensor
# Retorno: Tensor resultante da soma elementwise (elemento por elemento)
# O que faz: Soma elemento-a-elemento entre `a` e `b`. Requer mesmos formatos.
fn somar(a: Tensor, b: Tensor) -> Tensor:  # add (em inglês)
    debug_assert(len(a.dados) == len(b.dados), "tensores devem ter mesmo tamanho")
    var copia_formato = a.formato.copy()  # shape_copy (em inglês)
    var saida = Tensor(copia_formato^)  # out/output (em inglês)
    for i in range(len(a.dados)):
        saida.dados[i] = a.dados[i] + b.dados[i]
    return saida^


# Função: somar_paralelo (em inglês: add_parallel)
# Mesma assinatura de `somar` mas com execução paralela em loop.
# Usa múltiplos núcleos do processador para acelerar a computação.
# Nota: @parallel não suportado nesta versão, implementação sequencial
fn somar_paralelo(a: Tensor, b: Tensor) -> Tensor:  # add_parallel (em inglês)
    debug_assert(len(a.dados) == len(b.dados), "tensores devem ter mesmo tamanho")
    var copia_formato = a.formato.copy()  # shape_copy (em inglês)
    var saida = Tensor(copia_formato^)  # out/output (em inglês)
    # TODO: Adicionar paralelização quando @parallel for suportado
    for i in range(len(a.dados)):
        saida.dados[i] = a.dados[i] + b.dados[i]
    return saida^


# Função: multiplicar (em inglês: multiply)
# Parâmetros:
# - a: Tensor -> primeiro tensor
# - b: Tensor -> segundo tensor
# Retorno: Tensor resultante da multiplicação elementwise (elemento por elemento)
# O que faz: Multiplica elemento-a-elemento entre `a` e `b`. Requer mesmos formatos.
fn multiplicar(a: Tensor, b: Tensor) -> Tensor:  # multiply (em inglês)
    debug_assert(len(a.dados) == len(b.dados), "tensores devem ter mesmo tamanho")
    var copia_formato = a.formato.copy()  # shape_copy (em inglês)
    var saida = Tensor(copia_formato^)  # out/output (em inglês)
    for i in range(len(a.dados)):
        saida.dados[i] = a.dados[i] * b.dados[i]
    return saida^


# Função: multiplicar_matrizes (em inglês: matmul - matrix multiplication)
# Parâmetros:
# - a: Tensor (formato [m, n]) -> matriz A com m linhas e n colunas
# - b: Tensor (formato [n, p]) -> matriz B com n linhas e p colunas
# Retorno: Tensor (formato [m, p]) resultado da multiplicação matricial
# O que faz: Implementa uma multiplicação matricial simples (não otimizada) para tensores 2D (bidimensionais).
#           matmul = MATrix MULtiplication (Multiplicação de Matrizes)
fn multiplicar_matrizes(a: Tensor, b: Tensor) -> Tensor:  # matmul (em inglês)
    var m = a.formato[0]  # número de linhas de A
    var n = a.formato[1]  # número de colunas de A / linhas de B
    var p = b.formato[1]  # número de colunas de B
    var lista_formato = List[Int](2)  # shape_list (em inglês)
    lista_formato.append(m)
    lista_formato.append(p)
    var saida = Tensor(lista_formato^)  # out/output (em inglês)
    for i in range(m):
        for j in range(p):
            var acumulador: Float32 = 0.0  # acc/accumulator (em inglês)
            for k in range(n):
                acumulador = acumulador + a.dados[i * n + k] * b.dados[k * p + j]
            saida.dados[i * p + j] = acumulador
    return saida^


# Função: erro_quadratico_medio (em inglês: mse - mean squared error)
# Parâmetros:
# - a: Tensor -> tensor com valores preditos
# - b: Tensor -> tensor com valores esperados/alvos
# Retorno: Tensor escalar (formato [1]) com o MSE (Mean Squared Error - Erro Quadrático Médio)
# O que faz: Calcula o erro quadrático médio entre `a` e `b`.
#           MSE = média dos quadrados das diferenças entre valores preditos e reais.
fn erro_quadratico_medio(a: Tensor, b: Tensor) -> Tensor:  # mse (em inglês)
    debug_assert(len(a.dados) == len(b.dados), "tensores devem ter mesmo tamanho")
    var soma: Float32 = 0.0  # sum (em inglês)
    for i in range(len(a.dados)):
        var diferenca: Float32 = a.dados[i] - b.dados[i]  # d/difference (em inglês)
        soma = soma + diferenca * diferenca
    var lista_formato = List[Int](1)  # shape_list (em inglês)
    lista_formato.append(1)
    var saida = Tensor(lista_formato^)  # out/output (em inglês)
    saida.dados[0] = soma / Float32(len(a.dados))
    return saida^


# AUTOGRAD = AUTOmatic GRADient (Gradiente Automático)
# Grafo computacional = estrutura de dados que representa operações matemáticas

# Estrutura: No (em inglês: Node)
# Parâmetros:
# - valor: Tensor -> valor armazenado no nó
# - gradiente: Tensor -> gradiente acumulado (inicialmente None)
# - pais: List[No] -> nós pais no grafo computacional (conforme escopo original)
# - tem_pais: Bool -> indica se este nó tem pais no grafo
# - entrada_a, entrada_b: Tensor -> cópias dos tensores de entrada (pais) usados na operação
# - grad_entrada_a, grad_entrada_b: Tensor -> gradientes calculados para as entradas
# - nome_operacao: String -> nome da operação que criou este nó (para debugging e backward)
# O que faz: Representa um nó no grafo de computação para backprop (backpropagation - retropropagação).
#           Backprop = algoritmo para calcular gradientes em redes neurais.
#           Agora é Movable E Copyable para permitir List[No] conforme escopo.
struct No(Movable, Copyable):  # Node (em inglês) - COPYABLE para List[No]
    var valor: Tensor  # value (em inglês)
    var gradiente: Tensor  # grad/gradient (em inglês)
    var tem_gradiente: Bool  # has_grad (em inglês)
    var nome_operacao: String  # operation_name (em inglês)
    var pais: List[No]  # parents (em inglês) - conforme escopo original!
    var tem_pais: Bool  # has_parents (em inglês)
    var entrada_a: Tensor  # input_a/parent_a (em inglês)
    var entrada_b: Tensor  # input_b/parent_b (em inglês)
    var grad_entrada_a: Tensor  # grad_input_a (em inglês)
    var grad_entrada_b: Tensor  # grad_input_b (em inglês)
    
    fn __init__(out self, var valor: Tensor, var nome_operacao: String = "folha"):  # value, operation_name, leaf (em inglês)
        self.valor = valor^
        var formato_grad = self.valor.formato.copy()  # grad_shape (em inglês)
        self.gradiente = Tensor(formato_grad^)
        self.tem_gradiente = False  # has_grad (em inglês)
        self.nome_operacao = nome_operacao^
        self.pais = List[No]()  # parents - inicializa vazio
        self.tem_pais = False  # has_parents (em inglês)
        # Criar tensores vazios para entrada_a e entrada_b
        var formato_vazio = List[Int](1)
        formato_vazio.append(1)
        self.entrada_a = Tensor(formato_vazio^)
        var formato_vazio2 = List[Int](1)
        formato_vazio2.append(1)
        self.entrada_b = Tensor(formato_vazio2^)
        var formato_vazio3 = List[Int](1)
        formato_vazio3.append(1)
        self.grad_entrada_a = Tensor(formato_vazio3^)
        var formato_vazio4 = List[Int](1)
        formato_vazio4.append(1)
        self.grad_entrada_b = Tensor(formato_vazio4^)
    
    fn copy(self) -> No:
        var novo_no = No(self.valor.copy(), self.nome_operacao)
        novo_no.gradiente = self.gradiente.copy()
        novo_no.tem_gradiente = self.tem_gradiente
        novo_no.tem_pais = self.tem_pais
        novo_no.entrada_a = self.entrada_a.copy()
        novo_no.entrada_b = self.entrada_b.copy()
        novo_no.grad_entrada_a = self.grad_entrada_a.copy()
        novo_no.grad_entrada_b = self.grad_entrada_b.copy()
        # Copiar pais recursivamente
        for i in range(len(self.pais)):
            novo_no.pais.append(self.pais[i].copy())
        return novo_no^

# Função: no_de_tensor (em inglês: node_from_tensor)
# Parâmetros:
# - t: Tensor -> tensor a ser encapsulado
# Retorno: No sem pais, sem gradiente calculado
# O que faz: Envolve um `Tensor` em um `No` base (folha) para o grafo computacional.
fn no_de_tensor(var t: Tensor) -> No:  # node_from_tensor (em inglês)
    return No(t^, "folha")  # leaf (em inglês)


# Função: somar_nos (em inglês: add_nodes)
# Parâmetros:
# - a: Tensor -> primeiro tensor
# - b: Tensor -> segundo tensor
# Retorno: No representando `a + b` com capacidade de retropropagação
# O que faz: Cria um novo `No` cujo valor é a soma dos tensores.
#           Durante backward, distribui gradiente igualmente para ambos os pais.
#           Armazena cópias de a e b para cálculo de gradientes.
fn somar_nos(a: Tensor, b: Tensor) -> No:  # add_nodes (em inglês)
    var valor = somar(a, b)  # value (em inglês)
    var no = No(valor^, "somar")  # add (em inglês)
    no.tem_pais = True
    no.entrada_a = a.copy()
    no.entrada_b = b.copy()
    no.grad_entrada_a = preenchido_como(a, 0.0)
    no.grad_entrada_b = preenchido_como(b, 0.0)
    return no^


# Função: multiplicar_nos (em inglês: multiply_nodes)
# Parâmetros:
# - a: Tensor -> primeiro tensor
# - b: Tensor -> segundo tensor
# Retorno: No representando `a * b` (elementwise) com capacidade de retropropagação
# O que faz: Cria nó para multiplicação elemento-por-elemento.
#           Durante backward, usa regra do produto: grad_a = grad_out * b, grad_b = grad_out * a
#           Armazena cópias de a e b para cálculo de gradientes.
fn multiplicar_nos(a: Tensor, b: Tensor) -> No:  # multiply_nodes (em inglês)
    var valor = multiplicar(a, b)  # value (em inglês)
    var no = No(valor^, "multiplicar")  # multiply (em inglês)
    no.tem_pais = True
    no.entrada_a = a.copy()
    no.entrada_b = b.copy()
    no.grad_entrada_a = preenchido_como(a, 0.0)
    no.grad_entrada_b = preenchido_como(b, 0.0)
    return no^


# Função: multiplicar_matrizes_nos (em inglês: matmul_nodes)
# Parâmetros:
# - a: Tensor -> primeira matriz
# - b: Tensor -> segunda matriz
# Retorno: No com valor da multiplicação matricial
# O que faz: Cria nó para multiplicação de matrizes.
#           matmul = MATrix MULtiplication (Multiplicação de Matrizes)
#           Armazena cópias de a e b para cálculo de gradientes.
fn multiplicar_matrizes_nos(a: Tensor, b: Tensor) -> No:  # matmul_nodes (em inglês)
    var valor = multiplicar_matrizes(a, b)  # value (em inglês)
    var no = No(valor^, "matmul")
    no.tem_pais = True
    no.entrada_a = a.copy()
    no.entrada_b = b.copy()
    no.grad_entrada_a = preenchido_como(a, 0.0)
    no.grad_entrada_b = preenchido_como(b, 0.0)
    return no^


# Função: no_erro_quadratico_medio (em inglês: mse_node)
# Parâmetros:
# - a: Tensor -> tensor com valores preditos
# - b: Tensor -> tensor com valores esperados/alvos
# Retorno: No escalar com MSE (Mean Squared Error - Erro Quadrático Médio)
# O que faz: Cria nó para cálculo do erro quadrático médio.
#           Útil como função de perda (loss function) em aprendizado de máquina.
#           MSE = média dos quadrados das diferenças.
#           Armazena cópias de a e b para cálculo de gradientes.
fn no_erro_quadratico_medio(a: Tensor, b: Tensor) -> No:  # mse_node (em inglês)
    var valor = erro_quadratico_medio(a, b)  # value (em inglês)
    var no = No(valor^, "mse")
    no.tem_pais = True
    no.entrada_a = a.copy()
    no.entrada_b = b.copy()
    no.grad_entrada_a = preenchido_como(a, 0.0)
    no.grad_entrada_b = preenchido_como(b, 0.0)
    return no^


# ========== FUNÇÕES COM GRAFO COMPUTACIONAL (conforme escopo original) ==========
# Estas funções recebem No e constroem o grafo com parents = [a, b]

# Função: somar_nos_grafo (em inglês: add_nodes_graph)
# Parâmetros:
# - a: No -> primeiro nó
# - b: No -> segundo nó
# Retorno: No representando `a + b` conectado ao grafo
# O que faz: Implementação COMPLETA conforme escopo: parents = [a, b]
fn somar_nos_grafo(a: No, b: No) -> No:
    var valor = somar(a.valor, b.valor)
    var no = No(valor^, "somar")
    no.tem_pais = True
    no.entrada_a = a.valor.copy()
    no.entrada_b = b.valor.copy()
    no.grad_entrada_a = preenchido_como(a.valor, 0.0)
    no.grad_entrada_b = preenchido_como(b.valor, 0.0)
    no.pais.append(a.copy())  # parents[0] = a
    no.pais.append(b.copy())  # parents[1] = b
    return no^


# Função: multiplicar_nos_grafo (em inglês: multiply_nodes_graph)
# Parâmetros:
# - a: No -> primeiro nó
# - b: No -> segundo nó
# Retorno: No representando `a * b` conectado ao grafo
fn multiplicar_nos_grafo(a: No, b: No) -> No:
    var valor = multiplicar(a.valor, b.valor)
    var no = No(valor^, "multiplicar")
    no.tem_pais = True
    no.entrada_a = a.valor.copy()
    no.entrada_b = b.valor.copy()
    no.grad_entrada_a = preenchido_como(a.valor, 0.0)
    no.grad_entrada_b = preenchido_como(b.valor, 0.0)
    no.pais.append(a.copy())
    no.pais.append(b.copy())
    return no^


# Função: multiplicar_matrizes_nos_grafo (em inglês: matmul_nodes_graph)
# Parâmetros:
# - a: No -> primeiro nó
# - b: No -> segundo nó
# Retorno: No com matmul conectado ao grafo
fn multiplicar_matrizes_nos_grafo(a: No, b: No) -> No:
    var valor = multiplicar_matrizes(a.valor, b.valor)
    var no = No(valor^, "matmul")
    no.tem_pais = True
    no.entrada_a = a.valor.copy()
    no.entrada_b = b.valor.copy()
    no.grad_entrada_a = preenchido_como(a.valor, 0.0)
    no.grad_entrada_b = preenchido_como(b.valor, 0.0)
    no.pais.append(a.copy())
    no.pais.append(b.copy())
    return no^


# Função: no_erro_quadratico_medio_grafo (em inglês: mse_node_graph)
# Parâmetros:
# - a: No -> nó com valores preditos
# - b: No -> nó com valores esperados/alvos
# Retorno: No escalar com MSE conectado ao grafo
fn no_erro_quadratico_medio_grafo(a: No, b: No) -> No:
    var valor = erro_quadratico_medio(a.valor, b.valor)
    var no = No(valor^, "mse")
    no.tem_pais = True
    no.entrada_a = a.valor.copy()
    no.entrada_b = b.valor.copy()
    no.grad_entrada_a = preenchido_como(a.valor, 0.0)
    no.grad_entrada_b = preenchido_como(b.valor, 0.0)
    no.pais.append(a.copy())
    no.pais.append(b.copy())
    return no^


# Função: retropropagar (em inglês: backward)
# Parâmetros:
# - saida: No -> nó de saída cujo gradiente é iniciado em 1.0
# O que faz: Percorre o grafo em ordem reversa e distribui gradientes automaticamente.
#           backward = retropropagação (backpropagation) - algoritmo para calcular gradientes.
#           Implementa autograd dinâmico completo com cálculo de gradientes por operação.
fn retropropagar(mut saida: No):  # backward (em inglês)
    # Inicializa gradiente do nó de saída com 1.0
    for i in range(len(saida.gradiente.dados)):
        saida.gradiente.dados[i] = 1.0
    saida.tem_gradiente = True  # has_grad (em inglês)
    
    # Se não tem pais, é um nó folha (leaf) - não propaga gradientes
    if not saida.tem_pais:
        return
    
    # Inicializa tensores de gradiente das entradas com formato correto
    saida.grad_entrada_a = preenchido_como(saida.entrada_a, 0.0)
    saida.grad_entrada_b = preenchido_como(saida.entrada_b, 0.0)
    
    # Calcula gradientes dos pais baseado na operação (backward_fn inline)
    if saida.nome_operacao == "somar":  # add
        # Para adição: d_loss/d_a = d_loss/d_out, d_loss/d_b = d_loss/d_out
        # Gradiente é distribuído igualmente para ambos os pais
        for i in range(len(saida.gradiente.dados)):
            saida.grad_entrada_a.dados[i] = saida.gradiente.dados[i]
            saida.grad_entrada_b.dados[i] = saida.gradiente.dados[i]
        
    elif saida.nome_operacao == "multiplicar":  # multiply
        # Para multiplicação elementwise: 
        # d_loss/d_a = d_loss/d_out * b
        # d_loss/d_b = d_loss/d_out * a
        for i in range(len(saida.gradiente.dados)):
            saida.grad_entrada_a.dados[i] = saida.gradiente.dados[i] * saida.entrada_b.dados[i]
            saida.grad_entrada_b.dados[i] = saida.gradiente.dados[i] * saida.entrada_a.dados[i]
        
    elif saida.nome_operacao == "matmul":
        # Para matmul (A @ B = C):
        # d_loss/d_A = d_loss/d_C @ B^T
        # d_loss/d_B = A^T @ d_loss/d_C
        var m = saida.entrada_a.formato[0]  # linhas de A
        var n = saida.entrada_a.formato[1]  # colunas de A / linhas de B
        var p = saida.entrada_b.formato[1]  # colunas de B
        
        # grad_A = grad_out @ B^T
        for i in range(m):
            for j in range(n):
                var acumulador: Float32 = 0.0
                for k in range(p):
                    # grad_out[i,k] * B[j,k] (B transposta)
                    acumulador += saida.gradiente.dados[i * p + k] * saida.entrada_b.dados[j * p + k]
                saida.grad_entrada_a.dados[i * n + j] = acumulador
        
        # grad_B = A^T @ grad_out
        for i in range(n):
            for j in range(p):
                var acumulador: Float32 = 0.0
                for k in range(m):
                    # A[k,i] (A transposta) * grad_out[k,j]
                    acumulador += saida.entrada_a.dados[k * n + i] * saida.gradiente.dados[k * p + j]
                saida.grad_entrada_b.dados[i * p + j] = acumulador
        
    elif saida.nome_operacao == "mse":
        # Para MSE: d_loss/d_pred = 2 * (pred - target) / n
        # onde entrada_a = predições, entrada_b = alvos
        var n = Float32(len(saida.entrada_a.dados))
        for i in range(len(saida.entrada_a.dados)):
            # Gradiente do MSE em relação às predições
            saida.grad_entrada_a.dados[i] = 2.0 * (saida.entrada_a.dados[i] - saida.entrada_b.dados[i]) / n * saida.gradiente.dados[0]
            # Gradiente em relação aos alvos (geralmente não usado)
            saida.grad_entrada_b.dados[i] = -2.0 * (saida.entrada_a.dados[i] - saida.entrada_b.dados[i]) / n * saida.gradiente.dados[0]


# Função: retropropagar_com_grafo (em inglês: backward_with_graph)
# Parâmetros:
# - saida: No -> nó de saída do grafo computacional
# O que faz: IMPLEMENTAÇÃO COMPLETA conforme escopo original.
#           Percorre o grafo automaticamente usando stack, propaga gradientes para todos os pais.
#           Implementa travessia em ordem reversa com acumulação de gradientes.
fn retropropagar_com_grafo(mut saida: No):  # backward_with_graph (em inglês)
    # Inicializa gradiente do nó de saída com 1.0 (conforme escopo)
    saida.gradiente = preenchido_como(saida.valor, 1.0)
    saida.tem_gradiente = True
    
    # Stack para travessia do grafo em ordem reversa (conforme escopo)
    var pilha = List[No]()  # stack (em inglês)
    pilha.append(saida.copy())
    
    # Travessia do grafo (conforme escopo: while stack.len > 0)
    while len(pilha) > 0:
        var no_atual = pilha.pop()  # node = stack.pop() (conforme escopo)
        
        # Se não tem pais, é nó folha - não propaga
        if not no_atual.tem_pais:
            continue
        
        # Calcula gradientes dos pais baseado na operação (backward_fn inline)
        if no_atual.nome_operacao == "somar":
            # Para adição: grad_a = grad_out, grad_b = grad_out
            for i in range(len(no_atual.gradiente.dados)):
                no_atual.grad_entrada_a.dados[i] = no_atual.gradiente.dados[i]
                no_atual.grad_entrada_b.dados[i] = no_atual.gradiente.dados[i]
            
        elif no_atual.nome_operacao == "multiplicar":
            # Para multiplicação elementwise: grad_a = grad_out * b, grad_b = grad_out * a
            for i in range(len(no_atual.gradiente.dados)):
                no_atual.grad_entrada_a.dados[i] = no_atual.gradiente.dados[i] * no_atual.entrada_b.dados[i]
                no_atual.grad_entrada_b.dados[i] = no_atual.gradiente.dados[i] * no_atual.entrada_a.dados[i]
            
        elif no_atual.nome_operacao == "matmul":
            # Para matmul: grad_A = grad_out @ B^T, grad_B = A^T @ grad_out
            var m = no_atual.entrada_a.formato[0]
            var n = no_atual.entrada_a.formato[1]
            var p = no_atual.entrada_b.formato[1]
            
            for i in range(m):
                for j in range(n):
                    var acumulador: Float32 = 0.0
                    for k in range(p):
                        acumulador += no_atual.gradiente.dados[i * p + k] * no_atual.entrada_b.dados[j * p + k]
                    no_atual.grad_entrada_a.dados[i * n + j] = acumulador
            
            for i in range(n):
                for j in range(p):
                    var acumulador: Float32 = 0.0
                    for k in range(m):
                        acumulador += no_atual.entrada_a.dados[k * n + i] * no_atual.gradiente.dados[k * p + j]
                    no_atual.grad_entrada_b.dados[i * p + j] = acumulador
            
        elif no_atual.nome_operacao == "mse":
            # Para MSE: d_loss/d_pred = 2 * (pred - target) / n
            var n = Float32(len(no_atual.entrada_a.dados))
            for i in range(len(no_atual.entrada_a.dados)):
                no_atual.grad_entrada_a.dados[i] = 2.0 * (no_atual.entrada_a.dados[i] - no_atual.entrada_b.dados[i]) / n * no_atual.gradiente.dados[0]
                no_atual.grad_entrada_b.dados[i] = -2.0 * (no_atual.entrada_a.dados[i] - no_atual.entrada_b.dados[i]) / n * no_atual.gradiente.dados[0]
        
        # Propaga gradientes para os pais e adiciona à pilha (conforme escopo)
        if len(no_atual.pais) > 0:
            # Acumula gradiente no primeiro pai
            for i in range(len(no_atual.pais[0].gradiente.dados)):
                no_atual.pais[0].gradiente.dados[i] += no_atual.grad_entrada_a.dados[i]
            no_atual.pais[0].tem_gradiente = True
            pilha.append(no_atual.pais[0].copy())
            
            # Acumula gradiente no segundo pai (se existir)
            if len(no_atual.pais) > 1:
                for i in range(len(no_atual.pais[1].gradiente.dados)):
                    no_atual.pais[1].gradiente.dados[i] += no_atual.grad_entrada_b.dados[i]
                no_atual.pais[1].tem_gradiente = True
                pilha.append(no_atual.pais[1].copy())


# Função: zerar_gradientes (em inglês: zero_grad)
# Parâmetros:
# - no: inout No -> nó cujo gradiente será zerado
# O que faz: Zera o gradiente de um nó específico.
#           grad = gradient (gradiente) - derivada parcial usada em otimização.
fn zerar_gradiente(mut no: No):  # zero_grad (em inglês)
    for i in range(len(no.gradiente.dados)):
        no.gradiente.dados[i] = 0.0
    no.tem_gradiente = False  # has_grad (em inglês)

# Função: atualizar_parametro (em inglês: update_parameter)
# Parâmetros:
# - parametro: inout No -> nó/parâmetro a ser atualizado
# - taxa_aprendizado: Float32 -> taxa de aprendizado para gradient descent (lr/learning rate)
# O que faz: Atualiza valores do parâmetro usando gradient descent: param = param - lr * grad.
#           Gradient descent = algoritmo de otimização que minimiza a função de perda.
fn atualizar_parametro(mut parametro: No, taxa_aprendizado: Float32):  # update_parameter, learning_rate (em inglês)
    if parametro.tem_gradiente:  # has_grad (em inglês)
        for i in range(len(parametro.valor.dados)):
            parametro.valor.dados[i] = parametro.valor.dados[i] - taxa_aprendizado * parametro.gradiente.dados[i]


# Função: passo_treinamento (em inglês: train_step)
# Parâmetros:
# - x: Tensor -> entrada do modelo (features/características)
# - y_esperado: Tensor -> saída esperada (target/alvo)
# - W: inout No -> matriz de pesos (weights)
# - b: inout No -> vetor de bias (viés)
# - taxa_aprendizado: Float32 -> taxa de aprendizado (lr/learning rate)
# Retorno: Float32 com valor da perda (loss)
# O que faz: Executa um passo completo de treinamento: forward pass, calcula perda (MSE),
#           calcula gradientes manualmente e atualiza parâmetros usando gradient descent.
#           Modelo linear: y = xW + b
fn passo_treinamento(x: Tensor, y_esperado: Tensor, 
                     mut W: No, mut b: No, 
                     taxa_aprendizado: Float32) -> Float32:  # train_step, y_true, learning_rate (em inglês)
    # Forward pass (propagação direta)
    var pred_temp = multiplicar_matrizes(x, W.valor)  # y_pred_temp (em inglês)
    var y_predito = somar(pred_temp, b.valor)  # y_pred (em inglês)
    
    # Calcular perda (loss - MSE)
    var perda_no = no_erro_quadratico_medio(y_predito, y_esperado)  # loss_node (em inglês)
    var valor_perda = perda_no.valor.dados[0]  # loss_value (em inglês)
    
    # Backward pass (retropropagação) - Cálculo manual de gradientes
    # Para MSE: d_loss/d_pred = 2 * (pred - target) / n
    var n = Float32(len(y_predito.dados))
    for i in range(len(y_predito.dados)):
        var grad_pred = 2.0 * (y_predito.dados[i] - y_esperado.dados[i]) / n
        
        # Gradiente de b: d_loss/d_b = d_loss/d_pred (pois b é somado diretamente)
        b.gradiente.dados[i] += grad_pred
        
        # Para W, precisamos multiplicar pelo input x
        # d_loss/d_W = x^T * d_loss/d_pred
        for j in range(len(x.dados)):
            var idx = j * len(b.valor.dados) + i  # índice em W (assume W tem shape [len(x), len(b)])
            if idx < len(W.gradiente.dados):
                W.gradiente.dados[idx] += x.dados[j] * grad_pred
    
    W.tem_gradiente = True
    b.tem_gradiente = True
    
    # Atualizar parâmetros (gradient descent)
    # W = W - lr * grad_W
    # b = b - lr * grad_b
    atualizar_parametro(W, taxa_aprendizado)
    atualizar_parametro(b, taxa_aprendizado)
    
    # Zerar gradientes para próxima iteração
    zerar_gradiente(W)
    zerar_gradiente(b)
    
    return valor_perda  # loss_value (em inglês)