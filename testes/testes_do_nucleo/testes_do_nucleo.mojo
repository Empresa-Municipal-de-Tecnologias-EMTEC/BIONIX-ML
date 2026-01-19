def run_tests():
    # Testes agrupados do núcleo Bionix (forward + backward)
    import src.nucleo.nucleo as nucleo

    print("--- Iniciando testes do núcleo Bionix ---")

    # Teste 1: Soma elementwise (elemento por elemento)
    var formato1 = List[Int](2)  # shape1 (em inglês)
    formato1.append(2)
    formato1.append(2)
    var a = nucleo.Tensor(formato1^)
    var formato2 = List[Int](2)  # shape2 (em inglês)
    formato2.append(2)
    formato2.append(2)
    var b = nucleo.Tensor(formato2^)
    a.dados[0] = 1.0  # data (em inglês)
    a.dados[1] = 2.0
    a.dados[2] = 3.0
    a.dados[3] = 4.0
    b.dados[0] = 0.5
    b.dados[1] = 1.5
    b.dados[2] = -1.0
    b.dados[3] = 2.0

    var resultado_soma = nucleo.somar(a, b)  # out_add (em inglês)
    print("resultado da soma:")  # add result
    for i in range(len(resultado_soma.dados)):
        print("  ", resultado_soma.dados[i])

    # Teste 2: Multiplicação de Matrizes (2x2) - Matmul
    var formatoA = List[Int](2)  # shapeA (em inglês)
    formatoA.append(2)
    formatoA.append(2)
    var A = nucleo.Tensor(formatoA^)
    var formatoB = List[Int](2)  # shapeB (em inglês)
    formatoB.append(2)
    formatoB.append(2)
    var B = nucleo.Tensor(formatoB^)
    A.dados[0] = 1.0
    A.dados[1] = 2.0
    A.dados[2] = 3.0
    A.dados[3] = 4.0
    B.dados[0] = 5.0
    B.dados[1] = 6.0
    B.dados[2] = 7.0
    B.dados[3] = 8.0

    var resultado_matmul = nucleo.multiplicar_matrizes(A, B)  # out_mat (em inglês)
    print("resultado da multiplicação de matrizes:")  # matmul result
    for i in range(len(resultado_matmul.dados)):
        print("  ", resultado_matmul.dados[i])

    # Teste 3: Autograd (gradiente automático) simples com somar_nos (autograd simplificado)
    var nc = nucleo.somar_nos(a.copy(), b.copy())  # nc (node c - nó c)
    print("resultado de somar_nos:")  # add_nodes result
    for i in range(len(nc.valor.dados)):
        print("  ", nc.valor.dados[i])

    # Teste 4: Multiplicar_matrizes_nos + MSE (Erro Quadrático Médio)
    var n_saida = nucleo.multiplicar_matrizes_nos(A.copy(), B.copy())  # nOut (node output - nó de saída)
    print("resultado de multiplicar_matrizes_nos:")  # matmul_nodes result
    for i in range(len(n_saida.valor.dados)):
        print("  ", n_saida.valor.dados[i])

    # Criar alvo e calcular MSE (Mean Squared Error - Erro Quadrático Médio)
    var formato_alvo = List[Int](2)  # shape_target (em inglês)
    formato_alvo.append(2)
    formato_alvo.append(2)
    var alvo = nucleo.Tensor(formato_alvo^)  # target (em inglês)
    alvo.dados[0] = 1.0
    alvo.dados[1] = 1.0
    alvo.dados[2] = 1.0
    alvo.dados[3] = 1.0

    var tensor_mse = nucleo.erro_quadratico_medio(n_saida.valor, alvo)  # mse_tensor (tensor MSE)
    print("perda (MSE - Erro Quadrático Médio):", tensor_mse.dados[0])  # loss (MSE)

    # Teste 5: Multiplicação elementwise (elemento por elemento)
    print("\n--- Teste de multiplicação elementwise ---")
    var formato_mult = List[Int](2)  # shape_mult (em inglês)
    formato_mult.append(2)
    formato_mult.append(2)
    var C = nucleo.Tensor(formato_mult^)
    var formato_mult2 = List[Int](2)
    formato_mult2.append(2)
    formato_mult2.append(2)
    var D = nucleo.Tensor(formato_mult2^)
    C.dados[0] = 2.0
    C.dados[1] = 3.0
    C.dados[2] = 4.0
    C.dados[3] = 5.0
    D.dados[0] = 1.0
    D.dados[1] = 2.0
    D.dados[2] = 3.0
    D.dados[3] = 4.0
    
    var resultado_mult = nucleo.multiplicar(C, D)  # out_mult (em inglês)
    print("resultado da multiplicação elementwise:")  # multiply result
    for i in range(len(resultado_mult.dados)):
        print("  ", resultado_mult.dados[i])

    # Teste 6: multiplicar_nos (autograd)
    var n_mult = nucleo.multiplicar_nos(C.copy(), D.copy())  # n_mult (node multiply)
    print("\nresultado de multiplicar_nos:")  # multiply_nodes result
    for i in range(len(n_mult.valor.dados)):
        print("  ", n_mult.valor.dados[i])

    # Teste 7: Backward pass e gradientes
    print("\n--- Teste de backward e gradientes ---")
    var formato_grad = List[Int](1)  # shape_grad (em inglês)
    formato_grad.append(2)
    var tensor_grad = nucleo.Tensor(formato_grad^)
    tensor_grad.dados[0] = 3.0
    tensor_grad.dados[1] = 5.0
    var no_grad = nucleo.no_de_tensor(tensor_grad^)  # node_grad (em inglês)
    
    print("gradiente antes do backward:", no_grad.tem_gradiente)  # has_grad
    nucleo.retropropagar(no_grad)
    print("gradiente após backward:", no_grad.tem_gradiente)  # has_grad
    print("valores do gradiente:")  # gradient values
    for i in range(len(no_grad.gradiente.dados)):
        print("  ", no_grad.gradiente.dados[i])
    
    nucleo.zerar_gradiente(no_grad)
    print("gradiente após zerar:", no_grad.tem_gradiente)  # has_grad

    # Teste 8: Training loop simplificado
    print("\n--- Teste de training loop (passo de treinamento) ---")
    var formato_x = List[Int](2)  # shape_x (em inglês)
    formato_x.append(1)
    formato_x.append(2)
    var x_treino = nucleo.Tensor(formato_x^)  # x_train (em inglês)
    x_treino.dados[0] = 1.0
    x_treino.dados[1] = 2.0
    
    var formato_w = List[Int](2)  # shape_w (em inglês)
    formato_w.append(2)
    formato_w.append(1)
    var tensor_w = nucleo.Tensor(formato_w^)  # tensor_w (weights tensor)
    tensor_w.dados[0] = 0.5
    tensor_w.dados[1] = 0.3
    var w_no = nucleo.no_de_tensor(tensor_w^)  # w_node (weight node)
    
    var formato_b = List[Int](2)  # shape_b (em inglês)
    formato_b.append(1)
    formato_b.append(1)
    var tensor_b = nucleo.Tensor(formato_b^)  # tensor_b (bias tensor)
    tensor_b.dados[0] = 0.1
    var b_no = nucleo.no_de_tensor(tensor_b^)  # b_node (bias node)
    
    var formato_y = List[Int](2)  # shape_y (em inglês)
    formato_y.append(1)
    formato_y.append(1)
    var y_alvo = nucleo.Tensor(formato_y^)  # y_target (em inglês)
    y_alvo.dados[0] = 2.0
    
    print("W inicial:", w_no.valor.dados[0], w_no.valor.dados[1])
    print("b inicial:", b_no.valor.dados[0])
    
    var perda_treino = nucleo.passo_treinamento(x_treino.copy(), y_alvo.copy(), w_no, b_no, 0.01)  # loss_train (em inglês)
    print("perda após 1 passo:", perda_treino)  # loss after 1 step
    
    print("perda após 1 passo:", perda_treino)  # loss after 1 step
    
    print("--- Fim dos testes do núcleo ---")
