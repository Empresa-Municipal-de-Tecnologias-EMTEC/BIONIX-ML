def run_tests():
    # Testes agrupados do núcleo Bionix (forward + backward)
    import src.nucleo.nucleo as nucleo

    print("--- Iniciando testes do núcleo Bionix ---")

    # Teste 1: Soma elemento-a-elemento (elementwise em inglês)
    # Cada elemento é somado com o elemento correspondente: a[i] + b[i]
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

    # Teste 2: Multiplicação de Matrizes (matmul em inglês)
    # Produto matricial: cada elemento é a soma dos produtos linha × coluna
    # Diferente de multiplicação elemento-a-elemento
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

    # Teste 3: Soma com nós (nodes) para autograd
    # Autograd = gradiente automático (automatic gradient em inglês)
    # Nós armazenam valores e gradientes para retropropagação
    var nc = nucleo.somar_nos(a.copy(), b.copy())  # nc (node c - nó c)
    print("resultado de somar_nos:")  # add_nodes result
    for i in range(len(nc.valor.dados)):
        print("  ", nc.valor.dados[i])

    # Teste 4: Multiplicação de matrizes com nós + MSE (Erro Quadrático Médio)
    # MSE = Mean Squared Error em inglês, usado para calcular perda/erro
    var n_saida = nucleo.multiplicar_matrizes_nos(A.copy(), B.copy())  # nOut (node output - nó de saída)
    print("resultado de multiplicar_matrizes_nos:")  # matmul_nodes result
    for i in range(len(n_saida.valor.dados)):
        print("  ", n_saida.valor.dados[i])

    # Criar tensor alvo (target em inglês) e calcular MSE
    # MSE mede a diferença entre valores preditos e valores esperados
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

    # Teste 5: Multiplicação elemento-a-elemento (elementwise em inglês)
    # Cada elemento é multiplicado com o elemento correspondente: C[i] * D[i]
    # Diferente de multiplicação de matrizes (matmul)
    print("\n--- Teste de multiplicação elemento-a-elemento ---")
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
    print("resultado da multiplicação elemento-a-elemento:")  # multiply result
    for i in range(len(resultado_mult.dados)):
        print("  ", resultado_mult.dados[i])

    # Teste 6: Multiplicação elemento-a-elemento com nós (para autograd)
    # Permite calcular gradientes da multiplicação durante retropropagação
    var n_mult = nucleo.multiplicar_nos(C.copy(), D.copy())  # n_mult (node multiply)
    print("\nresultado de multiplicar_nos:")  # multiply_nodes result
    for i in range(len(n_mult.valor.dados)):
        print("  ", n_mult.valor.dados[i])

    # Teste 7: Retropropagação (backward pass em inglês) e gerenciamento de gradientes
    # Backward = calcular gradientes de trás para frente no grafo computacional
    # Gradientes são usados para atualizar pesos durante treinamento
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

    # Teste 8: Loop de treinamento (training loop em inglês)
    # Um passo completo: forward pass → calcular perda → backward pass → atualizar pesos
    # Simula como redes neurais aprendem
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
    
    print("W (Weigt, Peso) inicial:", w_no.valor.dados[0], w_no.valor.dados[1])
    print("b (Bias, Viés) inicial:", b_no.valor.dados[0])
    
    var perda_treino = nucleo.passo_treinamento(x_treino.copy(), y_alvo.copy(), w_no, b_no, 0.01)  # loss_train (em inglês)
    print("perda após 1 passo:", perda_treino)  # loss after 1 step
    
    # Teste 9: preenchido_como (filled_like em inglês)
    # Cria um tensor com o mesmo formato de outro tensor, mas preenchido com um valor específico
    print("\n--- Teste de preenchido_como ---")
    var formato_teste = List[Int](2)
    formato_teste.append(2)
    formato_teste.append(3)
    var tensor_ref = nucleo.Tensor(formato_teste^)  # tensor de referência
    var tensor_preenchido = nucleo.preenchido_como(tensor_ref, 7.5)  # filled_like
    print("Tensor preenchido com 7.5:")
    print("  formato:", tensor_preenchido.formato[0], "x", tensor_preenchido.formato[1])
    for i in range(len(tensor_preenchido.dados)):
        print("  ", tensor_preenchido.dados[i])
    
    # Teste 10: somar_paralelo (add_parallel em inglês)
    # Soma paralela usando múltiplos núcleos do processador
    # Deve produzir o mesmo resultado que soma sequencial
    print("\n--- Teste de somar_paralelo ---")
    var formato_par = List[Int](1)
    formato_par.append(4)
    var tensor_par1 = nucleo.Tensor(formato_par^)
    var formato_par2 = List[Int](1)
    formato_par2.append(4)
    var tensor_par2 = nucleo.Tensor(formato_par2^)
    tensor_par1.dados[0] = 1.0
    tensor_par1.dados[1] = 2.0
    tensor_par1.dados[2] = 3.0
    tensor_par1.dados[3] = 4.0
    tensor_par2.dados[0] = 5.0
    tensor_par2.dados[1] = 6.0
    tensor_par2.dados[2] = 7.0
    tensor_par2.dados[3] = 8.0
    
    var resultado_sequencial = nucleo.somar(tensor_par1, tensor_par2)
    var resultado_paralelo = nucleo.somar_paralelo(tensor_par1.copy(), tensor_par2.copy())
    print("Resultado sequencial:", resultado_sequencial.dados[0], resultado_sequencial.dados[1], resultado_sequencial.dados[2], resultado_sequencial.dados[3])
    print("Resultado paralelo:", resultado_paralelo.dados[0], resultado_paralelo.dados[1], resultado_paralelo.dados[2], resultado_paralelo.dados[3])
    
    # Teste 11: Autograd completo (backward automático)
    # Demonstra o cálculo automático de gradientes através de operações compostas
    # Cria grafo: z = (x * y) + w, depois calcula gradientes automaticamente
    print("\n--- Teste de autograd completo ---")
    var formato_auto = List[Int](1)
    formato_auto.append(2)
    var x_auto = nucleo.Tensor(formato_auto^)
    x_auto.dados[0] = 2.0
    x_auto.dados[1] = 3.0
    var formato_auto2 = List[Int](1)
    formato_auto2.append(2)
    var y_auto = nucleo.Tensor(formato_auto2^)
    y_auto.dados[0] = 4.0
    y_auto.dados[1] = 5.0
    var formato_auto3 = List[Int](1)
    formato_auto3.append(2)
    var w_auto = nucleo.Tensor(formato_auto3^)
    w_auto.dados[0] = 1.0
    w_auto.dados[1] = 1.0
    
    # Construir grafo: z = (x * y) + w
    var no_mult = nucleo.multiplicar_nos(x_auto.copy(), y_auto.copy())  # x * y
    var no_final = nucleo.somar_nos(no_mult.valor.copy(), w_auto.copy())  # (x*y) + w
    
    print("Valor de z = (x*y) + w:", no_final.valor.dados[0], no_final.valor.dados[1])
    print("  Esperado: (2*4)+1=9, (3*5)+1=16")
    
    # Calcular gradientes automaticamente
    nucleo.retropropagar(no_final)
    print("Gradiente de z (saída):", no_final.gradiente.dados[0], no_final.gradiente.dados[1])
    print("Gradiente de w:", no_final.grad_entrada_b.dados[0], no_final.grad_entrada_b.dados[1])
    print("  Esperado para w: [1.0, 1.0] (d_z/d_w = 1)")
    print("Gradiente de (x*y):", no_final.grad_entrada_a.dados[0], no_final.grad_entrada_a.dados[1])
    print("  Esperado para (x*y): [1.0, 1.0] (d_z/d_(xy) = 1)")
    
    # Propagar para camada anterior (x*y)
    nucleo.retropropagar(no_mult)
    print("Gradiente de x (via multiplicação):", no_mult.grad_entrada_a.dados[0], no_mult.grad_entrada_a.dados[1])
    print("  Esperado para x: [4.0, 5.0] (d_(xy)/d_x = y)")
    print("Gradiente de y (via multiplicação):", no_mult.grad_entrada_b.dados[0], no_mult.grad_entrada_b.dados[1])
    print("  Esperado para y: [2.0, 3.0] (d_(xy)/d_y = x)")
    
    print("\n--- Fim dos testes do núcleo ---")
