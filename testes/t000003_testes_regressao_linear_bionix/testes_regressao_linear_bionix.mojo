# Testes de Regressão Linear com Perceptron
# 
# PROBLEMA: Classificação de aprovação de empréstimo bancário
# Dataset com 3 variáveis de entrada:
#   - Renda anual (normalizada 0-1): salário do cliente
#   - Anos de histórico de crédito (normalizado 0-1): tempo de relacionamento bancário
#   - Score de crédito (normalizado 0-1): pontuação de crédito
# 
# Saída (target): 1.0 = Aprovar empréstimo, 0.0 = Negar empréstimo
# 
# DATASET (5 exemplos):
# Exemplo 1: [0.8, 0.9, 0.7] -> 1.0 (Renda alta, histórico longo, bom score = APROVAR)
# Exemplo 2: [0.2, 0.3, 0.4] -> 0.0 (Renda baixa, histórico curto, score ruim = NEGAR)
# Exemplo 3: [0.9, 0.8, 0.9] -> 1.0 (Renda muito alta, bom histórico, ótimo score = APROVAR)
# Exemplo 4: [0.3, 0.4, 0.3] -> 0.0 (Renda baixa, histórico médio, score baixo = NEGAR)
# Exemplo 5: [0.7, 0.6, 0.8] -> 1.0 (Renda boa, histórico razoável, bom score = APROVAR)
#
# MODELO: Perceptron simples (regressão linear)
# y = W1*x1 + W2*x2 + W3*x3 + b
# onde: W = pesos, b = bias, x = features
#
# TREINAMENTO: Gradient descent com MSE loss
# Objetivo: Ajustar W e b para minimizar erro entre predição e target

from src.nucleo.nucleo import Tensor, No
from src.nucleo.nucleo import somar_nos_grafo, multiplicar_matrizes_nos_grafo, no_erro_quadratico_medio_grafo
from src.nucleo.nucleo import retropropagar_com_grafo

fn teste_regressao_linear():
    print("\\n=== TESTE: Regressão Linear - Aprovação de Empréstimo ===\\n")
    
    # ===== PREPARAÇÃO DO DATASET =====
    print("Dataset: 5 clientes bancários")
    print("Features: [Renda, Histórico Crédito, Score]")
    print("Target: 1.0=Aprovar, 0.0=Negar\\n")
    
    # Criar tensores de entrada (5 exemplos x 3 features)
    var formato_x = List[Int]()
    formato_x.append(5)  # 5 exemplos
    formato_x.append(3)  # 3 features
    var X = Tensor(formato_x^)
    
    # Exemplo 1: [0.8, 0.9, 0.7] -> 1.0
    X.dados[0] = 0.8
    X.dados[1] = 0.9
    X.dados[2] = 0.7
    
    # Exemplo 2: [0.2, 0.3, 0.4] -> 0.0
    X.dados[3] = 0.2
    X.dados[4] = 0.3
    X.dados[5] = 0.4
    
    # Exemplo 3: [0.9, 0.8, 0.9] -> 1.0
    X.dados[6] = 0.9
    X.dados[7] = 0.8
    X.dados[8] = 0.9
    
    # Exemplo 4: [0.3, 0.4, 0.3] -> 0.0
    X.dados[9] = 0.3
    X.dados[10] = 0.4
    X.dados[11] = 0.3
    
    # Exemplo 5: [0.7, 0.6, 0.8] -> 1.0
    X.dados[12] = 0.7
    X.dados[13] = 0.6
    X.dados[14] = 0.8
    
    # Criar targets (5 exemplos x 1 saída)
    var formato_y = List[Int]()
    formato_y.append(5)  # 5 exemplos
    formato_y.append(1)  # 1 saída
    var y_true = Tensor(formato_y^)
    y_true.dados[0] = 1.0  # Ex1: Aprovar
    y_true.dados[1] = 0.0  # Ex2: Negar
    y_true.dados[2] = 1.0  # Ex3: Aprovar
    y_true.dados[3] = 0.0  # Ex4: Negar
    y_true.dados[4] = 1.0  # Ex5: Aprovar
    
    # ===== INICIALIZAÇÃO DOS PARÂMETROS =====
    # Pesos: W (3x1) - um peso para cada feature
    var formato_w = List[Int]()
    formato_w.append(3)  # 3 features
    formato_w.append(1)  # 1 saída
    var W = Tensor(formato_w^)
    # Inicialização aleatória pequena (simulada com valores fixos para reprodutibilidade)
    W.dados[0] = 0.1  # peso para Renda
    W.dados[1] = 0.1  # peso para Histórico
    W.dados[2] = 0.1  # peso para Score
    
    # Bias: b (5x1) - mesmo bias para todos os exemplos
    var formato_b = List[Int]()
    formato_b.append(5)
    formato_b.append(1)
    var b = Tensor(formato_b^)
    # Bias inicializado com zeros (já feito pelo construtor)
    
    print("Pesos iniciais W:", W.dados[0], W.dados[1], W.dados[2])
    print("Bias inicial b:", b.dados[0])
    print("Targets y_true:", y_true.dados[0], y_true.dados[1], y_true.dados[2], y_true.dados[3], y_true.dados[4])
    print("Features X[0]:", X.dados[0], X.dados[1], X.dados[2])
    
    # ===== TREINAMENTO =====
    var learning_rate: Float32 = 0.1
    var num_epochs: Int = 100
    var threshold: Float32 = 0.01  # Threshold de convergência
    
    print("\\nIniciando treinamento:")
    print("Learning rate:", learning_rate)
    print("Max épocas:", num_epochs)
    print("Threshold de convergência:", threshold, "\\n")
    
    for epoch in range(num_epochs):
        # FORWARD PASS usando o framework Bionix com autograd completo!
        # Criar nós do grafo computacional
        var X_node = No(X.copy(), "input")
        var W_node = No(W.copy(), "weight")
        var b_node = No(b.copy(), "bias")
        var y_true_node = No(y_true.copy(), "target")
        
        # y_pred = X @ W + b - construindo o grafo com parents!
        var XW_node = multiplicar_matrizes_nos_grafo(X_node, W_node)
        var y_pred_node = somar_nos_grafo(XW_node, b_node)
        
        # loss = MSE(y_pred, y_true) - usando função _grafo do framework
        var loss_node = no_erro_quadratico_medio_grafo(y_pred_node, y_true_node)
        
        var loss = loss_node.valor.dados[0]
        
        # Imprimir progresso a cada 10 épocas
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print("Época", epoch, "| Loss:", loss)
            if epoch % 20 == 0:
                print("  Predições:", 
                      y_pred_node.valor.dados[0], 
                      y_pred_node.valor.dados[1], 
                      y_pred_node.valor.dados[2], 
                      y_pred_node.valor.dados[3], 
                      y_pred_node.valor.dados[4])
        
        # Verificar convergência
        if loss < threshold:
            print("\\n✅ Convergiu na época", epoch, "com loss =", loss)
            
            # Mostrar predições finais
            print("\\nPredições finais:")
            for i in range(5):
                var pred = y_pred_node.valor.dados[i]
                var true_val = y_true.dados[i]
                var decision = "APROVAR" if pred > 0.5 else "NEGAR"
                var correct = "✓" if (pred > 0.5 and true_val > 0.5) or (pred <= 0.5 and true_val <= 0.5) else "✗"
                print("  Cliente", i+1, ": pred =", pred, "| target =", true_val, "|", decision, correct)
            
            print("\\nPesos finais W:", W.dados[0], W.dados[1], W.dados[2])
            print("Bias final b:", b.dados[0])
            break
        
        # BACKWARD PASS usando o framework Bionix com AUTOGRAD AUTOMÁTICO!
        # Esta é a implementação COMPLETA conforme o escopo original:
        # - retropropagar_com_grafo() percorre o grafo automaticamente com stack
        # - Calcula TODOS os gradientes automaticamente
        # - Propaga para os pais usando List[No]
        retropropagar_com_grafo(loss_node)
        
        # Agora os gradientes já foram calculados automaticamente!
        # Precisamos apenas extraí-los dos nós pais e atualizar os parâmetros
        
        # O grafo é: loss <- y_pred <- XW <- [X, W]
        #                       ^
        #                       |
        #                       b
        
        # Gradientes de W: estão em XW_node.pais[1] (segundo pai = W_node)
        # Gradientes de b: estão em y_pred_node.pais[1] (segundo pai = b_node)
        
        # Atualizar W usando gradientes calculados automaticamente pelo autograd
        for i in range(3):
            W.dados[i] -= learning_rate * XW_node.pais[1].gradiente.dados[i]
        
        # Atualizar b usando gradientes calculados automaticamente pelo autograd
        for i in range(5):
            b.dados[i] -= learning_rate * y_pred_node.pais[1].gradiente.dados[i]
    
    print("\\n=== FIM DO TESTE ===\\n")


fn executar_testes():
    teste_regressao_linear()