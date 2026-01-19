# Importa os testes separados e executa a função `executar_testes()`.
import testes.t000001_testes_do_nucleo.testes_do_nucleo as testes_do_nucleo
import testes.t000002_testes_regressao_linear.testes_regressao_linear as testes_regressao_linear
import testes.t000003_testes_regressao_linear_bionix.testes_regressao_linear_bionix as testes_regressao_bionix

#Ponto de entrada principal para executar os testes.
def main():
    print("\\n" + "="*60)
    print("EXECUTANDO TODOS OS TESTES DO BIONIX")
    print("="*60)
    
    # Teste 1: Testes do núcleo (tensor operations, autograd básico)
    print("\\n[1/3] Testes do Núcleo...")
    testes_do_nucleo.executar_testes()
    
    # Teste 2: Regressão linear manual (sem usar autograd completo)
    print("\\n[2/3] Testes de Regressão Linear (manual)...")
    testes_regressao_linear.executar_testes()
    
    # Teste 3: Regressão linear usando framework Bionix completo
    print("\\n[3/3] Testes de Regressão Linear (Bionix framework)...")
    testes_regressao_bionix.executar_testes()
    
    print("\\n" + "="*60)
    print("TODOS OS TESTES CONCLUÍDOS")
    print("="*60 + "\\n")
