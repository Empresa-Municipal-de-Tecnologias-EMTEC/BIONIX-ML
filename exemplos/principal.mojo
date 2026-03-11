import exemplos.e000001_exemplo.exemplo as exemplo
import exemplos.e000002_modelo_linear.exemplo as exemplo_linear
import exemplos.e000003_espirais_intercaladas.exemplo as exemplo_espirais
import exemplos.e000004_reconhecimento_digitos.exemplo as exemplo_digitos
import exemplos.e000005_reconhecimento_digitos_cuda.exemplo as exemplo_digitos_cuda

def main():
    print("\n" + "="*60)
    print("EXECUTANDO EXEMPLOS DO BIONIX")
    print("="*60)

    print("\n[1/5] Exemplo 1: Testes do Núcleo (exemplo)...")
    #exemplo.executar_exemplo()

    print("\n[2/5] Exemplo 2: Modelo Linear com CSV e persistência...")
    #exemplo_linear.executar_exemplo()

    print("\n[3/5] Exemplo 3: Espirais intercaladas com BMP + autograd + MLP...")
    #exemplo_espirais.executar_exemplo()

    print("\n[4/5] Exemplo 4: Reconhecimento de dígitos 0-9 com MLP...")
    exemplo_digitos.executar_exemplo()

    print("\n[5/5] Exemplo 5: Reconhecimento de dígitos 0-9 com MLP (CUDA)...")
    #exemplo_digitos_cuda.executar_exemplo()

    print("\n" + "="*60)
    print("CONCLUÍDO: EXEMPLOS")
    print("="*60 + "\n")
