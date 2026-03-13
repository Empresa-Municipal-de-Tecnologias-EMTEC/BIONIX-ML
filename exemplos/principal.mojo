import exemplos.e000001_exemplo.exemplo as exemplo
import exemplos.e000002_modelo_linear.exemplo as exemplo_linear
import exemplos.e000003_espirais_intercaladas.exemplo as exemplo_espirais
import exemplos.e000004_reconhecimento_digitos.exemplo as exemplo_digitos
import exemplos.e000005_reconhecimento_digitos_cuda.exemplo as exemplo_digitos_cuda
import exemplos.e000006_embeding_bpe.exemplo as exemplo_embedding_bpe
import exemplos.e000007_gpt_texto.exemplo as exemplo_gpt_texto
import exemplos.e000008_reconhecimento_facial.exemplo as exemplo_facial_cnn

def main():
    print("\n" + "="*60)
    print("EXECUTANDO EXEMPLOS DO BIONIX")
    print("="*60)

    print("\n[1/8] Exemplo 1: Testes do Núcleo (exemplo)...")
    exemplo.executar_exemplo()

    print("\n[2/8] Exemplo 2: Modelo Linear com CSV e persistência...")
    exemplo_linear.executar_exemplo()

    print("\n[3/8] Exemplo 3: Espirais intercaladas com BMP + autograd + MLP...")
    exemplo_espirais.executar_exemplo()

    print("\n[4/8] Exemplo 4: Reconhecimento de dígitos 0-9 com MLP...")
    exemplo_digitos.executar_exemplo()

    print("\n[5/8] Exemplo 5: Reconhecimento de dígitos 0-9 com MLP (CUDA)...")
    exemplo_digitos_cuda.executar_exemplo()

    print("\n[6/8] Exemplo 6: Embedding com BPE a partir de .txt...")
    exemplo_embedding_bpe.executar_exemplo()

    print("\n[7/8] Exemplo 7: GPT texto (treino + inferência em conversa/instruções/ferramentas)...")
    exemplo_gpt_texto.executar_exemplo()

    print("\n[8/8] Exemplo 8: Reconhecimento facial com bloco CNN...")
    exemplo_facial_cnn.executar_exemplo()

    print("\n" + "="*60)
    print("CONCLUÍDO: EXEMPLOS")
    print("="*60 + "\n")
