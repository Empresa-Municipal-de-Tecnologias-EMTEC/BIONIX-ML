import exemplos.e000004_reconhecimento_digitos.exemplo as exemplo_base
import src.computacao.cuda.cuda as cuda_backend


def executar_exemplo():
    print("--- Validação CUDA real (std.gpu) antes do treino ---")

    if not cuda_backend.gpu_disponivel_cuda():
        print("GPU CUDA compatível não detectada no ambiente atual. Seguindo com backend 'cuda' em modo de compatibilidade.")
    else:
        try:
            var nome_gpu = cuda_backend.gpu_nome_dispositivo()
            var ok = cuda_backend.smoke_test_vector_add_cuda()
            print("GPU detectada:", nome_gpu)
            if ok:
                print("Smoke test vector_add CUDA: OK")
            else:
                print("Smoke test vector_add CUDA: FALHOU")
        except Exception:
            print("Falha ao executar smoke test CUDA real. Seguindo com backend 'cuda' em modo de compatibilidade.")

    exemplo_base.executar_exemplo_configuravel(
        "cuda",
        "exemplos/e000005_reconhecimento_digitos_cuda/pesos_mlp_digits_cuda.txt",
        "e000005_cuda",
    )
