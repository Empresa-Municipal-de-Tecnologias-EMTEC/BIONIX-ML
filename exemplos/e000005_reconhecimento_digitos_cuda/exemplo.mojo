import exemplos.e000004_reconhecimento_digitos.exemplo as exemplo_base
import src.computacao.cuda.cuda as cuda_backend


def executar_exemplo():
    print("--- Validação CUDA real antes do treino ---")
    print("Checando dispositivos disponíveis...")
    var dispositivos = cuda_backend.listar_dispositivos_disponiveis_cuda()
    for i in range(len(dispositivos)):
        print("[", i, "] ", dispositivos[i])

    if not cuda_backend.gpu_disponivel_cuda():
        print("Nenhuma GPU CUDA disponível. Encerrando exemplo e000005 sem fallback para CPU.")
        return

    var nome_gpu = cuda_backend.gpu_nome_dispositivo()
    var ok = cuda_backend.smoke_test_vector_add_cuda()
    print("GPU detectada:", nome_gpu)
    if ok:
        print("Smoke test vector_add CUDA: OK")
    else:
        print("Smoke test vector_add CUDA: FALHOU")

    exemplo_base.executar_exemplo_configuravel(
        "cuda",
        "exemplos/e000005_reconhecimento_digitos_cuda/pesos_mlp_digits_cuda.txt",
        "e000005_cuda",
    )
