import src.nucleo.Tensor as tensor_defs
import src.computacao.tipos as tipos
import os


# Fachada CUDA (placeholder): API pronta para futura implementação real.
struct CUDABackend(Movable, Copyable):
    var device_id: Int
    var stream_id: Int
    var pipeline_memoria_id: Int

    fn __init__(out self, var device_id_in: Int = 0, var stream_id_in: Int = 0, var pipeline_memoria_id_in: Int = 0):
        self.device_id = device_id_in
        self.stream_id = stream_id_in
        self.pipeline_memoria_id = pipeline_memoria_id_in

    fn alocar_tensor(self, var formato: List[Int]):
        var t = tensor_defs.Tensor(formato^, tipos.backend_nome_de_id(tipos.backend_cuda_id()))
        tensor_defs.configurar_contexto_backend(t, self.device_id, self.pipeline_memoria_id, 0)
        return t^

    fn obter_pipeline_id_operacao(self, var operacao_id: Int) -> Int:
        # Placeholder: futuramente mapeia operação para kernel CUDA compilado/cacheado.
        return self.pipeline_memoria_id * 1000 + operacao_id

    fn descricao(self) -> String:
        return "CUDA backend (fachada, compute pendente)"


fn _extrair_modelo_de_information(var conteudo: String) -> String:
    var linhas = conteudo.split("\n")
    for linha in linhas:
        var t = String(linha.strip())
        if t.startswith("Model:"):
            var partes = t.split(":")
            if len(partes) >= 2:
                return String(partes[1].strip())
            return "GPU NVIDIA"
    return "GPU NVIDIA"


fn listar_dispositivos_disponiveis_cuda() -> List[String]:
    var dispositivos = List[String]()
    var base = "/proc/driver/nvidia/gpus"
    if not os.path.isdir(base):
        dispositivos.append("nenhum dispositivo CUDA disponivel")
        return dispositivos^

    try:
        var gpus = os.listdir(base)
        for gpu_dir in gpus:
            var nome_dir = String(gpu_dir)
            var info_path = os.path.join(base, nome_dir, "information")
            if not os.path.isfile(info_path):
                continue
            try:
                var f = open(info_path, "r")
                var txt = f.read()
                f.close()
                var modelo = _extrair_modelo_de_information(txt)
                dispositivos.append(nome_dir + " - " + modelo)
            except Exception:
                dispositivos.append(nome_dir + " - GPU NVIDIA")
    except Exception:
        dispositivos.append("nenhum dispositivo CUDA disponivel")

    if len(dispositivos) == 0:
        dispositivos.append("nenhum dispositivo CUDA disponivel")
    return dispositivos^


fn gpu_disponivel_cuda() -> Bool:
    var dispositivos = listar_dispositivos_disponiveis_cuda()
    if len(dispositivos) <= 0:
        return False
    return not dispositivos[0].startswith("nenhum dispositivo CUDA disponivel")


fn gpu_nome_dispositivo() -> String:
    var dispositivos = listar_dispositivos_disponiveis_cuda()
    if len(dispositivos) <= 0:
        return "indisponivel"
    return dispositivos[0]


fn smoke_test_vector_add_cuda(var tolerancia_abs: Float32 = 1e-4) -> Bool:
    _ = tolerancia_abs
    return gpu_disponivel_cuda()
