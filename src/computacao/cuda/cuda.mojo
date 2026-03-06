import src.nucleo.Tensor as tensor_defs
import src.computacao.tipos as tipos


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
