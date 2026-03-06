import src.nucleo.Tensor as tensor_defs
import src.computacao.tipos as tipos


# Fachada ROCm (placeholder): API pronta para futura implementação real.
struct ROCmBackend(Movable, Copyable):
    var device_id: Int
    var stream_id: Int
    var pipeline_memoria_id: Int

    fn __init__(out self, var device_id_in: Int = 0, var stream_id_in: Int = 0, var pipeline_memoria_id_in: Int = 0):
        self.device_id = device_id_in
        self.stream_id = stream_id_in
        self.pipeline_memoria_id = pipeline_memoria_id_in

    fn alocar_tensor(self, var formato: List[Int]):
        return tensor_defs.Tensor(formato^, tipos.backend_nome_de_id(tipos.backend_rocm_id()))

    fn obter_pipeline_id_operacao(self, var operacao_id: Int) -> Int:
        # Placeholder: futuramente mapeia operação para kernel HIP/ROCm.
        return self.pipeline_memoria_id * 1000 + operacao_id

    fn descricao(self) -> String:
        return "ROCm backend (fachada, compute pendente)"
