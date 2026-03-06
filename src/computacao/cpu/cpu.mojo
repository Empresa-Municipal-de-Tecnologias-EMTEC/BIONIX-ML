import src.nucleo.Tensor as tensor_defs
import src.computacao.tipos as tipos

# Backend CPU simples — aloca tensores usando a implementação do núcleo
struct CPUBackend(Movable, Copyable):
    var device_id: Int
    var pipeline_memoria_id: Int

    fn __init__(out self, var device_id_in: Int = 0, var pipeline_memoria_id_in: Int = 0):
        self.device_id = device_id_in
        self.pipeline_memoria_id = pipeline_memoria_id_in

    fn alocar_tensor(self, var formato: List[Int]):
        var t = tensor_defs.Tensor(formato^, tipos.backend_nome_de_id(tipos.backend_cpu_id()))
        tensor_defs.configurar_contexto_backend(t, self.device_id, self.pipeline_memoria_id, 0)
        return t^

    fn obter_pipeline_id_operacao(self, var operacao_id: Int) -> Int:
        return self.pipeline_memoria_id * 1000 + operacao_id

    fn descricao(self) -> String:
        return "CPU backend simples"
