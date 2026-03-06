import src.nucleo.Tensor as tensor_defs
import src.computacao.tipos as tipos

# Backend CPU simples — aloca tensores usando a implementação do núcleo
struct CPUBackend:
    fn __init__(out self):
        pass

    fn alocar_tensor(self, var formato: List[Int]):
        return tensor_defs.Tensor(formato^, tipos.backend_nome_de_id(tipos.backend_cpu_id()))

    fn descricao(self) -> String:
        return "CPU backend simples"
