import src.nucleo.Tensor as tensor_defs

# Backend CPU simples — aloca tensores usando a implementação do núcleo
struct CPUBackend:
    fn __init__(out self):
        pass

    fn alocar_tensor(self, var formato: List[Int]):
        return tensor_defs.Tensor(formato^)

    fn descricao(self) -> String:
        return "CPU backend simples"
