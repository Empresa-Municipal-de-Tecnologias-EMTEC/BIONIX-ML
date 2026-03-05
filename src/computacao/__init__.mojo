# Pacote `computacao` (backends) - interface de seleção de backend
import src.computacao.cpu.cpu as cpu

def escolher_backend(var nome: String):
    if nome == "cpu":
        return cpu.CPUBackend()
    else:
        raise Exception("Backend não reconhecido: " + nome)
