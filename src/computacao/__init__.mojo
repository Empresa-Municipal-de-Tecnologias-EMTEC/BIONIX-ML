# Pacote `computacao` (backends) - interface de seleção de backend
import src.computacao.cpu.cpu as cpu
import src.computacao.tipos as tipos

def backend_cpu_id() -> Int:
    return tipos.backend_cpu_id()

def backend_nome_normalizado(var nome: String) -> String:
    return tipos.backend_nome_normalizado(nome)

def backend_id_de_nome(var nome: String) -> Int:
    return tipos.backend_id_de_nome(nome)

def backend_nome_de_id(var backend_id: Int) -> String:
    return tipos.backend_nome_de_id(backend_id)

def backend_nome_valido(var nome: String) -> Bool:
    return tipos.backend_nome_valido(nome)

def backend_id_valido(var backend_id: Int) -> Bool:
    return tipos.backend_id_valido(backend_id)


def escolher_backend_por_id(var backend_id: Int):
    if backend_id == tipos.backend_cpu_id():
        return cpu.CPUBackend()
    else:
        raise Exception("Backend não reconhecido (id): " + String(backend_id))

def escolher_backend(var nome: String):
    var nome_ok = tipos.backend_nome_normalizado(nome)
    if nome_ok == "cpu":
        return cpu.CPUBackend()
    raise Exception("Backend não reconhecido: " + nome)
