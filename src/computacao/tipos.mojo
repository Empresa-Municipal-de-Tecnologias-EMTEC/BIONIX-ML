# Tipos de backend (enum-like) e utilitários de validação/normalização.

fn backend_cpu_id() -> Int:
    return 0


fn _normalizar_nome_interno(var nome: String) -> String:
    var n = nome.strip()
    if n == "cpu" or n == "CPU" or n == "Cpu" or n == "cPu" or n == "cpU" or n == "CPu" or n == "cPU":
        return "cpu"
    return n


fn backend_id_de_nome(var nome: String) -> Int:
    var n = _normalizar_nome_interno(nome)
    if n == "cpu":
        return backend_cpu_id()
    return -1


fn backend_nome_de_id(var backend_id: Int) -> String:
    if backend_id == backend_cpu_id():
        return "cpu"
    return "desconhecido"


fn backend_nome_normalizado(var nome: String) -> String:
    var id = backend_id_de_nome(nome)
    if id < 0:
        return "desconhecido"
    return backend_nome_de_id(id)


fn backend_id_valido(var backend_id: Int) -> Bool:
    return backend_id == backend_cpu_id()


fn backend_nome_valido(var nome: String) -> Bool:
    return backend_id_de_nome(nome) >= 0