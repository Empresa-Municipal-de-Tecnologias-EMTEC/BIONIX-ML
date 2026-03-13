# Pacote `computacao` (backends) - interface de seleção de backend
import src.computacao.cpu.cpu as cpu
import src.computacao.vulkan.vulkan as vulkan
import src.computacao.rocm.rocm as rocm
import src.computacao.cuda.cuda as cuda
import src.computacao.tipos as tipos
import src.computacao.sessao as sessao
import src.computacao.storage_sessao as storage_sessao
import src.computacao.kvcache_sessao as kvcache_sessao
import src.computacao.captura_camadas as captura_camadas

def backend_cpu_id() -> Int:
    return tipos.backend_cpu_id()

def backend_vulkan_id() -> Int:
    return tipos.backend_vulkan_id()

def backend_rocm_id() -> Int:
    return tipos.backend_rocm_id()

def backend_cuda_id() -> Int:
    return tipos.backend_cuda_id()

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


def driver_sessao_nenhum() -> sessao.DriverSessao:
    return sessao.driver_sessao_nenhum()


def driver_sessao_vram() -> sessao.DriverSessao:
    return sessao.driver_sessao_vram()


def driver_sessao_ram() -> sessao.DriverSessao:
    return sessao.driver_sessao_ram()


def driver_sessao_disco(var diretorio: String) -> sessao.DriverSessao:
    return sessao.driver_sessao_disco(diretorio)


def criar_storage_sessao(driver: sessao.DriverSessao) -> storage_sessao.StorageSessao:
    return storage_sessao.criar_storage_sessao(driver)


def salvar_tensor_sessao(mut storage: storage_sessao.StorageSessao, var chave: String, t):
    return storage_sessao.salvar_tensor_sessao(storage, chave, t)


def carregar_tensor_sessao(storage: storage_sessao.StorageSessao, var chave: String, fallback):
    return storage_sessao.carregar_tensor_sessao(storage, chave, fallback)


def configurar_checkpoint_incremental_sessao(mut storage: storage_sessao.StorageSessao, var checkpoint_interval: Int):
    storage_sessao.configurar_checkpoint_incremental(storage, checkpoint_interval)


def configurar_paginacao_ram_sessao(mut storage: storage_sessao.StorageSessao, var max_itens_ram: Int):
    storage_sessao.configurar_paginacao_ram(storage, max_itens_ram)


def prefetch_tensor_sessao(mut storage: storage_sessao.StorageSessao, var chave: String, fallback):
    return storage_sessao.prefetch_tensor_sessao(storage, chave, fallback)


def criar_kvcache_provider(driver: sessao.DriverSessao, var prefixo: String = "kvcache") -> kvcache_sessao.KVCacheProvider:
    return kvcache_sessao.criar_kvcache_provider(driver, prefixo)


def criar_captura_camadas_adaptador(
    driver: sessao.DriverSessao,
    var prefixo: String = "destilacao",
    var ativo: Bool = True,
) -> captura_camadas.CapturaCamadasAdaptador:
    return captura_camadas.criar_captura_camadas_adaptador(driver, prefixo, ativo)


def criar_captura_camadas_desativado() -> captura_camadas.CapturaCamadasAdaptador:
    return captura_camadas.criar_captura_camadas_desativado()


def configurar_intervalo_captura_camadas(mut adaptador: captura_camadas.CapturaCamadasAdaptador, var intervalo_captura: Int):
    captura_camadas.configurar_intervalo_captura(adaptador, intervalo_captura)


def configurar_fases_captura_camadas(mut adaptador: captura_camadas.CapturaCamadasAdaptador, var fases: List[String]):
    captura_camadas.configurar_fases_permitidas(adaptador, fases)


def configurar_prefixos_camada_captura(mut adaptador: captura_camadas.CapturaCamadasAdaptador, var prefixos: List[String]):
    captura_camadas.configurar_prefixos_camadas_permitidos(adaptador, prefixos)


def escolher_backend_por_id(var backend_id: Int):
    if backend_id == tipos.backend_cpu_id():
        return cpu.CPUBackend()
    if backend_id == tipos.backend_vulkan_id():
        return vulkan.VulkanBackend()
    if backend_id == tipos.backend_rocm_id():
        return rocm.ROCmBackend()
    if backend_id == tipos.backend_cuda_id():
        return cuda.CUDABackend()
    else:
        raise Exception("Backend não reconhecido (id): " + String(backend_id))

def escolher_backend(var nome: String):
    var nome_ok = tipos.backend_nome_normalizado(nome)
    if nome_ok == "cpu":
        return cpu.CPUBackend()
    if nome_ok == "vulkan":
        return vulkan.VulkanBackend()
    if nome_ok == "rocm":
        return rocm.ROCmBackend()
    if nome_ok == "cuda":
        return cuda.CUDABackend()
    raise Exception("Backend não reconhecido: " + nome)
