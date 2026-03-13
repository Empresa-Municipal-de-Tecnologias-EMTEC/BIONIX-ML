import src.computacao.sessao as sessao_driver
import src.computacao.storage_sessao as storage_sessao
import src.nucleo.Tensor as tensor_defs


struct KVCacheProvider(Movable, Copyable):
    var storage: storage_sessao.StorageSessao
    var prefixo: String

    fn __init__(out self, storage_in: storage_sessao.StorageSessao, var prefixo_in: String = "kvcache"):
        self.storage = storage_in.copy()
        self.prefixo = prefixo_in^

    fn copy(self) -> KVCacheProvider:
        return KVCacheProvider(self.storage, self.prefixo)

    fn descricao(self) -> String:
        return "KVCacheProvider(" + self.storage.descricao() + ", prefixo=" + self.prefixo + ")"


fn _chave_kv(provider: KVCacheProvider, var camada: Int, var posicao: Int, var tipo: String) -> String:
    return provider.prefixo + "/" + tipo + "/l" + String(camada) + "/p" + String(posicao)


fn criar_kvcache_provider(
    driver: sessao_driver.DriverSessao,
    var prefixo: String = "kvcache",
) -> KVCacheProvider:
    var modo_kv = driver.modo_kvcache
    var dir_kv = driver.diretorio_kvcache
    if len(modo_kv) == 0:
        modo_kv = driver.modo
        dir_kv = driver.diretorio_disco

    var driver_kv = sessao_driver.driver_sessao_custom(modo_kv, dir_kv)
    var storage = storage_sessao.criar_storage_sessao(driver_kv)
    return KVCacheProvider(storage, prefixo)


fn salvar_k_cache(mut provider: KVCacheProvider, var camada: Int, var posicao: Int, t: tensor_defs.Tensor) -> Bool:
    return storage_sessao.salvar_tensor_sessao(provider.storage, _chave_kv(provider, camada, posicao, "k"), t)


fn salvar_v_cache(mut provider: KVCacheProvider, var camada: Int, var posicao: Int, t: tensor_defs.Tensor) -> Bool:
    return storage_sessao.salvar_tensor_sessao(provider.storage, _chave_kv(provider, camada, posicao, "v"), t)


fn carregar_k_cache(provider: KVCacheProvider, var camada: Int, var posicao: Int, fallback: tensor_defs.Tensor) -> tensor_defs.Tensor:
    return storage_sessao.carregar_tensor_sessao(provider.storage, _chave_kv(provider, camada, posicao, "k"), fallback)


fn carregar_v_cache(provider: KVCacheProvider, var camada: Int, var posicao: Int, fallback: tensor_defs.Tensor) -> tensor_defs.Tensor:
    return storage_sessao.carregar_tensor_sessao(provider.storage, _chave_kv(provider, camada, posicao, "v"), fallback)


fn existe_k_cache(provider: KVCacheProvider, var camada: Int, var posicao: Int) -> Bool:
    return storage_sessao.existe_tensor_sessao(provider.storage, _chave_kv(provider, camada, posicao, "k"))


fn existe_v_cache(provider: KVCacheProvider, var camada: Int, var posicao: Int) -> Bool:
    return storage_sessao.existe_tensor_sessao(provider.storage, _chave_kv(provider, camada, posicao, "v"))
