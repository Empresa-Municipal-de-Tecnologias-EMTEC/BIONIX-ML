import src.computacao.sessao as sessao_driver
import src.nucleo.Tensor as tensor_defs
import src.uteis.arquivo as arquivo_io
import src.uteis.kv as kv_io
import src.uteis.texto as texto_io
import os


struct StorageSessao(Movable, Copyable):
    var driver: sessao_driver.DriverSessao
    var chaves_ram: List[String]
    var tensores_ram: List[tensor_defs.Tensor]
    var checkpoint_interval: Int
    var contador_saves: Int
    var chaves_manifesto: List[String]
    var max_itens_ram: Int
    var ordem_acesso_ram: List[String]

    fn __init__(out self, driver_in: sessao_driver.DriverSessao):
        self.driver = driver_in.copy()
        self.chaves_ram = List[String]()
        self.tensores_ram = List[tensor_defs.Tensor]()
        self.checkpoint_interval = 0
        self.contador_saves = 0
        self.chaves_manifesto = List[String]()
        self.max_itens_ram = 0
        self.ordem_acesso_ram = List[String]()

    fn copy(self) -> StorageSessao:
        var out = StorageSessao(self.driver)
        out.chaves_ram = self.chaves_ram.copy()
        out.tensores_ram = List[tensor_defs.Tensor]()
        for t in self.tensores_ram:
            out.tensores_ram.append(t.copy())
        out.checkpoint_interval = self.checkpoint_interval
        out.contador_saves = self.contador_saves
        out.chaves_manifesto = self.chaves_manifesto.copy()
        out.max_itens_ram = self.max_itens_ram
        out.ordem_acesso_ram = self.ordem_acesso_ram.copy()
        return out^

    fn descricao(self) -> String:
        return "StorageSessao(" + self.driver.descricao() + ")"


fn criar_storage_sessao(driver: sessao_driver.DriverSessao) -> StorageSessao:
    var storage = StorageSessao(driver)
    if driver.modo == "disco":
        debug_assert(len(driver.diretorio_disco) > 0, "StorageSessao disco requer diretorio")
        var ok_dir = _garantir_diretorio_disco(driver.diretorio_disco)
        debug_assert(ok_dir, "nao foi possivel criar/validar diretorio de sessao em disco")
    return storage^


fn _modo_usa_memoria(var modo: String) -> Bool:
    return modo == "ram" or modo == "vram"


fn _garantir_diretorio_disco(var diretorio: String) -> Bool:
    if len(diretorio) == 0:
        return False
    try:
        if not os.path.isdir(diretorio):
            os.makedirs(diretorio)
    except Exception:
        pass
    try:
        return os.path.isdir(diretorio)
    except Exception:
        return False


fn _sanitizar_chave_arquivo(var chave: String) -> String:
    var out = ""
    for i in range(len(chave)):
        var c = chave[i:i+1]
        if c == "/" or c == "\\" or c == ":" or c == "*" or c == "?" or c == "\"" or c == "<" or c == ">" or c == "|":
            out = out + "_"
        else:
            out = out + c
    return out


fn _int_list_para_csv(var valores: List[Int]) -> String:
    var out = ""
    for i in range(len(valores)):
        out = out + String(valores[i])
        if i < len(valores) - 1:
            out = out + ","
    return out


fn _csv_para_int_list(var texto_csv: String) -> List[Int]:
    var out = List[Int]()
    var itens = texto_io.split_csv_simples(texto_csv)
    for it in itens:
        var f = texto_io.parse_float_ascii(it)
        out.append(Int(f))
    return out^


fn _csv_para_float_list(var texto_csv: String) -> List[Float32]:
    var out = List[Float32]()
    var itens = texto_io.split_csv_simples(texto_csv)
    for it in itens:
        out.append(texto_io.parse_float_ascii(it))
    return out^


fn _caminho_arquivo_tensor(driver: sessao_driver.DriverSessao, var chave: String) -> String:
    var nome = _sanitizar_chave_arquivo(chave) + ".tensor.txt"
    if len(driver.diretorio_disco) == 0:
        return nome
    var sep = "/"
    if driver.diretorio_disco[len(driver.diretorio_disco) - 1:len(driver.diretorio_disco)] == "/" or driver.diretorio_disco[len(driver.diretorio_disco) - 1:len(driver.diretorio_disco)] == "\\":
        sep = ""
    return driver.diretorio_disco + sep + nome


fn _caminho_manifesto(storage: StorageSessao) -> String:
    if storage.driver.modo != "disco":
        return ""
    return _caminho_arquivo_tensor(storage.driver, "_manifest")


fn _registrar_chave_manifesto(mut storage: StorageSessao, var chave: String):
    for ch in storage.chaves_manifesto:
        if ch == chave:
            return
    storage.chaves_manifesto.append(chave)


fn _salvar_manifesto(mut storage: StorageSessao) -> Bool:
    if storage.driver.modo != "disco":
        return False
    var chaves = List[String]()
    var valores = List[String]()
    chaves.append("contador_saves")
    valores.append(String(storage.contador_saves))
    chaves.append("checkpoint_interval")
    valores.append(String(storage.checkpoint_interval))
    var chaves_csv = ""
    for i in range(len(storage.chaves_manifesto)):
        chaves_csv = chaves_csv + storage.chaves_manifesto[i]
        if i < len(storage.chaves_manifesto) - 1:
            chaves_csv = chaves_csv + ","
    chaves.append("tensores")
    valores.append(chaves_csv)
    return kv_io.salvar_kv_arquivo_seguro(_caminho_manifesto(storage), chaves, valores)


fn configurar_checkpoint_incremental(mut storage: StorageSessao, var checkpoint_interval: Int):
    if checkpoint_interval < 0:
        checkpoint_interval = 0
    storage.checkpoint_interval = checkpoint_interval


fn _talvez_checkpoint_incremental(mut storage: StorageSessao):
    if storage.driver.modo != "disco":
        return
    if storage.checkpoint_interval <= 0:
        return
    if storage.contador_saves % storage.checkpoint_interval != 0:
        return
    _ = _salvar_manifesto(storage)


fn _put_prefetch_ram(mut storage: StorageSessao, var chave: String, t: tensor_defs.Tensor):
    for i in range(len(storage.chaves_ram)):
        if storage.chaves_ram[i] == chave:
            storage.tensores_ram[i] = t.copy()
            return
    storage.chaves_ram.append(chave)
    storage.tensores_ram.append(t.copy())
    _touch_chave_ram(storage, chave)
    _evict_ram_se_necessario(storage)


fn _touch_chave_ram(mut storage: StorageSessao, var chave: String):
    var idx: Int = -1
    for i in range(len(storage.ordem_acesso_ram)):
        if storage.ordem_acesso_ram[i] == chave:
            idx = i
            break
    if idx >= 0:
        storage.ordem_acesso_ram.erase(idx)
    storage.ordem_acesso_ram.append(chave)


fn _evict_ram_se_necessario(mut storage: StorageSessao):
    if storage.max_itens_ram <= 0:
        return
    while len(storage.chaves_ram) > storage.max_itens_ram and len(storage.ordem_acesso_ram) > 0:
        var chave_antiga = storage.ordem_acesso_ram[0]
        storage.ordem_acesso_ram.erase(0)

        var idx_rm: Int = -1
        for i in range(len(storage.chaves_ram)):
            if storage.chaves_ram[i] == chave_antiga:
                idx_rm = i
                break
        if idx_rm >= 0:
            storage.chaves_ram.erase(idx_rm)
            storage.tensores_ram.erase(idx_rm)


fn configurar_paginacao_ram(mut storage: StorageSessao, var max_itens_ram: Int):
    if max_itens_ram < 0:
        max_itens_ram = 0
    storage.max_itens_ram = max_itens_ram
    _evict_ram_se_necessario(storage)


fn _salvar_tensor_em_ram(mut storage: StorageSessao, var chave: String, t: tensor_defs.Tensor):
    for i in range(len(storage.chaves_ram)):
        if storage.chaves_ram[i] == chave:
            storage.tensores_ram[i] = t.copy()
            _touch_chave_ram(storage, chave)
            return
    storage.chaves_ram.append(chave)
    storage.tensores_ram.append(t.copy())
    _touch_chave_ram(storage, chave)
    _evict_ram_se_necessario(storage)


fn _salvar_tensor_em_disco(storage: StorageSessao, var chave: String, t: tensor_defs.Tensor) -> Bool:
    var chaves = List[String]()
    var valores = List[String]()
    chaves.append("tipo")
    valores.append(t.tipo_computacao)
    chaves.append("formato")
    valores.append(_int_list_para_csv(t.formato))
    chaves.append("dados")
    valores.append(texto_io.float_list_para_csv(t.dados))

    var caminho = _caminho_arquivo_tensor(storage.driver, chave)
    return kv_io.salvar_kv_arquivo_seguro(caminho, chaves, valores)


fn salvar_tensor_sessao(mut storage: StorageSessao, var chave: String, t: tensor_defs.Tensor) -> Bool:
    if storage.driver.modo == "nenhum":
        return False
    if _modo_usa_memoria(storage.driver.modo):
        _salvar_tensor_em_ram(storage, chave, t)
        return True
    if storage.driver.modo == "disco":
        var ok = _salvar_tensor_em_disco(storage, chave, t)
        if ok:
            storage.contador_saves = storage.contador_saves + 1
            _registrar_chave_manifesto(storage, chave)
            _talvez_checkpoint_incremental(storage)
        return ok
    return False


fn existe_tensor_sessao(storage: StorageSessao, var chave: String) -> Bool:
    if _modo_usa_memoria(storage.driver.modo):
        for i in range(len(storage.chaves_ram)):
            if storage.chaves_ram[i] == chave:
                return True
        return False
    if storage.driver.modo == "disco":
        var caminho = _caminho_arquivo_tensor(storage.driver, chave)
        var conteudo = arquivo_io.ler_texto_seguro(caminho)
        return len(conteudo) > 0
    return False


fn _carregar_tensor_ram(storage: StorageSessao, var chave: String, fallback: tensor_defs.Tensor) -> tensor_defs.Tensor:
    for i in range(len(storage.chaves_ram)):
        if storage.chaves_ram[i] == chave:
            return storage.tensores_ram[i].copy()
    return fallback.copy()


fn _carregar_tensor_disco(storage: StorageSessao, var chave: String, fallback: tensor_defs.Tensor) -> tensor_defs.Tensor:
    var caminho = _caminho_arquivo_tensor(storage.driver, chave)
    var kv = kv_io.carregar_kv_arquivo_seguro(caminho)
    var tipo = kv_io.obter_valor_ou_padrao(kv, "tipo", fallback.tipo_computacao)
    var formato_csv = kv_io.obter_valor_ou_padrao(kv, "formato", "")
    var dados_csv = kv_io.obter_valor_ou_padrao(kv, "dados", "")

    if len(formato_csv) == 0 or len(dados_csv) == 0:
        return fallback.copy()

    var formato = _csv_para_int_list(formato_csv)
    if len(formato) == 0:
        return fallback.copy()

    var out = tensor_defs.Tensor(formato^, tipo)
    var dados = _csv_para_float_list(dados_csv)
    var n = len(out.dados)
    if len(dados) < n:
        n = len(dados)
    for i in range(n):
        out.dados[i] = dados[i]
    return out^


fn carregar_tensor_sessao(storage: StorageSessao, var chave: String, fallback: tensor_defs.Tensor) -> tensor_defs.Tensor:
    if _modo_usa_memoria(storage.driver.modo):
        return _carregar_tensor_ram(storage, chave, fallback)
    if storage.driver.modo == "disco":
        return _carregar_tensor_disco(storage, chave, fallback)
    return fallback.copy()


fn prefetch_tensor_sessao(mut storage: StorageSessao, var chave: String, fallback: tensor_defs.Tensor) -> Bool:
    if storage.driver.modo != "disco":
        return False
    var t = _carregar_tensor_disco(storage, chave, fallback)
    _put_prefetch_ram(storage, chave, t)
    return True
