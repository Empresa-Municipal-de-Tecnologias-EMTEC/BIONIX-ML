struct DriverSessao(Movable, Copyable):
    var modo: String
    var diretorio_disco: String
    var modo_kvcache: String
    var diretorio_kvcache: String

    fn __init__(
        out self,
        var modo_in: String = "nenhum",
        var diretorio_disco_in: String = "",
        var modo_kvcache_in: String = "nenhum",
        var diretorio_kvcache_in: String = "",
    ):
        self.modo = modo_in^
        self.diretorio_disco = diretorio_disco_in^
        self.modo_kvcache = modo_kvcache_in^
        self.diretorio_kvcache = diretorio_kvcache_in^

    fn copy(self) -> DriverSessao:
        return DriverSessao(self.modo, self.diretorio_disco, self.modo_kvcache, self.diretorio_kvcache)^

    fn descricao(self) -> String:
        var base = "DriverSessao(" + self.modo + ")"
        if self.modo == "disco":
            base = base + " dir=" + self.diretorio_disco
        if self.modo_kvcache != "nenhum":
            base = base + " kv=" + self.modo_kvcache
            if self.modo_kvcache == "disco":
                base = base + " kv_dir=" + self.diretorio_kvcache
        return base

    fn nenhum() -> DriverSessao:
        return driver_sessao_nenhum()

    fn vram() -> DriverSessao:
        return driver_sessao_vram()

    fn ram() -> DriverSessao:
        return driver_sessao_ram()

    fn disco(var diretorio: String) -> DriverSessao:
        return driver_sessao_disco(diretorio)

    fn custom(var modo: String, var diretorio: String = "") -> DriverSessao:
        return driver_sessao_custom(modo, diretorio)

    fn com_kvcache(self, var modo_kvcache: String, var diretorio_kvcache: String = "") -> DriverSessao:
        var copia = self.copy()
        configurar_driver_kvcache(copia, modo_kvcache, diretorio_kvcache)
        return copia^


fn _modo_valido(var modo: String) -> Bool:
    return modo == "nenhum" or modo == "vram" or modo == "ram" or modo == "disco"


fn _normalizar_modo(var modo: String) -> String:
    if modo == "VRAM":
        return "vram"
    if modo == "RAM":
        return "ram"
    if modo == "DISCO":
        return "disco"
    if modo == "NENHUM":
        return "nenhum"
    return modo


fn driver_sessao_nenhum() -> DriverSessao:
    return DriverSessao("nenhum", "", "nenhum", "")


fn driver_sessao_vram() -> DriverSessao:
    return DriverSessao("vram", "", "vram", "")


fn driver_sessao_ram() -> DriverSessao:
    return DriverSessao("ram", "", "ram", "")


fn driver_sessao_disco(var diretorio: String) -> DriverSessao:
    debug_assert(len(diretorio) > 0, "DriverSessao.disco requer diretorio")
    return DriverSessao("disco", diretorio, "disco", diretorio)


fn driver_sessao_custom(var modo: String, var diretorio: String = "") -> DriverSessao:
    var modo_ok = _normalizar_modo(modo)
    debug_assert(_modo_valido(modo_ok), "modo de DriverSessao invalido")
    if modo_ok == "disco":
        debug_assert(len(diretorio) > 0, "modo disco requer diretorio")
    return DriverSessao(modo_ok, diretorio, modo_ok, diretorio)


fn configurar_driver_kvcache(mut driver: DriverSessao, var modo_kvcache: String, var diretorio_kvcache: String = ""):
    var modo_ok = _normalizar_modo(modo_kvcache)
    debug_assert(_modo_valido(modo_ok), "modo de KVCache invalido")
    if modo_ok == "disco":
        debug_assert(len(diretorio_kvcache) > 0, "KVCache em disco requer diretorio")
    driver.modo_kvcache = modo_ok
    driver.diretorio_kvcache = diretorio_kvcache^
