import src.computacao.sessao as sessao_driver
import src.computacao.storage_sessao as storage_sessao
import src.nucleo.Tensor as tensor_defs


struct CapturaCamadasAdaptador(Movable, Copyable):
    var ativo: Bool
    var storage: storage_sessao.StorageSessao
    var prefixo: String
    var contador_eventos: Int
    var intervalo_captura: Int
    var fases_permitidas: List[String]
    var prefixos_camadas_permitidos: List[String]

    fn __init__(
        out self,
        storage_in: storage_sessao.StorageSessao,
        var prefixo_in: String = "destilacao",
        var ativo_in: Bool = True,
        var intervalo_captura_in: Int = 1,
        var fases_permitidas_in: List[String] = List[String](),
        var prefixos_camadas_permitidos_in: List[String] = List[String](),
    ):
        self.ativo = ativo_in
        self.storage = storage_in.copy()
        self.prefixo = prefixo_in^
        self.contador_eventos = 0
        self.intervalo_captura = intervalo_captura_in
        if self.intervalo_captura <= 0:
            self.intervalo_captura = 1
        self.fases_permitidas = fases_permitidas_in^
        self.prefixos_camadas_permitidos = prefixos_camadas_permitidos_in^

    fn copy(self) -> CapturaCamadasAdaptador:
        var out = CapturaCamadasAdaptador(
            self.storage,
            self.prefixo,
            self.ativo,
            self.intervalo_captura,
            self.fases_permitidas.copy(),
            self.prefixos_camadas_permitidos.copy(),
        )
        out.contador_eventos = self.contador_eventos
        return out^

    fn descricao(self) -> String:
        return "CapturaCamadasAdaptador(ativo=" + String(self.ativo) + ", prefixo=" + self.prefixo + ")"


fn criar_captura_camadas_desativado() -> CapturaCamadasAdaptador:
    var storage = storage_sessao.criar_storage_sessao(sessao_driver.driver_sessao_nenhum())
    return CapturaCamadasAdaptador(storage, "destilacao", False, 1, List[String](), List[String]())


fn criar_captura_camadas_adaptador(
    driver_sessao: sessao_driver.DriverSessao,
    var prefixo: String = "destilacao",
    var ativo: Bool = True,
    var intervalo_captura: Int = 1,
    var fases_permitidas: List[String] = List[String](),
    var prefixos_camadas_permitidos: List[String] = List[String](),
) -> CapturaCamadasAdaptador:
    var storage = storage_sessao.criar_storage_sessao(driver_sessao)
    return CapturaCamadasAdaptador(
        storage,
        prefixo,
        ativo,
        intervalo_captura,
        fases_permitidas,
        prefixos_camadas_permitidos,
    )


fn configurar_intervalo_captura(mut adaptador: CapturaCamadasAdaptador, var intervalo_captura: Int):
    if intervalo_captura <= 0:
        intervalo_captura = 1
    adaptador.intervalo_captura = intervalo_captura


fn configurar_fases_permitidas(mut adaptador: CapturaCamadasAdaptador, var fases: List[String]):
    adaptador.fases_permitidas = fases^


fn configurar_prefixos_camadas_permitidos(mut adaptador: CapturaCamadasAdaptador, var prefixos: List[String]):
    adaptador.prefixos_camadas_permitidos = prefixos^


fn _fase_permitida(adaptador: CapturaCamadasAdaptador, var fase: String) -> Bool:
    if len(adaptador.fases_permitidas) == 0:
        return True
    for f in adaptador.fases_permitidas:
        if f == fase:
            return True
    return False


fn _camada_permitida(adaptador: CapturaCamadasAdaptador, var camada: String) -> Bool:
    if len(adaptador.prefixos_camadas_permitidos) == 0:
        return True
    for p in adaptador.prefixos_camadas_permitidos:
        if len(camada) >= len(p) and camada[0:len(p)] == p:
            return True
    return False


fn _chave_captura(
    adaptador: CapturaCamadasAdaptador,
    var fase: String,
    var camada: String,
    var io: String,
    var evento: Int,
) -> String:
    return adaptador.prefixo + "/" + fase + "/" + camada + "/" + io + "/e" + String(evento)


fn capturar_io_camada(
    mut adaptador: CapturaCamadasAdaptador,
    var fase: String,
    var camada: String,
    entrada: tensor_defs.Tensor,
    saida: tensor_defs.Tensor,
) -> Bool:
    if not adaptador.ativo:
        return False

    var evento = adaptador.contador_eventos
    adaptador.contador_eventos = adaptador.contador_eventos + 1

    if evento % adaptador.intervalo_captura != 0:
        return False
    if not _fase_permitida(adaptador, fase):
        return False
    if not _camada_permitida(adaptador, camada):
        return False

    var k_in = _chave_captura(adaptador, fase, camada, "entrada", evento)
    var k_out = _chave_captura(adaptador, fase, camada, "saida", evento)

    var ok_in = storage_sessao.salvar_tensor_sessao(adaptador.storage, k_in, entrada)
    var ok_out = storage_sessao.salvar_tensor_sessao(adaptador.storage, k_out, saida)
    return ok_in and ok_out


fn carregar_entrada_capturada(
    adaptador: CapturaCamadasAdaptador,
    var fase: String,
    var camada: String,
    var evento: Int,
    fallback: tensor_defs.Tensor,
) -> tensor_defs.Tensor:
    var k = _chave_captura(adaptador, fase, camada, "entrada", evento)
    return storage_sessao.carregar_tensor_sessao(adaptador.storage, k, fallback)


fn carregar_saida_capturada(
    adaptador: CapturaCamadasAdaptador,
    var fase: String,
    var camada: String,
    var evento: Int,
    fallback: tensor_defs.Tensor,
) -> tensor_defs.Tensor:
    var k = _chave_captura(adaptador, fase, camada, "saida", evento)
    return storage_sessao.carregar_tensor_sessao(adaptador.storage, k, fallback)
