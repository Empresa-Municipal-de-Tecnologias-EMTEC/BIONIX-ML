import src.nucleo.Tensor as tensor_defs
import src.computacao.tipos as tipos
import src.computacao.cuda.kernels_tensor as kernels_tensor
import src.computacao.sessao as sessao_driver
from gpu.host import DeviceContext
from sys import has_nvidia_gpu_accelerator


# Fachada CUDA (placeholder): API pronta para futura implementação real.
struct CUDABackend(Movable, Copyable):
    var device_id: Int
    var stream_id: Int
    var pipeline_memoria_id: Int

    fn __init__(out self, var device_id_in: Int = 0, var stream_id_in: Int = 0, var pipeline_memoria_id_in: Int = 0):
        self.device_id = device_id_in
        self.stream_id = stream_id_in
        self.pipeline_memoria_id = pipeline_memoria_id_in

    fn alocar_tensor(self, var formato: List[Int]):
        var t = tensor_defs.Tensor(formato^, tipos.backend_nome_de_id(tipos.backend_cuda_id()))
        tensor_defs.configurar_contexto_backend(t, self.device_id, self.pipeline_memoria_id, 0)
        return t^

    fn obter_pipeline_id_operacao(self, var operacao_id: Int) -> Int:
        # Placeholder: futuramente mapeia operação para kernel CUDA compilado/cacheado.
        return self.pipeline_memoria_id * 1000 + operacao_id

    fn descricao(self) -> String:
        return "CUDA backend (fachada, compute pendente)"


struct CUDASessaoExecucao(Movable, Copyable):
    var backend: CUDABackend
    var driver_sessao: sessao_driver.DriverSessao
    var contexto_longo_prazo_habilitado: Bool

    fn __init__(
        out self,
        backend_in: CUDABackend,
        driver_sessao_in: sessao_driver.DriverSessao,
    ):
        self.backend = backend_in
        self.driver_sessao = driver_sessao_in.copy()
        self.contexto_longo_prazo_habilitado = self.driver_sessao.modo == "vram"

    fn descricao(self) -> String:
        return "CUDASessaoExecucao(" + self.driver_sessao.descricao() + ")"


fn criar_sessao_execucao_cuda(
    driver: sessao_driver.DriverSessao,
    var device_id: Int = 0,
    var stream_id: Int = 0,
    var pipeline_memoria_id: Int = 0,
) -> CUDASessaoExecucao:
    var backend = CUDABackend(device_id, stream_id, pipeline_memoria_id)
    return CUDASessaoExecucao(backend, driver)


fn listar_dispositivos_disponiveis_cuda() -> List[String]:
    var dispositivos = List[String]()
    @parameter
    if has_nvidia_gpu_accelerator():
        try:
            var n = DeviceContext.number_of_devices()
            for i in range(n):
                with DeviceContext(i, api="cuda") as ctx:
                    dispositivos.append(String(i) + " - " + ctx.name())
        except _:
            dispositivos.append("nenhum dispositivo CUDA disponivel")
    else:
        dispositivos.append("nenhum dispositivo CUDA disponivel")

    if len(dispositivos) == 0:
        dispositivos.append("nenhum dispositivo CUDA disponivel")
    return dispositivos^


fn gpu_disponivel_cuda() -> Bool:
    return has_nvidia_gpu_accelerator()


fn gpu_nome_dispositivo() -> String:
    var dispositivos = listar_dispositivos_disponiveis_cuda()
    if len(dispositivos) <= 0:
        return "indisponivel"
    return dispositivos[0]


fn smoke_test_vector_add_cuda(var tolerancia_abs: Float32 = 1e-4) -> Bool:
    if not gpu_disponivel_cuda():
        return False

    var shape = List[Int]()
    shape.append(1)
    shape.append(8)
    var a = tensor_defs.Tensor(shape.copy(), tipos.backend_nome_de_id(tipos.backend_cuda_id()))
    var b = tensor_defs.Tensor(shape.copy(), tipos.backend_nome_de_id(tipos.backend_cuda_id()))

    for i in range(8):
        a.dados[i] = Float32(i)
        b.dados[i] = 1.0

    var out = kernels_tensor.somar_elemento_a_elemento_cuda(a, b, pipeline_id=1)
    for i in range(8):
        var esperado = Float32(i) + 1.0
        var diff = out.dados[i] - esperado
        if diff < 0.0:
            diff = -diff
        if diff > tolerancia_abs:
            return False
    return True
