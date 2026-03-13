# Estrutura para pesos device-resident (CUDA)
from gpu.host import DeviceBuffer

struct PesosDeviceResident(Movable, Copyable):
    var buffer: DeviceBuffer
    var formato: List[Int]
    var tipo: String

    fn __init__(out self, var formato_in: List[Int], var tipo_in: String, dados_host: List[Float32]):
        self.formato = formato_in.copy()
        self.tipo = tipo_in
        self.buffer = DeviceBuffer(len(dados_host))
        with self.buffer.map_to_host() as host:
            for i in range(len(dados_host)):
                host[i] = dados_host[i]

    fn to_host(self) -> List[Float32]:
        var out = List[Float32]()
        out.reserve(len(self.buffer))
        with self.buffer.map_to_host() as host:
            for i in range(len(self.buffer)):
                out.append(host[i])
        return out^
