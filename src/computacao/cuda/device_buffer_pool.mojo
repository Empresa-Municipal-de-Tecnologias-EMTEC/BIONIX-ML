# Pool persistente de DeviceBuffer para CUDA
from gpu.host import DeviceBuffer

struct DeviceBufferPool(Movable, Copyable):
    var pool: Dict[Int, List[DeviceBuffer]]  # chave: tamanho, valor: lista de buffers livres

    fn __init__(out self):
        self.pool = Dict[Int, List[DeviceBuffer]]()

    fn acquire(self, var tamanho: Int) -> DeviceBuffer:
        if self.pool.contains(tamanho) and len(self.pool[tamanho]) > 0:
            return self.pool[tamanho].pop()
        return DeviceBuffer(tamanho)

    fn release(self, buf: DeviceBuffer):
        var tamanho = buf.size
        if not self.pool.contains(tamanho):
            self.pool[tamanho] = List[DeviceBuffer]()
        self.pool[tamanho].append(buf)

    fn clear(self):
        self.pool.clear()

# Singleton global (pode ser atrelado ao contexto CUDA futuramente)
global device_buffer_pool = DeviceBufferPool()
