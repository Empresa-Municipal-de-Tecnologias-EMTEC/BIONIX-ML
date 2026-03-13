import src.nucleo.Tensor as tensor_defs
import src.computacao.dispatcher_tensor as dispatcher_tensor
import src.computacao.tipos as backend_tipos
import math


struct AdaptadorAtencao(Movable, Copyable):
    var modo: String
    var ativo: Bool
    var rank: Int
    var alpha: Float32
    var bloco_flash: Int
    var pilha_modos: List[String]

    fn __init__(
        out self,
        var modo_in: String = "nenhum",
        var ativo_in: Bool = False,
        var rank_in: Int = 0,
        var alpha_in: Float32 = 0.0,
        var bloco_flash_in: Int = 128,
        var pilha_modos_in: List[String] = List[String](),
    ):
        self.modo = modo_in^
        self.ativo = ativo_in
        self.rank = rank_in
        self.alpha = alpha_in
        self.bloco_flash = bloco_flash_in
        self.pilha_modos = pilha_modos_in^

    fn copy(self) -> AdaptadorAtencao:
        return AdaptadorAtencao(self.modo, self.ativo, self.rank, self.alpha, self.bloco_flash, self.pilha_modos.copy())


fn criar_adaptador_atencao_nenhum() -> AdaptadorAtencao:
    return AdaptadorAtencao("nenhum", False)


fn criar_adaptador_atencao_svd(var rank: Int = 8) -> AdaptadorAtencao:
    if rank <= 0:
        rank = 8
    return AdaptadorAtencao("svd", True, rank, 1.0, 128, List[String]())


fn criar_adaptador_atencao_lora(var rank: Int = 8, var alpha: Float32 = 16.0) -> AdaptadorAtencao:
    if rank <= 0:
        rank = 8
    if alpha <= 0.0:
        alpha = 16.0
    return AdaptadorAtencao("lora", True, rank, alpha, 128, List[String]())


fn criar_adaptador_atencao_flash(var bloco_flash: Int = 128) -> AdaptadorAtencao:
    if bloco_flash <= 0:
        bloco_flash = 128
    return AdaptadorAtencao("flash", True, 0, 0.0, bloco_flash, List[String]())


fn criar_adaptador_atencao_combinado(var pilha_modos: List[String]) -> AdaptadorAtencao:
    return AdaptadorAtencao("combinado", True, 0, 0.0, 128, pilha_modos)


fn _coef_svd_u(var i: Int, var r: Int, var dim: Int) -> Float32:
    var seed = (i + 1) * 211 + (r + 1) * 463 + dim * 19
    seed = seed % 9949
    var u = Float32(seed) / Float32(9949)
    return (u * 2.0 - 1.0) * 0.05


fn _coef_svd_v(var r: Int, var j: Int, var dim: Int) -> Float32:
    var seed = (r + 1) * 157 + (j + 1) * 887 + dim * 31
    seed = seed % 9923
    var u = Float32(seed) / Float32(9923)
    return (u * 2.0 - 1.0) * 0.05


fn _sigma_svd(var r: Int, var rank: Int) -> Float32:
    var denom = Float32(rank + 1)
    return (Float32(rank - r) + 1.0) / denom


fn _aplicar_svd_referencia_cpu(x: tensor_defs.Tensor, adaptador: AdaptadorAtencao) -> tensor_defs.Tensor:
    if len(x.formato) != 2:
        return x.copy()

    var batch = x.formato[0]
    var dim = x.formato[1]
    if batch <= 0 or dim <= 0:
        return x.copy()

    var rank = adaptador.rank
    if rank <= 0:
        rank = 8
    if rank > dim:
        rank = dim

    var formato = x.formato.copy()
    var out = tensor_defs.Tensor(formato^, x.tipo_computacao)

    var proj = List[Float32]()
    for _ in range(batch * rank):
        proj.append(0.0)

    for b in range(batch):
        for r in range(rank):
            var acc: Float32 = 0.0
            for i in range(dim):
                acc = acc + x.dados[b * dim + i] * _coef_svd_u(i, r, dim)
            proj[b * rank + r] = acc

    var blend = adaptador.alpha
    if blend <= 0.0:
        blend = 1.0
    if blend > 1.0:
        blend = 1.0

    for b in range(batch):
        for j in range(dim):
            var rec: Float32 = 0.0
            for r in range(rank):
                rec = rec + proj[b * rank + r] * _sigma_svd(r, rank) * _coef_svd_v(r, j, dim)
            out.dados[b * dim + j] = (1.0 - blend) * x.dados[b * dim + j] + blend * rec

    return out^


fn _coef_lora_a(var i: Int, var r: Int, var dim: Int) -> Float32:
    var seed = (i + 1) * 131 + (r + 1) * 911 + dim * 17
    seed = seed % 9973
    var u = Float32(seed) / Float32(9973)
    return (u * 2.0 - 1.0) * 0.02


fn _coef_lora_b(var r: Int, var j: Int, var dim: Int) -> Float32:
    var seed = (r + 1) * 719 + (j + 1) * 313 + dim * 29
    seed = seed % 9967
    var u = Float32(seed) / Float32(9967)
    return (u * 2.0 - 1.0) * 0.02


fn _aplicar_lora_referencia_cpu(x: tensor_defs.Tensor, adaptador: AdaptadorAtencao) -> tensor_defs.Tensor:
    if len(x.formato) != 2:
        return x.copy()

    var batch = x.formato[0]
    var dim = x.formato[1]
    if batch <= 0 or dim <= 0:
        return x.copy()

    var rank = adaptador.rank
    if rank <= 0:
        rank = 8

    var formato = x.formato.copy()
    var out = tensor_defs.Tensor(formato^, x.tipo_computacao)

    var escala = adaptador.alpha
    if escala <= 0.0:
        escala = Float32(rank)
    escala = escala / Float32(rank)

    var low_rank = List[Float32]()
    for _ in range(batch * rank):
        low_rank.append(0.0)

    for b in range(batch):
        for r in range(rank):
            var acc: Float32 = 0.0
            for i in range(dim):
                var xbi = x.dados[b * dim + i]
                acc = acc + xbi * _coef_lora_a(i, r, dim)
            low_rank[b * rank + r] = acc

    for b in range(batch):
        for j in range(dim):
            var delta: Float32 = 0.0
            for r in range(rank):
                delta = delta + low_rank[b * rank + r] * _coef_lora_b(r, j, dim)
            out.dados[b * dim + j] = x.dados[b * dim + j] + escala * delta

    return out^


fn _tensor_zeros(var linhas: Int, var colunas: Int, var tipo_computacao: String) -> tensor_defs.Tensor:
    var formato = List[Int]()
    formato.append(linhas)
    formato.append(colunas)
    return tensor_defs.Tensor(formato^, tipo_computacao)


fn _tensor_escalar_like(base: tensor_defs.Tensor, var valor: Float32) -> tensor_defs.Tensor:
    var formato = base.formato.copy()
    var out = tensor_defs.Tensor(formato^, base.tipo_computacao)
    for i in range(len(out.dados)):
        out.dados[i] = valor
    return out^


fn _matriz_lora_a(var dim: Int, var rank: Int, var tipo_computacao: String) -> tensor_defs.Tensor:
    var a = _tensor_zeros(dim, rank, tipo_computacao)
    for i in range(dim):
        for r in range(rank):
            a.dados[i * rank + r] = _coef_lora_a(i, r, dim)
    return a^


fn _matriz_lora_b(var rank: Int, var dim: Int, var tipo_computacao: String) -> tensor_defs.Tensor:
    var b = _tensor_zeros(rank, dim, tipo_computacao)
    for r in range(rank):
        for j in range(dim):
            b.dados[r * dim + j] = _coef_lora_b(r, j, dim)
    return b^


fn _aplicar_lora_dispatcher_cuda(x: tensor_defs.Tensor, adaptador: AdaptadorAtencao) -> tensor_defs.Tensor:
    if len(x.formato) != 2:
        return x.copy()

    var batch = x.formato[0]
    var dim = x.formato[1]
    if batch <= 0 or dim <= 0:
        return x.copy()

    var rank = adaptador.rank
    if rank <= 0:
        rank = 8
    if rank > dim:
        rank = dim

    var escala = adaptador.alpha
    if escala <= 0.0:
        escala = Float32(rank)
    escala = escala / Float32(rank)

    var a = _matriz_lora_a(dim, rank, x.tipo_computacao)
    var b = _matriz_lora_b(rank, dim, x.tipo_computacao)

    var low_rank = dispatcher_tensor.multiplicar_matrizes(x, a)
    var delta = dispatcher_tensor.multiplicar_matrizes(low_rank, b)
    var escala_tensor = _tensor_escalar_like(delta, escala)
    var delta_escalado = dispatcher_tensor.multiplicar_elemento_a_elemento(delta, escala_tensor)
    return dispatcher_tensor.somar_elemento_a_elemento(x, delta_escalado)


fn _dot_linha(x: tensor_defs.Tensor, var i: Int, var j: Int, var dim: Int) -> Float32:
    var acc: Float32 = 0.0
    var bi = i * dim
    var bj = j * dim
    for d in range(dim):
        acc = acc + x.dados[bi + d] * x.dados[bj + d]
    return acc


fn _aplicar_flash_referencia_cpu(x: tensor_defs.Tensor, adaptador: AdaptadorAtencao) -> tensor_defs.Tensor:
    if len(x.formato) != 2:
        return x.copy()

    var seq = x.formato[0]
    var dim = x.formato[1]
    if seq <= 0 or dim <= 0:
        return x.copy()

    var formato = x.formato.copy()
    var out = tensor_defs.Tensor(formato^, x.tipo_computacao)

    var bloco = adaptador.bloco_flash
    if bloco <= 0:
        bloco = 128

    var escala = 1.0 / Float32(math.sqrt(Float64(dim)))

    for inicio in range(0, seq, bloco):
        var fim = inicio + bloco
        if fim > seq:
            fim = seq

        for i in range(inicio, fim):
            var max_s: Float32 = -1e30
            for j in range(seq):
                var s = _dot_linha(x, i, j, dim) * escala
                if s > max_s:
                    max_s = s

            var soma_exp: Float32 = 0.0
            for j in range(seq):
                var s = _dot_linha(x, i, j, dim) * escala
                soma_exp = soma_exp + Float32(math.exp(Float64(s - max_s)))

            if soma_exp <= 0.0:
                soma_exp = 1.0

            for d in range(dim):
                var acc: Float32 = 0.0
                for j in range(seq):
                    var s = _dot_linha(x, i, j, dim) * escala
                    var w = Float32(math.exp(Float64(s - max_s))) / soma_exp
                    acc = acc + w * x.dados[j * dim + d]
                out.dados[i * dim + d] = acc

    return out^


fn _aplicar_modo(x: tensor_defs.Tensor, var modo: String, adaptador: AdaptadorAtencao) -> tensor_defs.Tensor:
    if modo == "svd":
        return _aplicar_svd_referencia_cpu(x, adaptador)
    if modo == "lora":
        if x.id_backend == backend_tipos.backend_cuda_id():
            return _aplicar_lora_dispatcher_cuda(x, adaptador)
        return _aplicar_lora_referencia_cpu(x, adaptador)
    if modo == "flash":
        return _aplicar_flash_referencia_cpu(x, adaptador)
    return x.copy()


fn aplicar_adaptador_atencao(x: tensor_defs.Tensor, adaptador: AdaptadorAtencao) -> tensor_defs.Tensor:
    if not adaptador.ativo or adaptador.modo == "nenhum":
        return x.copy()

    if adaptador.modo == "combinado":
        var atual = x.copy()
        for modo in adaptador.pilha_modos:
            atual = _aplicar_modo(atual, modo, adaptador)
        return atual^

    return _aplicar_modo(x, adaptador.modo, adaptador)
