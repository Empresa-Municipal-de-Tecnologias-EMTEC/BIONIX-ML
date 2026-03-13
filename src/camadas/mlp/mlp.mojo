import src.autograd as autograd
import src.autograd.tipos_mlp as tipos_mlp
import src.computacao.dispatcher_gradiente as dispatcher_gradiente
import src.computacao.sessao as sessao_driver
import src.computacao.storage_sessao as storage_sessao
import src.computacao.captura_camadas as captura_camadas
import src.computacao.cuda.cuda as cuda_backend
import src.computacao.cuda.pesos_device_resident as cuda_pesos
import src.conjuntos.lotes_supervisionados as lotes_sup
import src.nucleo.Tensor as tensor_defs
import math

struct BlocoMLP(Movable, Copyable):
    var topologia: List[Int]
    var pesos: List[tensor_defs.Tensor]
    var biases: List[tensor_defs.Tensor]
    var tipo_computacao: String
    var ativacao_saida_id: Int
    var perda_id: Int

    fn __init__(out self, var num_entradas: Int, var num_ocultas: Int = 16, var tipo_computacao_in: String = "cpu"):
        var topologia_local = List[Int]()
        topologia_local.append(num_entradas)
        topologia_local.append(num_ocultas)
        topologia_local.append(num_ocultas)
        topologia_local.append(1)

        _ = self.__init__(topologia_local^, tipo_computacao_in, -1, -1)

    fn __init__(
        out self,
        var topologia_in: List[Int],
        var tipo_computacao_in: String = "cpu",
        var ativacao_saida_id_in: Int = -1,
        var perda_id_in: Int = -1,
    ):
        debug_assert(len(topologia_in) >= 2, "topologia deve ter ao menos entrada e saída")

        self.topologia = topologia_in^
        self.tipo_computacao = tipo_computacao_in^
        self.pesos = List[tensor_defs.Tensor]()
        self.biases = List[tensor_defs.Tensor]()

        var num_saidas = self.topologia[len(self.topologia) - 1]
        self.ativacao_saida_id = ativacao_saida_id_in
        self.perda_id = perda_id_in
        if not tipos_mlp.ativacao_saida_id_valido(self.ativacao_saida_id):
            self.ativacao_saida_id = tipos_mlp.ativacao_saida_padrao_id(num_saidas)
        if not tipos_mlp.perda_id_valido(self.perda_id):
            self.perda_id = tipos_mlp.perda_padrao_id(num_saidas)

        debug_assert(num_saidas > 1 or self.ativacao_saida_id != tipos_mlp.ativacao_saida_softmax_id(), "softmax requer pelo menos 2 saídas")
        debug_assert(self.perda_id != tipos_mlp.perda_cross_entropy_id() or self.ativacao_saida_id == tipos_mlp.ativacao_saida_softmax_id(), "cross-entropy requer ativação de saída softmax")

        var num_camadas = len(self.topologia) - 1
        for camada in range(num_camadas):
            var formato_w = List[Int]()
            formato_w.append(self.topologia[camada])
            formato_w.append(self.topologia[camada + 1])
            var w = tensor_defs.Tensor(formato_w^, self.tipo_computacao)

            var formato_b = List[Int]()
            formato_b.append(1)
            formato_b.append(self.topologia[camada + 1])
            var b = tensor_defs.Tensor(formato_b^, self.tipo_computacao)

            var fan_in = self.topologia[camada]
            var fan_out = self.topologia[camada + 1]
            var limite = Float32(math.sqrt(6.0 / Float64(fan_in + fan_out)))

            var seed = (fan_in * 1103515245 + fan_out * 12345 + (camada + 1) * 97) % 2147483647
            if seed <= 0:
                seed = 1234567 + camada

            for i in range(len(w.dados)):
                seed = (seed * 1664525 + 1013904223 + i) % 2147483647
                var u01 = Float32(seed) / Float32(2147483647)
                var normalizado = u01 * 2.0 - 1.0
                w.dados[i] = normalizado * limite
            for i in range(len(b.dados)):
                b.dados[i] = 0.01 if camada < num_camadas - 1 else Float32(0.0)

            self.pesos.append(w.copy())
            self.biases.append(b.copy())

    fn copy(self) -> BlocoMLP:
        var copia = BlocoMLP(self.topologia.copy(), self.tipo_computacao, self.ativacao_saida_id, self.perda_id)
        copia.pesos = List[tensor_defs.Tensor]()
        for w in self.pesos:
            copia.pesos.append(w.copy())
        copia.biases = List[tensor_defs.Tensor]()
        for b in self.biases:
            copia.biases.append(b.copy())
        return copia^


fn prever(bloco: BlocoMLP, entradas: tensor_defs.Tensor) -> tensor_defs.Tensor:
    debug_assert(len(entradas.formato) == 2, "entradas deve ser tensor 2D")
    var saidas = bloco.topologia[len(bloco.topologia) - 1]
    var formato_y = List[Int]()
    formato_y.append(entradas.formato[0])
    formato_y.append(saidas)
    var alvos_dummy = tensor_defs.Tensor(formato_y^, entradas.tipo_computacao)
    var ctx = autograd.construir_contexto_mlp(entradas, alvos_dummy, bloco.pesos, bloco.biases, bloco.ativacao_saida_id, bloco.perda_id)
    return ctx.pred.copy()


fn inferir(bloco: BlocoMLP, entradas: tensor_defs.Tensor) -> tensor_defs.Tensor:
    return prever(bloco, entradas)


fn _chave_storage_peso(var camada: Int) -> String:
    return "mlp/peso/" + String(camada)


fn _chave_storage_bias(var camada: Int) -> String:
    return "mlp/bias/" + String(camada)


fn _carregar_bloco_de_storage(mut bloco: BlocoMLP, storage: storage_sessao.StorageSessao):
    for camada in range(len(bloco.pesos)):
        bloco.pesos[camada] = storage_sessao.carregar_tensor_sessao(storage, _chave_storage_peso(camada), bloco.pesos[camada])
        bloco.biases[camada] = storage_sessao.carregar_tensor_sessao(storage, _chave_storage_bias(camada), bloco.biases[camada])


fn _salvar_bloco_em_storage(mut bloco: BlocoMLP, mut storage: storage_sessao.StorageSessao):
    for camada in range(len(bloco.pesos)):
        _ = storage_sessao.salvar_tensor_sessao(storage, _chave_storage_peso(camada), bloco.pesos[camada])
        _ = storage_sessao.salvar_tensor_sessao(storage, _chave_storage_bias(camada), bloco.biases[camada])


fn prever_com_sessao(
    mut bloco: BlocoMLP,
    entradas: tensor_defs.Tensor,
    driver_sessao: sessao_driver.DriverSessao = sessao_driver.driver_sessao_nenhum(),
    mut captura_adaptador: captura_camadas.CapturaCamadasAdaptador = captura_camadas.criar_captura_camadas_desativado(),
) -> tensor_defs.Tensor:
    var storage = storage_sessao.criar_storage_sessao(driver_sessao)
    if driver_sessao.modo != "nenhum":
        _carregar_bloco_de_storage(bloco, storage)
    var saida = prever(bloco, entradas)
    _ = captura_camadas.capturar_io_camada(captura_adaptador, "infer", "mlp/bloco", entradas, saida)
    return saida^


fn inferir_com_sessao(
    mut bloco: BlocoMLP,
    entradas: tensor_defs.Tensor,
    driver_sessao: sessao_driver.DriverSessao = sessao_driver.driver_sessao_nenhum(),
    mut captura_adaptador: captura_camadas.CapturaCamadasAdaptador = captura_camadas.criar_captura_camadas_desativado(),
) -> tensor_defs.Tensor:
    return prever_com_sessao(bloco, entradas, driver_sessao, captura_adaptador)


fn _loss_medio_lotes_validacao(bloco: BlocoMLP, lotes_validacao: List[lotes_sup.LoteSupervisionado]) -> Float32:
    if len(lotes_validacao) == 0:
        return 0.0

    var soma: Float32 = 0.0
    for lote in lotes_validacao:
        var pred = inferir(bloco, lote.entradas)
        soma = soma + autograd.calcular_loss_mlp(pred, lote.alvos, bloco.perda_id)

    return soma / Float32(len(lotes_validacao))


fn treinar_por_lotes(
    mut bloco: BlocoMLP,
    lotes_treino_por_epoca: List[lotes_sup.LoteEpocaSupervisionado],
    lotes_validacao: List[lotes_sup.LoteSupervisionado],
    var taxa_aprendizado: Float32 = 0.03,
    var imprimir_cada_epoca: Int = 1,
    var manter_gradientes_na_ram_principal: Bool = False,
    driver_sessao: sessao_driver.DriverSessao = sessao_driver.driver_sessao_nenhum(),
    var checkpoint_interval: Int = 0,
    var max_itens_ram_cache: Int = 0,
    mut captura_adaptador: captura_camadas.CapturaCamadasAdaptador = captura_camadas.criar_captura_camadas_desativado(),
) -> Float32:
    if len(lotes_treino_por_epoca) == 0:
        return 0.0

    var loss_lote_final: Float32 = 0.0
    var epoca_atual = lotes_treino_por_epoca[0].epoca
    var soma_loss_epoca: Float32 = 0.0
    var quantidade_lotes_epoca: Int = 0
    var usar_workspace_cuda = bloco.tipo_computacao == "cuda" and not manter_gradientes_na_ram_principal
    var workspace_cuda = dispatcher_gradiente.criar_workspace_gradiente_cuda()
    var storage = storage_sessao.criar_storage_sessao(driver_sessao)
    storage_sessao.configurar_checkpoint_incremental(storage, checkpoint_interval)
    storage_sessao.configurar_paginacao_ram(storage, max_itens_ram_cache)
    var sessao_execucao_cuda = cuda_backend.criar_sessao_execucao_cuda(driver_sessao, 0, 0, 0)
    dispatcher_gradiente.configurar_driver_sessao_workspace_cuda(
        workspace_cuda,
        sessao_execucao_cuda.driver_sessao.modo,
        sessao_execucao_cuda.driver_sessao.diretorio_disco,
    )

    if driver_sessao.modo != "nenhum":
        _carregar_bloco_de_storage(bloco, storage)
        for camada in range(len(bloco.pesos)):
            _ = storage_sessao.prefetch_tensor_sessao(storage, _chave_storage_peso(camada), bloco.pesos[camada])
            _ = storage_sessao.prefetch_tensor_sessao(storage, _chave_storage_bias(camada), bloco.biases[camada])

    for item in lotes_treino_por_epoca:
        if item.epoca != epoca_atual:
            var loss_medio_epoca = soma_loss_epoca / Float32(quantidade_lotes_epoca) if quantidade_lotes_epoca > 0 else 0.0
            var loss_validacao = _loss_medio_lotes_validacao(bloco, lotes_validacao)
            if imprimir_cada_epoca > 0 and (epoca_atual % imprimir_cada_epoca == 0):
                print("Época", epoca_atual, "| Loss treino médio:", loss_medio_epoca, "| Loss validação:", loss_validacao)

            epoca_atual = item.epoca
            soma_loss_epoca = 0.0
            quantidade_lotes_epoca = 0

        var entradas = item.lote.entradas.copy()
        var alvos = item.lote.alvos.copy()

        var ctx = autograd.construir_contexto_mlp(entradas, alvos, bloco.pesos, bloco.biases, bloco.ativacao_saida_id, bloco.perda_id)
        _ = captura_camadas.capturar_io_camada(captura_adaptador, "train", "mlp/bloco", ctx.entradas, ctx.pred)
        var grads = autograd.MLPGradientes(List[tensor_defs.Tensor](), List[tensor_defs.Tensor](), 0.0)
        if usar_workspace_cuda:
            grads = dispatcher_gradiente.calcular_gradientes_mlp_com_workspace_cuda_e_aplicar_sgd(
                ctx,
                bloco.pesos,
                bloco.biases,
                taxa_aprendizado,
                workspace_cuda,
                manter_gradientes_na_ram_principal,
            )
        else:
            grads = dispatcher_gradiente.calcular_gradientes_mlp(ctx, bloco.pesos, manter_gradientes_na_ram_principal)
        loss_lote_final = grads.loss

        if not usar_workspace_cuda:
            for camada in range(len(bloco.pesos)):
                for i in range(len(bloco.pesos[camada].dados)):
                    bloco.pesos[camada].dados[i] = bloco.pesos[camada].dados[i] - taxa_aprendizado * grads.grad_ws[camada].dados[i]
                for j in range(len(bloco.biases[camada].dados)):
                    bloco.biases[camada].dados[j] = bloco.biases[camada].dados[j] - taxa_aprendizado * grads.grad_bs[camada].dados[j]

        if driver_sessao.modo != "nenhum":
            _salvar_bloco_em_storage(bloco, storage)

        soma_loss_epoca = soma_loss_epoca + grads.loss
        quantidade_lotes_epoca = quantidade_lotes_epoca + 1

    var loss_medio_epoca_final = soma_loss_epoca / Float32(quantidade_lotes_epoca) if quantidade_lotes_epoca > 0 else 0.0
    var loss_validacao_final = _loss_medio_lotes_validacao(bloco, lotes_validacao)
    if imprimir_cada_epoca > 0 and (epoca_atual % imprimir_cada_epoca == 0):
        print("Época", epoca_atual, "| Loss treino médio:", loss_medio_epoca_final, "| Loss validação:", loss_validacao_final)

    return loss_lote_final


fn treinar(
    mut bloco: BlocoMLP,
    entradas: tensor_defs.Tensor,
    alvos: tensor_defs.Tensor,
    var taxa_aprendizado: Float32 = 0.03,
    var epocas: Int = 1200,
    var imprimir_cada: Int = 200,
    var manter_gradientes_na_ram_principal: Bool = False,
    driver_sessao: sessao_driver.DriverSessao = sessao_driver.driver_sessao_nenhum(),
    var checkpoint_interval: Int = 0,
    var max_itens_ram_cache: Int = 0,
    mut captura_adaptador: captura_camadas.CapturaCamadasAdaptador = captura_camadas.criar_captura_camadas_desativado(),
) -> Float32:
    debug_assert(len(entradas.formato) == 2, "entradas deve ser tensor 2D")
    debug_assert(len(alvos.formato) == 2 and alvos.formato[1] == 1, "alvos deve ser tensor 2D [N,1]")
    debug_assert(entradas.formato[0] == alvos.formato[0], "entradas/alvos com número de linhas diferente")
    debug_assert(entradas.formato[1] == bloco.topologia[0], "número de features incompatível")

    var loss_final: Float32 = 0.0
    var usar_workspace_cuda = bloco.tipo_computacao == "cuda" and not manter_gradientes_na_ram_principal
    var workspace_cuda = dispatcher_gradiente.criar_workspace_gradiente_cuda()
    var storage = storage_sessao.criar_storage_sessao(driver_sessao)
    storage_sessao.configurar_checkpoint_incremental(storage, checkpoint_interval)
    storage_sessao.configurar_paginacao_ram(storage, max_itens_ram_cache)
    var sessao_execucao_cuda = cuda_backend.criar_sessao_execucao_cuda(driver_sessao, 0, 0, 0)
    dispatcher_gradiente.configurar_driver_sessao_workspace_cuda(
        workspace_cuda,
        sessao_execucao_cuda.driver_sessao.modo,
        sessao_execucao_cuda.driver_sessao.diretorio_disco,
    )

    if driver_sessao.modo != "nenhum":
        _carregar_bloco_de_storage(bloco, storage)
        for camada in range(len(bloco.pesos)):
            _ = storage_sessao.prefetch_tensor_sessao(storage, _chave_storage_peso(camada), bloco.pesos[camada])
            _ = storage_sessao.prefetch_tensor_sessao(storage, _chave_storage_bias(camada), bloco.biases[camada])

    for epoca in range(epocas):
        var ctx = autograd.construir_contexto_mlp(entradas, alvos, bloco.pesos, bloco.biases, bloco.ativacao_saida_id, bloco.perda_id)
        _ = captura_camadas.capturar_io_camada(captura_adaptador, "train", "mlp/bloco", ctx.entradas, ctx.pred)
        var grads = autograd.MLPGradientes(List[tensor_defs.Tensor](), List[tensor_defs.Tensor](), 0.0)
        if usar_workspace_cuda:
            grads = dispatcher_gradiente.calcular_gradientes_mlp_com_workspace_cuda_e_aplicar_sgd(
                ctx,
                bloco.pesos,
                bloco.biases,
                taxa_aprendizado,
                workspace_cuda,
                manter_gradientes_na_ram_principal,
            )
        else:
            grads = dispatcher_gradiente.calcular_gradientes_mlp(ctx, bloco.pesos, manter_gradientes_na_ram_principal)
        loss_final = grads.loss

        if not usar_workspace_cuda:
            for camada in range(len(bloco.pesos)):
                for i in range(len(bloco.pesos[camada].dados)):
                    bloco.pesos[camada].dados[i] = bloco.pesos[camada].dados[i] - taxa_aprendizado * grads.grad_ws[camada].dados[i]
                for j in range(len(bloco.biases[camada].dados)):
                    bloco.biases[camada].dados[j] = bloco.biases[camada].dados[j] - taxa_aprendizado * grads.grad_bs[camada].dados[j]

        if driver_sessao.modo != "nenhum":
            _salvar_bloco_em_storage(bloco, storage)

        if imprimir_cada > 0 and (epoca % imprimir_cada == 0 or epoca == epocas - 1):
            print("Época", epoca, "| Loss:", loss_final)
            if epoca == 0:
                print("Grafo de computação (forward):")
                for op in ctx.operacoes:
                    print("  -", op)
                print("Nós do grafo:")
                for no in ctx.grafo.nos:
                    print("  *", no)
                print("Arestas do grafo:")
                for ar in ctx.grafo.arestas:
                    print("  ->", ar)

    return loss_final


# Exemplo de forward+update device-resident para uma camada
import src.computacao.cuda.kernels_tensor as cuda_kernels

def exemplo_forward_update_device_resident(
    bloco: BlocoMLPDeviceResident,
    entrada_dev: DeviceBuffer[DType.float32],
    grad_saida_dev: DeviceBuffer[DType.float32],
    batch: Int,
    fan_in: Int,
    fan_out: Int,
    taxa_aprendizado: Float32,
    sincronizar_para_host: Bool = False,
) -> DeviceBuffer[DType.float32]:
    # Forward: matmul + bias (device-resident)
    var w_dev = bloco.pesos[0].buffer
    var b_dev = bloco.biases[0].buffer
    var len_out = batch * fan_out
    var z_dev = cuda_kernels.buffer_pool.device_buffer_pool.acquire(len_out)
    # Matmul (entrada_dev x w_dev -> z_dev)
    # Aqui você chamaria um kernel matmul device-resident (não implementado neste exemplo)
    # cuda_kernels.matmul_device_resident(entrada_dev, w_dev, z_dev, batch, fan_in, fan_out)
    # Bias (z_dev + b_dev -> z_dev)
    # cuda_kernels.add_bias_device_resident(z_dev, b_dev, z_dev, batch, fan_out)
    # Ativação (ReLU, por exemplo)
    # cuda_kernels.relu_inplace_device_resident(z_dev, len_out)
    # Backward/update: SGD direto no device
    var grad_dev = grad_saida_dev
    var w_atualizado_dev = cuda_kernels.aplicar_sgd_em_pesos_device_resident_cuda(w_dev, grad_dev, len_out, taxa_aprendizado, 0)
    # Opcional: sincronizar para host
    if sincronizar_para_host:
        var saida_host = List[Float32]()
        with z_dev.map_to_host() as host:
            for i in range(len_out):
                saida_host.append(host[i])
        return saida_host
    return z_dev
