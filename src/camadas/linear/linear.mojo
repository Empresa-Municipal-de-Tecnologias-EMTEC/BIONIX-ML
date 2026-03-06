import src.nucleo.Tensor as tensor_defs
import src.uteis as uteis

struct CamadaLinear(Movable, Copyable):
    var pesos: tensor_defs.Tensor
    var bias: tensor_defs.Tensor
    var tipo_computacao: String

    fn __init__(out self, var num_entradas: Int, var tipo_computacao_in: String = "cpu"):
        self.tipo_computacao = tipo_computacao_in^

        var formato_pesos = List[Int]()
        formato_pesos.append(num_entradas)
        formato_pesos.append(1)
        self.pesos = tensor_defs.Tensor(formato_pesos^, self.tipo_computacao)

        var formato_bias = List[Int]()
        formato_bias.append(1)
        formato_bias.append(1)
        self.bias = tensor_defs.Tensor(formato_bias^, self.tipo_computacao)

        for i in range(len(self.pesos.dados)):
            self.pesos.dados[i] = Float32(i + 1) * 0.01
        self.bias.dados[0] = 0.0

    fn copy(self) -> CamadaLinear:
        var nova = CamadaLinear(self.pesos.formato[0], self.tipo_computacao)
        nova.pesos = self.pesos.copy()
        nova.bias = self.bias.copy()
        return nova^


fn prever(camada: CamadaLinear, entradas: tensor_defs.Tensor) -> tensor_defs.Tensor:
    debug_assert(len(entradas.formato) == 2, "entradas deve ser tensor 2D")
    debug_assert(entradas.formato[1] == camada.pesos.formato[0], "número de features incompatível")
    var projecao = tensor_defs.multiplicar_matrizes(entradas, camada.pesos)
    return tensor_defs.adicionar_bias_coluna(projecao, camada.bias)


fn inferir(camada: CamadaLinear, entradas: tensor_defs.Tensor) -> tensor_defs.Tensor:
    return prever(camada, entradas)


fn erro_quadratico_medio(predicoes: tensor_defs.Tensor, alvos: tensor_defs.Tensor) -> Float32:
    return tensor_defs.erro_quadratico_medio_escalar(predicoes, alvos)


fn treinar(
    mut camada: CamadaLinear,
    entradas: tensor_defs.Tensor,
    alvos: tensor_defs.Tensor,
    var taxa_aprendizado: Float32 = 0.05,
    var epocas: Int = 500,
    var imprimir_cada: Int = 100,
) -> Float32:
    debug_assert(len(entradas.formato) == 2, "entradas deve ser tensor 2D")
    debug_assert(len(alvos.formato) == 2, "alvos deve ser tensor 2D")
    debug_assert(entradas.formato[0] == alvos.formato[0], "número de linhas de entradas e alvos deve ser igual")
    debug_assert(alvos.formato[1] == 1, "alvos deve ter uma coluna")

    var amostras = entradas.formato[0]
    var features = entradas.formato[1]
    var n = Float32(amostras)
    var loss_final: Float32 = 0.0

    for epoca in range(epocas):
        var pred = prever(camada, entradas)
        loss_final = erro_quadratico_medio(pred, alvos)

        var grad_pred = tensor_defs.gradiente_mse(pred, alvos)
        var x_t = tensor_defs.transpor(entradas)
        var grad_w = tensor_defs.multiplicar_matrizes(x_t, grad_pred)
        var grad_b = tensor_defs.soma_total(grad_pred)

        for j in range(features):
            camada.pesos.dados[j] = camada.pesos.dados[j] - taxa_aprendizado * grad_w.dados[j]
        camada.bias.dados[0] = camada.bias.dados[0] - taxa_aprendizado * grad_b

        if imprimir_cada > 0 and (epoca % imprimir_cada == 0 or epoca == epocas - 1):
            print("Época", epoca, "| MSE:", loss_final)

    return loss_final


fn salvar_pesos(camada: CamadaLinear, var caminho: String):
    var chaves = List[String]()
    var valores = List[String]()
    chaves.append("tipo")
    valores.append(camada.tipo_computacao)
    chaves.append("num_entradas")
    valores.append(String(camada.pesos.formato[0]))
    chaves.append("pesos")
    valores.append(uteis.float_list_para_csv(camada.pesos.dados.copy()))
    chaves.append("bias")
    valores.append(String(camada.bias.dados[0]))
    _ = uteis.salvar_kv_arquivo_seguro(caminho, chaves, valores)


fn carregar_pesos(var caminho: String, var tipo_computacao_padrao: String = "cpu") -> CamadaLinear:
    var kv = uteis.carregar_kv_arquivo_seguro(caminho)
    if len(kv.chaves) == 0:
        return CamadaLinear(1, tipo_computacao_padrao)^

    var tipo = uteis.obter_valor_ou_padrao(kv, "tipo", tipo_computacao_padrao)
    var num_entradas: Int = Int(uteis.parse_float_ascii(uteis.obter_valor_ou_padrao(kv, "num_entradas", "0")))
    var pesos_lidos = List[Float32]()
    var pesos_csv = uteis.obter_valor_ou_padrao(kv, "pesos", "")
    var itens = uteis.split_csv_simples(pesos_csv)
    for it in itens:
        if len(it.strip()) > 0:
            pesos_lidos.append(uteis.parse_float_ascii(it))
    var bias_lido: Float32 = uteis.parse_float_ascii(uteis.obter_valor_ou_padrao(kv, "bias", "0"))

    if num_entradas <= 0:
        num_entradas = len(pesos_lidos)
    if num_entradas <= 0:
        num_entradas = 1

    var camada = CamadaLinear(num_entradas, tipo)
    var total_pesos = len(pesos_lidos)
    if len(camada.pesos.dados) < total_pesos:
        total_pesos = len(camada.pesos.dados)
    for i in range(total_pesos):
        camada.pesos.dados[i] = pesos_lidos[i]
    camada.bias.dados[0] = bias_lido
    return camada^