import src.graficos.bmp as bmp


fn criar_imagem_grayscale(var largura: Int, var altura: Int, var valor_inicial: Int = 0) -> List[Int]:
    return bmp.criar_imagem_grayscale(largura, altura, valor_inicial)


fn redimensionar_matriz_grayscale_nearest(var origem: List[List[Float32]], var altura_alvo: Int, var largura_alvo: Int) -> List[List[Float32]]:
    return bmp.redimensionar_matriz_grayscale_nearest(origem, altura_alvo, largura_alvo)


fn redimensionar_matriz_preto_branco_nearest(var origem: List[List[Float32]], var altura_alvo: Int, var largura_alvo: Int) -> List[List[Float32]]:
    return bmp.redimensionar_matriz_preto_branco_nearest(origem, altura_alvo, largura_alvo)


fn redimensionar_matriz_rgb_nearest(var origem: List[List[List[Float32]]], var altura_alvo: Int, var largura_alvo: Int) -> List[List[List[Float32]]]:
    return bmp.redimensionar_matriz_rgb_nearest(origem, altura_alvo, largura_alvo)


fn desenhar_ponto_grayscale(mut imagem: List[Int], var largura: Int, var altura: Int, var x: Int, var y: Int, var valor: Int):
    bmp.desenhar_ponto_grayscale(imagem, largura, altura, x, y, valor)


fn plotar_espiral_grayscale(mut imagem: List[Int], var largura: Int, var altura: Int, var classe: Int):
    bmp.plotar_espiral_grayscale(imagem, largura, altura, classe)


fn gerar_bmp_24bits_de_grayscale(var imagem: List[Int], var largura: Int, var altura: Int) -> List[Int]:
    return bmp.gerar_bmp_24bits_de_grayscale(imagem, largura, altura)


fn gerar_bmp_espirais_intercaladas_bytes(var largura: Int = 192, var altura: Int = 192) -> List[Int]:
    return bmp.gerar_bmp_espirais_intercaladas_bytes(largura, altura)


fn gerar_bmp_espirais_intercaladas_variacao_bytes(
    var largura: Int = 192,
    var altura: Int = 192,
    var offset_classe0: Int = 0,
    var offset_classe1: Int = 0,
    var delta_intensidade: Int = 0,
) -> List[Int]:
    return bmp.gerar_bmp_espirais_intercaladas_variacao_bytes(
        largura,
        altura,
        offset_classe0,
        offset_classe1,
        delta_intensidade,
    )
