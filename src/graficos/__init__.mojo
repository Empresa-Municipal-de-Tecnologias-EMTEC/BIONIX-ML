import src.graficos.bmp as bmp


fn criar_imagem_grayscale(var largura: Int, var altura: Int, var valor_inicial: Int = 0) -> List[Int]:
    return bmp.criar_imagem_grayscale(largura, altura, valor_inicial)


fn desenhar_ponto_grayscale(mut imagem: List[Int], var largura: Int, var altura: Int, var x: Int, var y: Int, var valor: Int):
    bmp.desenhar_ponto_grayscale(imagem, largura, altura, x, y, valor)


fn plotar_espiral_grayscale(mut imagem: List[Int], var largura: Int, var altura: Int, var classe: Int):
    bmp.plotar_espiral_grayscale(imagem, largura, altura, classe)


fn gerar_bmp_24bits_de_grayscale(var imagem: List[Int], var largura: Int, var altura: Int) -> List[Int]:
    return bmp.gerar_bmp_24bits_de_grayscale(imagem, largura, altura)


fn gerar_bmp_espirais_intercaladas_bytes(var largura: Int = 192, var altura: Int = 192) -> List[Int]:
    return bmp.gerar_bmp_espirais_intercaladas_bytes(largura, altura)
