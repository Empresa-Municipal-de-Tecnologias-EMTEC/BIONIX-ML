def executar_exemplo():
    # Exemplo baseado nos testes do núcleo — ponto de partida para exemplos
    import src.nucleo.nucleo as nucleo

    print("--- Iniciando exemplo do núcleo Bionix ---")

    var formato = List[Int](2)
    formato.append(2)
    formato.append(2)
    var a = nucleo.Tensor(formato^)
    var b = nucleo.Tensor(formato^)
    a.dados[0] = 1.0
    a.dados[1] = 2.0
    a.dados[2] = 3.0
    a.dados[3] = 4.0
    b.dados[0] = 0.5
    b.dados[1] = 1.5
    b.dados[2] = -1.0
    b.dados[3] = 2.0

    var soma = nucleo.somar(a, b)
    print("resultado da soma no exemplo:")
    for i in range(len(soma.dados)):
        print("  ", soma.dados[i])

    print("--- Fim do exemplo ---")
