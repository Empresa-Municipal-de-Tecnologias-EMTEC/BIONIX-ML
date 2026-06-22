# BIONIX-ML (.NET)

<p align="center">
	<img src="ICONE.png" alt="Ícone do BIONIX-ML" width="160">
</p>

Framework de Machine Learning para leitura de dados multimodais, normalização, codificação, treinamento, inferência e implantação de modelos.

Visão geral

O **BIONIX-ML** é um framework de inteligência artificial que fornece uma base para projetos de IA em diferentes níveis: aprendizado, pesquisa e produção. A proposta é entregar uma pilha simples e padronizada para:

- leitura de texto, imagem e áudio
- normalização e conversão para tensores
- blocos reutilizáveis para modelagem (Linear, MLP, CNN, Transformer base)
- treinamento, avaliação e inferência
- exportação/persistência de pesos e artefatos de modelo

Ambição

Ser uma pilha leve, performática e customizável, viabilizando execução local com baixo custo e mecanismos para reduzir consumo de memória e custo de execução.

Status do projeto

- Backends previstos: CPU (operacional), Vulkan, CUDA, ROCm (planejados)
- Blocos disponíveis: Linear, MLP, CNN, blocos básicos de Transformer
- Exemplos: regressão linear, MLP, reconhecimento facial (exemplos/)

O repositório contém módulos para o núcleo tensorial, preparação de dados (CSV/BMP/WAV), conjuntos supervisionados, camadas de modelo, e utilitários para treino e persistência de pesos.

Roadmap (visão resumida)

- Paralelismo (dados/modelo/pipeline)
- Precisão mista e optimizadores plugáveis
- Observabilidade e profiling
- Mecanismos de offload e resiliência operacional

Resumo
- **Propósito**: biblioteca core `Bionix.ML` com tensores, backend CPU, utilitários de imagem e exemplos mínimos.
- **Target .NET**: `net8.0` (projetos retargetados para evitar dependência de runtimes EOL).

Estrutura principal
- `src/Bionix.ML/` — biblioteca principal (Tensor, ComputacaoContexto, utilitários de imagem).
- `src/Examples/Exemplo0001RegressaoLinear/` — exemplo imediato para validar pipeline tensorial.

Início rápido (uso com Vivaz)
- O core `BIONIX-ML` é consumido pelos projetos de nível superior em `BIONIX-ML-VIVAZ` para detecção e identificação facial.
- Se você pretende executar a stack Vivaz, veja o arquivo [BIONIX-ML-VIVAZ/README.md](BIONIX-ML-VIVAZ/README.md) para instruções detalhadas de execução, treinamento e Docker.

Como compilar
1. Restore dependências:

	dotnet restore

2. Build (Release):

	dotnet build -c Release

Executar o exemplo de regressão

dotnet run --project src/Examples/Exemplo0001RegressaoLinear -c Release

Contribuição
Abra um pull request com descrições claras das mudanças e inclua um caso de uso ou teste mínimo para validar o comportamento esperado.
