# BIONIX-ML

<p align="center">
	<img src="ICONE.png" alt="Ícone do BIONIX-ML" width="180">
</p>

<p align="center">
	Framework de IA em <strong>Mojo</strong> para leitura de dados multimodais, normalização, conversão para tensor, treinamento, inferência e futura implantação de modelos em uma pilha unificada.
</p>

## Visão geral

O **BIONIX-ML** é um framework de inteligência artificial cujo objetivo é fornecer uma base única para projetos de IA em diferentes níveis de maturidade: **aprendizado, pesquisa e produção comercial**.

A proposta do projeto é entregar uma pilha mais simples, padronizada e centralizada para:

- leitura de **texto**, **imagem** e **áudio**;
- normalização e conversão desses dados para **tensor**;
- construção de blocos arquiteturais de modelos de IA;
- treinamento, análise, inferência e posterior implantação;
- evolução para execução em múltiplos backends computacionais.

O foco é reduzir a curva de aprendizado sem perder flexibilidade, mantendo um ambiente **leve, performático, econômico, customizável e fácil de usar**, inclusive para ensino e experimentação.

## Ambição do framework

A ambição do BIONIX-ML é ser um framework de IA com módulos para:

- ler **texto**;
- ler **imagem**;
- ler **áudio**;
- normalizar dados;
- retornar os dados em formato de **tensor**.

Além disso, a proposta é reunir em uma única pilha os blocos necessários para evolução de projetos de IA em vários níveis, desde o estudo inicial até cenários mais avançados de pesquisa e uso comercial.

Essa ambição também inclui viabilizar uma operação de **baixo custo**, com potencial de **execução local**, melhor aproveitamento de hardware disponível e mecanismos para reduzir consumo de memória, custo de execução e desperdício de recursos computacionais.

## Status do projeto

### Backends pretendidos

- CPU
- Vulkan
- CUDA
- ROCm

### Backend operacional hoje

- **CPU**

Embora a base do projeto já esteja preparada para lidar com múltiplos backends, a execução efetiva atualmente acontece em **CPU**. Os caminhos para **Vulkan**, **CUDA** e **ROCm** ainda estão em evolução.

### Blocos arquiteturais disponíveis hoje

- **Linear**

No momento, o bloco arquitetural funcional implementado é a camada **Linear**, com suporte a inferência, treinamento simples e persistência de pesos.

O objetivo de longo prazo é que o framework contenha diversos **blocos arquiteturais de modelos de IA**, mas atualmente o bloco disponível é apenas `Linear`.

## O que já existe no repositório

O repositório já conta com módulos organizados para cobrir a base do framework:

### Núcleo tensorial

- estrutura de `Tensor`;
- controle de formato, passos e gradiente;
- operações tensoriais básicas;
- abstração inicial de backend computacional.

### Entrada e preparação de dados

- leitura de **CSV**;
- leitura de **BMP**;
- leitura de **WAV**;
- normalização de dados;
- conversão de dados carregados para **tensor**.

### Conjuntos supervisionados

- carregamento de dataset supervisionado a partir de CSV;
- separação entre entradas e alvo;
- normalização das entradas e do alvo;
- preparação dos tensores para treino e inferência.

### Blocos de modelo

- camada **Linear**;
- treinamento básico por erro quadrático médio;
- persistência e recarga de pesos.

### Exemplos

- exemplo de uso do núcleo;
- exemplo de modelo linear com CSV e persistência.

## Objetivo técnico do framework

O objetivo do BIONIX-ML é evoluir para um framework capaz de:

1. receber múltiplos tipos de dados de entrada;
2. padronizar a preparação desses dados;
3. convertê-los em tensores prontos para processamento;
4. disponibilizar blocos reutilizáveis para composição de modelos;
5. oferecer caminhos consistentes para treino, avaliação, inferência e operação.

Em termos práticos, a visão é cobrir o fluxo completo:

**carregamento de dados → análise → normalização → tensorização → modelagem → treinamento → inferência → implantação**

## Roadmap e planejamento

Os itens abaixo ainda estão em planejamento e representam a direção estratégica do projeto:

### Modelagem e treinamento

- redes de múltiplas camadas;
- funções de ativação;
- blocos transformer;
- blocos convolucionais;
- LLMs.

### Voz, fala e multimodalidade

- clonagem de voz;
- texto para fala;
- fala para texto.

### Observabilidade e destilação

- instrumentação de camadas e modelos;
- ferramentas para análise de modelos e camadas;
- modo de execução com captura de entrada e saída por camada;
- suporte a estratégias de destilação.

### Paralelismo e escala

- paralelismo de dados;
- paralelismo de tensor;
- paralelismo de pipeline;
- paralelismo híbrido;
- mecanismos para aumento de vazão e viabilização do treinamento de modelos maiores.

### Resiliência operacional

- offload em memória e disco;
- offload em disco para cenários de restrição de memória e execução local;
- persistência do estado de treinamento;
- persistência de pesos a cada iteração;
- continuidade automática do procedimento em caso de falha.

### Interoperabilidade e inferência

- módulo padrão de API para inferência com payload dinâmico;
- suporte a múltiplas entradas numeradas, por exemplo:
	- `texto[1]`
	- `imagem[1]`
	- `imagem[2]`
	- `audio[1]`
- leitura e exportação de **GGUF**.

### Compressão e eficiência de modelos

- instrumentação para criação de métodos ótimos de compressão ainda no treinamento;
- compressão máxima do modelo já em tempo de treinamento;
- estratégias customizáveis de alocação de tensor;
- redução de consumo de memória e custo de execução;
- liberação de recursos para paralelismo e aumento de vazão das requisições;
- foco em viabilizar treinamento e inferência com melhor aproveitamento de recursos;
- aumento de paralelismo e vazão de modelos.

## TODOs estratégicos do framework

Para refletir de forma objetiva o estado pretendido do projeto, os principais `TODOs` atuais são:

- [ ] expandir de uma arquitetura linear simples para **redes de múltiplas camadas**;
- [ ] adicionar **funções de ativação**;
- [ ] implementar blocos **transformer** e **convolucionais**;
- [ ] evoluir recursos de **clonagem de voz**, **texto para fala** e **fala para texto**;
- [ ] introduzir suporte a **LLMs**;
- [ ] criar mecanismos de **instrumentação de camadas e modelos**;
- [ ] criar ferramentas para **análise de modelos e camadas**;
- [ ] disponibilizar modo de execução com captura de entrada e saída por camada para **destilação**;
- [ ] suportar **paralelismo de dados, tensor, pipeline e híbrido** para aumento de vazão e viabilização do treinamento;
- [ ] implementar **offload em memória e disco**;
- [ ] viabilizar **execução de baixo custo com potencial de execução local**;
- [ ] persistir **estado de treinamento e pesos a cada iteração** para continuidade após falhas;
- [ ] criar um **módulo padrão de API de inferência** com payload dinâmico para múltiplas entradas numeradas, como `texto[1]`, `imagem[1]`, `imagem[2]` e `audio[1]`;
- [ ] adicionar **leitura e exportação de GGUF**;
- [ ] criar um módulo de instrumentação e compressão para obter **métodos ótimos de compressão máxima do modelo ainda durante o treinamento**, com foco em viabilizar treino, inferência e maior paralelismo orientado a vazão;
- [ ] implementar **estratégias customizáveis de alocação de tensor** com objetivo de reduzir consumo de memória, custo de execução e liberar recursos para paralelismo.

## Diferencial do projeto

O principal diferencial do BIONIX-ML é a proposta de **amenizar a curva de aprendizado** e **padronizar o ambiente de trabalho** para IA, sem limitar o projeto a um único estágio de uso.

Isso significa oferecer uma base que atenda:

- quem está aprendendo;
- quem está pesquisando;
- quem precisa construir soluções com aplicação comercial.

Ao consolidar carregamento de dados, preparação, modelagem, treinamento e implantação em uma mesma pilha, o projeto busca reduzir fragmentação, facilitar manutenção e tornar o ambiente mais previsível para evolução de soluções de IA.

Em outras palavras, o diferencial do projeto é permitir acesso em nível de **aprendizado, pesquisa e produção comercial** por meio da **amenização da curva de aprendizado** e da **padronização do ambiente de trabalho**, fornecendo uma pilha completa para projetos de IA em vários níveis: **do carregamento dos dados, análise e treinamento até a implantação do modelo**, mantendo o ambiente centralizado, leve, performático, econômico, customizável e fácil de utilizar.

Dentro dessa visão, também faz parte do diferencial do projeto buscar caminhos para **execução local de menor custo**, **offload inteligente**, **análise e destilação de modelos e camadas**, **compressão de modelos** e **estratégias flexíveis de alocação de tensores**, com o objetivo de reduzir consumo de memória, custo operacional e ampliar a capacidade de paralelismo e vazão.

## Estrutura atual do repositório

```text
src/
	camadas/
		linear/
	computacao/
		cpu/
		cuda/
		rocm/
		vulkan/
	conjuntos/
	dados/
	nucleo/
	uteis/
exemplos/
testes/
```

## Requisitos atuais

- **Pixi**
- **Mojo 0.25.6**
- ambiente **Linux** ou **WSL/Ubuntu**

O arquivo `pixi.toml` do projeto está configurado atualmente para a plataforma `linux-64`.

## Como executar

### Execução no Windows

No estado atual do projeto, o caminho mais indicado no Windows é usar **WSL/Ubuntu**.

Isso acontece porque o ambiente do projeto em `pixi.toml` está configurado atualmente para `linux-64`.

#### Fluxo recomendado no Windows

1. abrir o projeto pelo **WSL/Ubuntu**;
2. garantir que o `pixi` esteja instalado no ambiente Linux;
3. executar os comandos do projeto a partir da sessão Linux.

#### Execução nativa no Windows

**TODO:** formalizar e documentar um fluxo padrão de execução nativa no Windows, caso o projeto passe a suportar esse modo oficialmente.

### Execução no Linux

Em Linux, ou em uma sessão **WSL/Ubuntu**, instale os pré-requisitos e execute o projeto com `pixi` + `mojo`.

Pré-requisitos atuais:

- `pixi`
- `mojo 0.25.6`

Se estiver em WSL e quiser um passo a passo mais direto para coleta de saída, consulte também `README_RUN.md`.

### Executar os exemplos

```bash
pixi run mojo run exemplos/principal.mojo
```

### Executar os exemplos e salvar a saída em arquivo

```bash
pixi run mojo run exemplos/principal.mojo > exemplos/pixi_run.log 2>&1
```

### Visualizar o arquivo de saída

#### Linux / WSL

```bash
cat exemplos/pixi_run.log
```

#### Windows PowerShell

```powershell
Get-Content .\exemplos\pixi_run.log
```

### Executar os testes

```bash
pixi run mojo run testes/principal.mojo
```

### Observação sobre WSL

Se você estiver rodando em Windows, a forma mais segura no estado atual do projeto é utilizar **WSL/Ubuntu**, conforme as orientações em `README_RUN.md`.

## Como importar o BIONIX em um novo projeto

Hoje, dentro deste repositório, os exemplos utilizam imports diretamente a partir da árvore `src`, por exemplo:

- `import src.dados as dados_pkg`
- `import src.conjuntos as conjuntos_pkg`
- `import src.camadas.linear as linear_pkg`
- `import src.computacao as computacao_pkg`
- `import src.nucleo.Tensor as tensor_defs`

Esse padrão funciona para o desenvolvimento interno do próprio repositório.

### Integração padrão em projeto externo

**TODO:** definir e documentar uma forma oficial e estável de consumir o BIONIX em um novo projeto, incluindo ao menos um dos caminhos abaixo:

- distribuição como dependência reutilizável;
- template de projeto com o BIONIX já integrado;
- convenção oficial de importação para módulos externos;
- instruções padronizadas de organização de workspace e paths.

Até essa padronização existir, ainda não há no repositório uma maneira oficialmente documentada para importar o BIONIX como dependência externa em um novo projeto.

## Estado atual em uma frase

Hoje, o BIONIX-ML já oferece uma base funcional para **carregar dados, normalizar, converter para tensor e experimentar modelos lineares em CPU**, enquanto evolui para se tornar uma pilha completa de IA multimodal e multi-backend.
