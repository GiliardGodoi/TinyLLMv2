# Experimentos TinyLLM + LoRA

> Responsável: Giliard Almeida de Godoi
>
> Data da versão: 13 de fevereiro de 2026


## Introdução

O objetio deste experimentos é avaliar a perda de desempenho ao utilizar o procedimento de treinamento descrito por Tian et al. (2025) juntamente com a técnica de redução parâmetros de treinamento Low Rank Adaptation (LoRA) e quantização dos parâmetros do modelo estudante.

A técnica de LoRA congela os pesos originais do modelo estudante e representa a atualização $\Delta W$ pela multiplicação de duas matrizes de menor posto $B_{d \times r} \times A_{r \times k}$ Hu et al. (2021). A quantização do modelo é responsável por carregar os pesos do modelo considerando uma representação numérica com um menor número de bits, que requerem menos memória. Essas duas técnicas possuem esse mesmo objetivo, diminuir os requisitos computacionais necessários para o treinamento de modelos de linguagem.

Contudo, uma vez que a quantização representa uma perda de informação, e o LoRA existe a atualização do modelo é feita de forma indireta, com um menor número de parâmetros, é esperado uma perda de desempenho em comparação à atualização dos parâmetros originais do modelo.

## Leituras relacionadas

[Schulman (2025)](https://thinkingmachines.ai/blog/lora/) traz algumas recomendações sobre a utilização de LoRA para realizar o fine-tuning de modelos de linguagem como, por exemplo, aumentar por um fator de 10 a taxa de aprendizado do experimento original, sem LoRA; aplicar as matrizes do LoRA às camadas densas e não somente às camadas de atenção do modelo; aumentar o parâmetro *rank*.

A intuição por trás dessas recomendações é clara. Se aumentamos o número de parâmetros treináveis do LoRA aumentando o _rank_ (r) ou então o número de camadas onde as matrizes do LoRA são anexadas, possívelmente o modelo vai conseguir incorporar mais informações, aprender melhor os padrões disponíveis. Por outro lado, uma maior quantidade de parâmetros irá requerer uma maior quantidade de memória, mais tempo de treinamento, etc.

## Material e métodos

Esse experimento utiliza o código do TinyLLM adaptado executar atulização com LoRA, e que está disponível no *branch* giliard-dev do repositório [TinyLLMv2](https://github.com/GiliardGodoi/TinyLLMv2/tree/giliard-dev) no Github.

Como modelo, continua-se utilizando o modelo [Flan-T5-large](https://huggingface.co/google/flan-t5-large) como modelo estudante, que possui 780 milhões de parâmetros.

Os conjuntos de dados são aqueles utilizados por Tian et al. (2025):

1. [OpenBookQA Dataset](https://allenai.org/data/open-book-qa)
2. [AI2 Reasoning Challenge (ARC) 2018](https://allenai.org/data/arc)
3. [PIQA (Physical Interaction: Question Answering)](https://yonatanbisk.com/piqa/)
4. [RiddleSense: Reasoning about Riddle Questions Featuring Linguistic Creativity and Commonsense Knowledge](https://inklab.usc.edu/RiddleSense/)
5. [PubMedQA](https://pubmedqa.github.io/)
6. [BioASQ (Biomedical Semantic Indexing and Question Answering)](http://participants-area.bioasq.org/datasets/)

## Experimento

Especificamente, o presente experimento tem por objetivo compara o resultado obtido quando se aplica as matrizes de atualização do LoRA nas camadas lineares do modelo estudante ou não.


O modelo base, Flan-T5-large (780 M), que serve como modelo estudante é carregado considerando uma quantização, isto é, uma representação numérica de ponto flutuante de apenas 4 bits. Essa configuração é realizada no seguinte trecho de código:

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
```

### Experimento variação 001

A variação 001 consiste na aplicação das matrizes de atualização do LoRA também nas camadas lineares, além das camadas específicas do mecanismo de atenção, do modelo Flan-t5

```python
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q", "v", "wi_0", "wi_1", "wo"],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)
```

A definição das camadas lineares é dependente das configurações de cada modelo. O modelo FLAN-T5-large nomeia as camadas lineares desta forma, então os parâmetros passados a `target_modules` deve fazer referência a denominação usada por esse modelo. Outros modelos vão denominar as camadas lineares de forma diferente. Portanto, é preciso estar atento a essa modificação ao se usar outros modelos de linguagem.

### Experimento variação 002

Já a variação 002 consiste na aplicação das matrizes do LoRA nas camadas de *query* (q) e *value* (v) do mecanismo de atenção.

```python
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)
```
Consultar código disponível em [trainer.py](../tiny/trainer.py).

A diferença desse experimento para os experimentos iniciais utilizando essa abordagem, é que a taxa de aprendizado foi multiplicada por um fator de 10. Portanto, foi utilizada a taxa de aprendizado de $5-10^4$, enquanto que nos experimentos iniciais foi adotada a mesma taxa de aprendizagem reportada no artigo de Tian et al. (2025), que é de $5-10^5$. Essa alteração segue as recomendações propostas por Schulman (2025).

## Resultado

Os diretórios contento os resultados gerados dos experimentos e os respectivos _notebooks_ que consolidam os dados e geram as tabelas estão organizados da seguinte maneira.

O diretório `outputs_000.001` refere-se aos dados originados da execução do experimento e o arquivo de _notebook_ `notebooks\000.001-analise-dados.ipynb` à consolidação desses dados, referente a variação 001 do experimento.

De forma análoga, o diretório `outputs_000.002` e o arquivo `notebooks\000.002-analise-dados.ipynb` refere-se ao diretório de dados e ao notebook de consolidação desses dados para a variação 002 do experimento.

A seguir são apresentadas as tabelas com os resultados para cada variação.

### Variação 001

Os resultados de acurácia para a partição de validação foram:

**Tabela 00 - Resultados para a variação 001 do experimento.**
| _timestamp                 | dataset   |    epoch |   batch_size |   grad_steps |     lr |   eval_accuracy |   eval_token_accuracy |   eval_loss |
|:---------------------------|:----------|---------:|-------------:|-------------:|-------:|----------------:|----------------------:|------------:|
| 2025-11-22T13:25:09.677978 | obqa      |  89.7484 |            8 |            8 | 0.005  |        0        |            0.00295385 |   15.2705   |
| 2025-11-23T23:30:32.850101 | obqa      |  89.7484 |            8 |            8 | 0.0005 |      **0.722**  |            0.467545   |    3.23244  |
| 2025-11-25T03:22:56.570681 | arc       | 388.914  |            8 |            8 | 0.0005 |        0.549356 |            0.684053   |    6.0938   |
| 2025-11-26T17:51:12.135111 | piqa      |  27.7782 |            8 |            8 | 0.0005 |    **0.70185**  |            0.0686181  |    0.804071 |
| 2025-11-28T01:18:24.173980 | riddle    | 127.273  |            8 |            8 | 0.0005 |        0.613725 |            0.779412   |    5.04408  |
| 2025-11-29T07:47:43.850149 | bioasq    | 437.525  |            8 |            8 | 0.0005 |    **0.918699** |            0.85       |    4.63522  |
| 2025-12-01T14:07:29.881824 | pubmedqa  | 875      |            8 |            8 | 0.0005 |        0.69     |            0.85       |    7.57872  |


O tempo de treinamento são os seguintes:

**Tabela 00 - Tempo de treinamento reportado pelo _framework_ de treinamento da biblioteca _transformers_. Variação 001.**
| _timestamp                 | dataset   |   train_runtime_hours |
|:---------------------------|:----------|----------------------:|
| 2025-11-22T13:25:09.677978 | obqa      |               29.4344 |
| 2025-11-23T23:30:32.850101 | obqa      |               27.7199 |
| 2025-11-25T03:22:56.570681 | arc       |               38.0836 |
| 2025-11-26T17:51:12.135111 | piqa      |               31.1548 |
| 2025-11-28T01:18:24.173980 | riddle    |               30.2807 |
| 2025-11-29T07:47:43.850149 | bioasq    |               54.2512 |
| 2025-12-01T14:07:29.881824 | pubmedqa  |               60.5691 |


### Variação 002

**Tabela 00 - Resultados para a variação 002 do experimento.**
| _timestamp                 | dataset   |    epoch |   batch_size |   grad_steps |     lr |   eval_accuracy |   eval_token_accuracy |   eval_loss |
|:---------------------------|:----------|---------:|-------------:|-------------:|-------:|----------------:|----------------------:|------------:|
| 2025-12-04T18:39:09.736796 | obqa      |  89.7484 |            8 |            8 | 0.0005 |        0.706    |             0.469545  |    2.78093  |
| 2025-12-05T19:52:12.998636 | arc       | 388.914  |            8 |            8 | 0.0005 |    **0.551073** |             0.683969  |    4.55604  |
| 2025-12-07T05:11:57.959040 | piqa      |  27.7782 |            8 |            8 | 0.0005 |        0.697497 |             0.0675372 |    0.646115 |
| 2025-12-08T09:15:01.531844 | riddle    | 127.273  |            8 |            8 | 0.0005 |    **0.658824** |             0.781471  |    4.10908  |
| 2025-12-09T10:26:03.781083 | bioasq    | 437.525  |            8 |            8 | 0.0005 |        0.894309 |             0.85      |    3.78312  |
| 2025-12-11T09:50:12.990520 | pubmedqa  | 875      |            8 |            8 | 0.0005 |      **0.695**  |             0.85      |    6.90329  |

Para a segunda variação, o tempo de treinamento é o seguinte:

**Tabela 00 - Tempo de treinamento reportado pelo _framework_ de treinamento da biblioteca _transformers_. Variação 002.**
| _timestamp                 | dataset   |   train_runtime_hours |
|:---------------------------|:----------|----------------------:|
| 2025-12-04T18:39:09.736796 | obqa      |               25.0746 |
| 2025-12-05T19:52:12.998636 | arc       |               32.9772 |
| 2025-12-07T05:11:57.959040 | piqa      |               27.7973 |
| 2025-12-08T09:15:01.531844 | riddle    |               25.0346 |
| 2025-12-09T10:26:03.781083 | bioasq    |               47.3543 |
| 2025-12-11T09:50:12.990520 | pubmedqa  |               52.8259 |

## Discussão

Analisando o valor da acurácia (`eval_accuracy`), os resultados não são unânimes sobre qual abordagem seria melhor do que a outra. Somente uma execução de cada variação não é suficiente para produzir dados para decidir estatisticamente que uma abordagem é melhor ou pior do que outra. Além disso, a diferença entre os valores correspondentes da acurácia entre uma variação e outra, que variam de $0.005$ até a $0.03$ em outros casos, não parece justificar a execução de novas rodadas desse experimento.

Com a adição dos parâmetros LoRA nas camadas lineares do modelo, o treinamento torna-se um pouco mais lento. Isso pode ser constatado pelas tabelas B1 e B2, que reportam o tempo, em horas, para o treinamento de cada variação dos experimentos. Comparando os tempos de treinamento entre os conjuntos de dados correspondentes de cada variação, uma boa aproximação é dizer que o treinamento aumentou em 5 horas para a variação 001, que adiciona as matrizes do LoRA para as camadas lineares do modelo.

## Conclusão

Se os resultados não são conclusivos sobre a utilização ou não de todas as recomendações propostas por Schulman (2025), pelo menos algumas recomendações para os próximos experimentos podem ser feitas.

1. Utilizar uma taxa de aprendizado mais agressiva. Enquanto o Tian et al. (2025) utiliza uma taxa de aprendizado de $5-10^5$ para o treinamento completo dos modelos, com a utilização do LoRA podemos iniciar com uma taxa de aprendizado de pelo menos $5-10^4$. Taxas de aprendizados maiores não apresentaram bons resultados para a convergência do modelo (conforme outros experimentos executados (?)).

Alterar a taxa de aprendizagem inicialmente não altera os recursos necessários para carregar o modelo ou treinar o modelo, então pode ser um bom início já começar com uma taxa de aprendizagem mais agressiva.

2. Alterar o parâmetro de rank ($r$) do LoRA ou as camadas onde suas matrizes são aplicadas, por outro lado, alteram a quantidade de parâmetros e, por conseguinte, os recursos computacionais necessários para realizar o treinamento. Considerando que as alterações nos resultados pode não ser tão significativa, pode-se iniciar os experimentos com os valores padrões para esses parâmetros e, se assim for necessário, e os recursos computacionais permitirem, ir testando novos valores e acompanhando a alteração dos recursos necessários.


## Referências

1. Hu et al. (2021) *LoRA: Low-Rank Adaptation of Large Language Models*. Disponível em <https://arxiv.org/abs/2106.09685>. Último acesso em 13 de fev. de 2026.
2. Schulman, J. (2025). *LoRA without regret*. Disponível em <https://thinkingmachines.ai/blog/lora/>. Último acesso em 13 de fev. de 2026.
3. Google. (2023). *Flan-T5-large*. Disponível em <https://huggingface.co/google/flan-t5-large>. Último acesso em 13 de fev. de 2026.