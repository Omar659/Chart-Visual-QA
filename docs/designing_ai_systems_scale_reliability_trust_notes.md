# Designing AI Systems for Scale, Reliability, and Trust

Compilado das principais ideias da aula, organizado em três pilares: **escala**, **confiabilidade** e **trust/governança**.  
O foco deste resumo é destacar o que realmente importa para sistemas de IA em produção, especialmente em **LLM serving**, **arquiteturas distribuídas**, **segurança**, **prompt injection**, **PII**, **RAG** e **model governance**.

---

## Visão geral da aula

A principal mensagem da aula é que um sistema de IA em produção não é apenas um modelo. É uma arquitetura completa composta por:

- infraestrutura;
- serving;
- roteamento;
- fallback;
- observabilidade;
- segurança;
- avaliação contínua;
- governança;
- compliance;
- controle de dados;
- controle de versões de modelos.

Uma frase central:

> A production AI system is not just a model. It is a model embedded in a reliable, observable, scalable, and governable system.

Em português:

> Um sistema de IA em produção não é só um modelo. É um modelo dentro de uma arquitetura confiável, observável, escalável e governável.

---

# Pillar 1 — Architecting for Massive Scale

O primeiro pilar é sobre como arquitetar sistemas de IA para suportar alto volume de usuários, alto throughput, baixa latência, custo controlado e uso eficiente de hardware.

A ideia central:

> At massive scale, architecture matters as much as model quality.

Ou seja, um modelo bom não basta. Se a arquitetura não escala, o sistema falha em produção.

---

## 1. Geographic dispersion

**Geographic dispersion** significa distribuir o sistema por múltiplas regiões geográficas ou data centers.

Exemplo:

```text
Users in Europe  -> EU region
Users in US      -> US region
Users in Asia    -> Asia region
```

Isso reduz latência e aumenta resiliência.

### Por que isso importa

Sistemas globais precisam lidar com:

- usuários em diferentes regiões;
- latência de rede;
- falhas regionais;
- restrições de compliance;
- disaster recovery;
- balanceamento global de carga.

### Aplicação em produção

Em produção, isso normalmente envolve:

- multi-region deployment;
- global load balancer;
- health-aware routing;
- failover entre regiões;
- replicação de bancos, caches e índices vetoriais;
- observabilidade distribuída.

Frase boa:

> Geographic dispersion improves latency and resilience, but it requires dynamic orchestration to manage distributed AI workloads efficiently.

---

## 2. Dynamic orchestration

Ao distribuir o sistema por várias regiões e diferentes tipos de hardware, é necessário ter uma camada de orquestração dinâmica.

Ferramentas como **Kubernetes** são importantes porque ajudam a gerenciar:

- containers;
- autoscaling;
- health checks;
- rolling updates;
- service discovery;
- load balancing;
- failover;
- GPU workloads;
- model replicas.

A orquestração não serve apenas para “subir containers”. Em sistemas de IA em escala, ela precisa gerenciar tráfego, capacidade e falhas.

### Traffic shifting

O orquestrador precisa conseguir alterar os caminhos de tráfego dinamicamente.

Exemplo:

```text
User request
   ↓
Global load balancer
   ↓
Traffic router / orchestrator
   ↓
Region A model service overloaded
   ↓
Shift traffic to Region B / Region C
   ↓
Autoscale more replicas
```

### Aplicação em produção

O orquestrador precisa monitorar:

| Métrica | Por que importa |
|---|---|
| Request rate | Detecta picos de tráfego |
| Queue length | Mostra backlog |
| GPU utilization | Mede pressão computacional |
| Memory usage | Importante para model serving |
| KV cache usage | Limita concorrência em LLMs |
| Token throughput | Mede capacidade real de serving |
| TTFT | Tempo até o primeiro token |
| TPOT | Tempo por token gerado |
| Error rate | Detecta falhas ou sobrecarga |

Frase boa:

> At scale, orchestration is not static deployment; it is continuous traffic and capacity management.

---

## 3. Hardware landscape

Sistemas de IA em larga escala precisam lidar com hardware heterogêneo.

| Hardware | Melhor uso | Observações |
|---|---|---|
| High-end discrete GPUs | Treinamento, fine-tuning, LLM inference, multimodal | Alto desempenho, alto custo |
| Specialized TPUs | Operações tensorais em cloud, treinamento/inferência em escala | Muito eficientes para workloads específicos |
| Edge NPUs and SoCs | Inferência local, mobile, dispositivos embarcados | Baixo consumo, baixa latência, privacidade |
| Enterprise CPUs | Pré-processamento, modelos pequenos, fallback, workloads clássicos | Flexíveis, mas menos eficientes para LLMs grandes |

### Aplicação em produção

A arquitetura deve decidir onde rodar cada workload:

```text
Large model inference -> GPU/TPU
Small classifier      -> CPU/NPU
Edge inference        -> SoC/NPU
Fallback logic        -> CPU/local model
```

Frase boa:

> At massive scale, AI deployment is heterogeneous: workloads must be orchestrated across GPUs, TPUs, edge accelerators, SoCs, and CPUs depending on latency, cost, availability, and model requirements.

---

## 4. Model serving parallelism

Para servir modelos grandes, muitas vezes é necessário paralelizar o workload.

Os três principais tipos discutidos foram:

| Tipo | O que divide? | Ideia principal | Melhor uso |
|---|---|---|---|
| Tensor parallelism | Operações/tensores dentro de camadas | Divide a matemática de uma camada entre GPUs | Camadas muito grandes |
| Pipeline parallelism | Camadas do modelo | Coloca blocos do modelo em GPUs diferentes | Modelos profundos que não cabem em uma GPU |
| Data parallelism | Dados/requisições | Replica o modelo inteiro e distribui requests | Aumentar throughput |

### Tensor parallelism

Divide a computação interna de uma camada entre dispositivos.

```text
Layer computation
   ↓
GPU 1 handles part of matrix
GPU 2 handles another part
GPU 3 handles another part
   ↓
Results are combined
```

### Pipeline parallelism

Divide o modelo por blocos ou camadas.

```text
Input
 ↓
GPU 1: Layers 1-10
 ↓
GPU 2: Layers 11-20
 ↓
GPU 3: Layers 21-30
 ↓
Output
```

### Data parallelism

Replica o modelo inteiro em múltiplas GPUs e divide as requisições.

```text
Request batch
   ↓
Replica 1: full model on GPU 1
Replica 2: full model on GPU 2
Replica 3: full model on GPU 3
```

Frase boa:

> Tensor parallelism splits the math, pipeline parallelism splits the layers, and data parallelism splits the requests.

---

## 5. The LLM inference pipeline

A inferência em LLMs normalmente tem duas fases principais:

```text
User request
   ↓
Tokenization
   ↓
Prefill phase
   ↓
Decode phase
   ↓
Streaming response
```

---

### Prefill

O **prefill** processa o prompt inteiro e constrói o **KV cache**.

É uma fase mais **compute-bound**, porque o modelo processa muitos tokens de entrada em paralelo.

Métrica importante:

> TTFT — Time To First Token

TTFT é afetado por:

- fila;
- tamanho do prompt;
- prefill;
- GPU load;
- batching;
- contexto longo.

---

### Decode

O **decode** gera a resposta token por token.

```text
Token 1 -> Token 2 -> Token 3 -> Token 4 -> ...
```

É frequentemente mais **memory-bound**, porque a cada token o modelo precisa acessar pesos e KV cache.

Métrica importante:

> TPOT — Time Per Output Token

---

## 6. Continuous batching

Um serving ingênuo processa requisições separadamente:

```text
Request A -> run alone
Request B -> wait
Request C -> wait
```

Isso desperdiça GPU.

**Continuous batching** resolve isso adicionando e removendo requisições dinamicamente de um batch em execução.

```text
Time step 1: A, B, C
Time step 2: A, B, C, D joins
Time step 3: A finishes, B, C, D continue
Time step 4: E joins, B, C, D, E continue
```

### Aplicação em produção

Continuous batching melhora:

- throughput;
- GPU utilization;
- custo por request;
- handling de picos;
- eficiência de serving.

Frase boa:

> Continuous batching keeps the GPU saturated by mixing active requests and inserting new ones as others finish.

---

## 7. Chunked prefill

**Chunked prefill** divide prompts longos em pedaços menores.

Sem chunked prefill:

```text
Long prompt -> huge prefill block -> decode waits
```

Com chunked prefill:

```text
Decode tokens for active users
Prefill chunk for new long request
Decode tokens again
Prefill next chunk
```

### Por que importa

Prompts muito longos podem monopolizar a GPU e aumentar latência de outros usuários.  
Chunked prefill permite intercalar prefill e decode de forma mais justa.

Frase boa:

> Continuous batching keeps the GPU busy; chunked prefill prevents long prompts from blocking everyone else.

---

## 8. KV cache memory

O **KV cache** armazena os tensores de **Key** e **Value** da atenção para tokens anteriores.

Sem KV cache, o modelo teria que recomputar o contexto inteiro a cada token gerado.

```text
Token 1 generated
   ↓
Store K/V for token 1

Token 2 generated
   ↓
Reuse K/V from token 1
Store K/V for token 2
```

### Por que o KV cache é crítico

O KV cache acelera o decode, mas consome muita memória de GPU.

O uso cresce com:

- contexto longo;
- batch size;
- número de usuários concorrentes;
- número de camadas;
- número de attention heads;
- quantidade de tokens gerados.

Em muitos sistemas de LLM serving, o gargalo não é só compute. É memória.

Frase importante:

> The model weights may fit in GPU memory, but the KV cache can still limit how many requests can be served at the same time.

---

## 9. Paged Attention

**Paged Attention** gerencia o KV cache em pequenos blocos ou páginas, semelhante à memória virtual em sistemas operacionais.

Sem Paged Attention, o sistema pode precisar de grandes blocos contínuos de memória, causando fragmentação.

Com Paged Attention:

```text
Request A KV cache:
Page 1 -> Page 7 -> Page 9

Request B KV cache:
Page 2 -> Page 3 -> Page 8 -> Page 12
```

Isso permite alocação não-contígua e melhor reaproveitamento de memória.

### Aplicação em produção

Paged Attention ajuda a:

- reduzir fragmentação;
- aumentar concorrência;
- melhorar uso da GPU;
- reduzir requests bloqueados por falta de memória contígua;
- manter GPU mais ocupada.

Frase boa:

> Paged Attention is like virtual memory for the KV cache: it reduces fragmentation so GPU memory is used more efficiently.

---

## 10. Quantization benchmark

Quantization benchmarks comparam diferentes formatos de precisão contra uma baseline.

| Formato | Papel | Trade-off |
|---|---|---|
| Native FP16 | Baseline | Alta qualidade, maior memória |
| INT8 weight-only | Compressão moderada | Menos memória, pouca perda geralmente |
| FP8 mixed precision | Moderno e eficiente em hardware suportado | Bom equilíbrio entre throughput e qualidade |
| INT4 highly compressed | Compressão agressiva | Grande economia de memória, maior risco de degradação |

### FP16 baseline

FP16 é usado como baseline porque já é otimizado para GPU e preserva bem qualidade.

### INT8 weight-only

Quantiza pesos para 8 bits, mantendo outras partes em maior precisão.

Bom para:

- reduzir footprint;
- reduzir bandwidth;
- manter qualidade relativamente estável.

### FP8 mixed precision

Usa FP8 em partes seguras da computação e mantém operações sensíveis em FP16/BF16.

Bom para:

- throughput;
- hardware moderno;
- inferência eficiente.

### INT4

Muito comprimido. Pode reduzir memória drasticamente, mas tem maior risco de perda de qualidade.

Frase boa:

> Quantization is not just about making the model smaller; it is a trade-off between memory, latency, throughput, hardware support, and quality.

---

## 11. Core cloud vs intelligent edge

Uma arquitetura escalável pode dividir workloads entre **core cloud** e **intelligent edge**.

---

### Core cloud operations

A cloud central lida com workloads pesados:

- massive fine-tuning pipelines;
- large model training;
- large-scale inference;
- batch processing;
- model registry;
- evaluation;
- governance;
- fallback com modelos grandes;
- central observability.

---

### Intelligent edge operations

A edge roda modelos altamente comprimidos, próximos ao usuário ou dispositivo.

Exemplos:

- classificação local;
- intent detection;
- preprocessing;
- safety filtering local;
- inferência offline;
- anomaly detection;
- modelos quantizados/distilados.

### Edge-to-cloud fallback

Arquitetura:

```text
User / Device
   ↓
Compressed edge model
   ↓
Confidence / quality check
   ↓
High confidence -> respond locally
Low confidence / failure -> route to core cloud
```

Benefícios:

- menor latência;
- menor custo cloud;
- melhor privacidade;
- menor uso de bandwidth;
- maior resiliência.

Frase boa:

> The edge handles the common case; the cloud handles the hard case.

---

# Pillar 2 — Engineering for Reliability

O segundo pilar é sobre fazer o sistema de IA se comportar como software de produção: resiliente, observável, previsível e recuperável.

A ideia central:

> Reliable AI systems are designed around the assumption that models and upstream services will sometimes fail.

---

## 1. Reliability challenges

### Erratic token latency

Latência de LLMs pode ser instável devido a:

- queueing;
- prefill;
- decode token a token;
- long context;
- GPU saturation;
- batching;
- KV cache pressure;
- network delay;
- provider load.

Métricas principais:

| Métrica | Significado |
|---|---|
| TTFT | Time to first token |
| TPOT | Time per output token |
| End-to-end latency | Tempo total de resposta |
| Token throughput | Tokens por segundo |

---

### Upstream outages

Sistemas de IA frequentemente dependem de serviços externos:

- LLM APIs;
- embedding APIs;
- vector databases;
- search APIs;
- cloud storage;
- auth systems;
- tool APIs.

Falhas comuns:

| Falha | Impacto |
|---|---|
| Rate limit | Requests rejeitados ou atrasados |
| Timeout | App fica esperando demais |
| Partial outage | Falhas intermitentes |
| Complete outage | Feature central para de funcionar |
| Slow degradation | Latência cresce antes da falha total |

Frase boa:

> Upstream outages can break downstream AI applications unless fallbacks and degradation strategies are designed in advance.

---

### Structured output drift

Structured output drift acontece quando o modelo deixa de seguir o formato esperado.

Exemplo esperado:

```json
{
  "answer": "...",
  "confidence": 0.87,
  "category": "chart_question"
}
```

Exemplo problemático:

```text
Sure! Here is the JSON:

answer: yes
confidence is probably high
category: chart question
```

Isso quebra parsers e pode causar crashes.

---

### Non-deterministic malformed output

LLMs podem emitir:

- JSON inválido;
- campos faltando;
- campos extras;
- tipos incorretos;
- datas malformadas;
- enum values inexistentes;
- texto misturado com JSON.

Erro de arquitetura:

> Assuming the model will always follow the format.

Postura correta:

> The model may produce invalid output, so we must validate, repair, retry, or fail gracefully.

---

## 2. Adaptive circuit breakers

Um **circuit breaker** impede que o sistema continue enviando tráfego para uma dependência que está falhando.

Em IA, isso pode ser:

- model endpoint;
- embedding service;
- vector database;
- reranker;
- moderation model;
- tool API.

Fluxo:

```text
Request
  ↓
Model endpoint unhealthy?
  ↓
Yes -> circuit breaker opens
  ↓
Stop routing traffic there
  ↓
Use fallback path
```

### Sinais monitorados

| Sinal | Indica |
|---|---|
| Error rate | Endpoint falhando |
| Timeout rate | Endpoint lento |
| Rate-limit responses | Provider rejeitando tráfego |
| Latency spike | Sobrecarga |
| Malformed output rate | Modelo retornando output inválido |
| Queue depth | Congestionamento |

Frase boa:

> Adaptive circuit breakers intercept failing model endpoints before they cascade into total application failure.

---

## 3. Graceful fallback chain

A fallback chain evita que o sistema dependa de um único caminho.

Exemplo:

```text
Primary large cloud model
  ↓ if unavailable / slow / expensive
Secondary cloud model
  ↓ if unavailable
Local smaller model
  ↓ if insufficient
Rule-based response / cached response / safe degraded response
```

A ideia não é manter a mesma qualidade sempre. É manter o sistema vivo com funcionalidade reduzida.

Frase boa:

> Graceful degradation is better than catastrophic failure.

---

## 4. Hierarchical model routing

Nunca depender de um único ponto de roteamento/modelo.

Arquitetura:

```text
User request
  ↓
Input validation
  ↓
Task classifier / complexity estimator
  ↓
Cheap local model for simple tasks
  ↓
Medium model for normal tasks
  ↓
Heavy model for complex tasks
  ↓
Fallback chain if any layer fails
```

### Aplicação em produção

Camadas possíveis:

| Camada | Papel |
|---|---|
| Rules/cache | Casos óbvios ou repetidos |
| Small local model | Classificação, segurança, extração simples |
| Medium model | Requests normais |
| Heavy model | Raciocínio complexo, contexto longo |
| Human review/safe response | Casos críticos ou não resolvidos |

Frase boa:

> Reliable AI systems avoid single points of model failure.

---

## 5. Schema enforcement engines

O objetivo é evitar que o modelo quebre o contrato de saída.

Níveis de enforcement:

---

### Level 1 — Parse, validate, retry

```text
LLM output
  ↓
JSON parser
  ↓
Schema validator
  ↓
Valid? -> continue
Invalid? -> retry / repair / fallback
```

Bom, mas reativo.

---

### Level 2 — Function calling / JSON mode

Define-se uma estrutura esperada, e o modelo é orientado a preencher argumentos.

Mais forte que prompt livre, mas ainda deve haver validação posterior.

---

### Level 3 — Constrained decoding / logit masking

Aqui entra a pergunta importante feita na aula.

A ideia:

> Instead of only asking the model to output valid JSON, constrain the decoding process so invalid next tokens are masked.

Fluxo:

```text
1. Model outputs logits for all possible next tokens.
2. A schema parser determines which tokens are valid next.
3. Invalid tokens are masked, usually set to near-zero probability.
4. Decoder chooses only from valid tokens.
5. Parser state updates.
6. Repeat until output is complete.
```

Exemplo:

Se o modelo já gerou:

```json
{ "allowed":
```

E o schema exige boolean, os próximos tokens válidos são:

```text
true
false
```

Tokens bloqueados:

```text
"yes"
"probably"
42
null
```

Isso é muito mais forte que prompt engineering.

Frase boa:

> Schema enforcement can guarantee format, but not necessarily truth.

Ou seja:

| Problema | Solução |
|---|---|
| JSON inválido | Constrained decoding / schema validation |
| Categoria errada | Melhor modelo, evals, calibração |
| Campo faltando | Schema validation |
| Alucinação | Grounding checks / evals |
| Decisão insegura | Guardrails / policy validation |

---

# Pillar 3 — Establishing Systemic Trust

O terceiro pilar trata de confiança sistêmica: transparência, governança, controle de dados, compliance, segurança, auditabilidade e controle sobre o modelo.

A ideia central:

> Trust is not only about whether the model gives good answers. It is about whether the whole AI system is transparent, controlled, monitored, and accountable.

---

## 1. Risk of black-box endpoints

Um endpoint black-box é um serviço externo de IA cujo modelo interno não é totalmente controlado ou auditável pela sua organização.

```text
Our application
   ↓
External LLM API endpoint
   ↓
Unknown model internals / unknown updates / unknown data handling
```

Riscos:

- arquitetura do modelo desconhecida;
- dados de treinamento desconhecidos;
- política de update desconhecida;
- comportamento de safety pode mudar;
- retenção de dados pode não ser clara;
- regressões podem aparecer sem aviso.

Frase importante:

> A stable API is not the same as a stable model.

---

## 2. Unannounced provider drift

Provider drift acontece quando o provedor atualiza o modelo, safety layer, decoding behavior ou routing interno sem sua aplicação mudar.

```text
Same API endpoint
Same application code
Same prompt
   ↓
Provider updates model
   ↓
Different behavior
   ↓
Regression in our app
```

Possíveis efeitos:

- menor accuracy;
- mudança de tom;
- mudanças em refusals;
- structured output quebrado;
- latência diferente;
- custo diferente;
- piora em RAG;
- regressão em workflows antes estáveis.

Frase boa:

> The API contract may remain stable while the model behavior changes.

---

## 3. Sensitive data sent to external providers

Risco: enviar dados sensíveis para endpoints externos sem visibilidade completa de uso.

Dados sensíveis incluem:

- PII;
- dados financeiros;
- dados médicos;
- documentos internos;
- código-fonte;
- credenciais;
- informações estratégicas;
- dados de clientes.

Perguntas necessárias:

- O dado é retido?
- É usado para treinamento?
- É logado?
- Onde é processado?
- Quem tem acesso?
- Há criptografia?
- É possível auditar?
- Há compliance com GDPR ou outras normas?

---

## 4. Sovereign open innovation

A resposta proposta é aumentar controle com **open-weight models** e infraestrutura privada.

Ideia:

> Trust can be increased by using open-weight models that are accessible, auditable, and deployable inside private infrastructure.

Com open weights, a organização pode:

- fixar versões;
- auditar comportamento;
- rodar em infraestrutura privada;
- controlar updates;
- evitar provider drift inesperado;
- proteger dados sensíveis;
- fazer fine-tuning interno;
- manter logs e avaliações próprias.

Frase boa:

> Open weights turn the model from an opaque external dependency into a controllable, auditable system component.

---

## 5. System-level guardrails

Trust não vem só do modelo. Vem das camadas ao redor do modelo.

Guardrails devem existir em:

```text
Input -> Retrieval -> Generation -> Output -> Monitoring
```

Frase boa:

> In trustworthy AI systems, every boundary is controlled: input, retrieval, generation, and output.

---

## 6. Input boundary filtering

Antes do input chegar ao modelo principal, o sistema deve fazer sanity checks rápidos.

Detectar:

- prompt injection;
- jailbreak;
- PII;
- sensitive data leakage;
- unsafe content;
- out-of-scope requests;
- malicious instructions escondidas em documentos;
- policy violations.

Arquitetura:

```text
User input
  ↓
Input boundary filter
  ↓
Prompt injection check
  ↓
PII / sensitive data check
  ↓
Allowed? -> continue
Blocked? -> refuse / redact / escalate
```

### Llama Guard e modelos similares

Modelos como **Llama Guard** podem ser usados como guard models para classificar input/output contra políticas de segurança.

Uso em produção:

- camada rápida antes do LLM principal;
- classificação de risco;
- bloqueio de prompt injection;
- detecção de PII;
- output safety;
- monitoramento contínuo.

---

## 7. Prompt injection security

Prompt injection é quando o usuário ou um documento tenta fazer o modelo ignorar instruções do sistema.

Exemplos de risco:

- “Ignore previous instructions”
- documentos RAG contendo instruções maliciosas;
- tentativa de extrair secrets;
- tentativa de mudar policy;
- tentativa de burlar guardrails;
- instruções escondidas em texto recuperado.

Mitigações:

- input boundary filtering;
- separação clara entre system instructions, user input e retrieved context;
- não confiar cegamente em documentos recuperados;
- detectar instruções dentro de documentos;
- usar allowlists de ferramentas;
- restringir tool access;
- validar ações antes de executar;
- output validation;
- human-in-the-loop para ações críticas.

Frase boa:

> Retrieved documents are data, not instructions.

---

## 8. PII and sensitive data protection

PII deve ser detectada, mascarada, minimizada ou bloqueada dependendo do caso.

Estratégias:

- PII detection antes do LLM;
- redaction;
- tokenization;
- pseudonymization;
- minimização de dados;
- não enviar dados sensíveis a provedores externos sem base legal;
- logs sem dados sensíveis;
- controle de retenção;
- criptografia;
- acesso com RBAC;
- auditoria.

Checklist prático:

```text
Before sending data to model:
- Does it contain PII?
- Is the data necessary?
- Can it be redacted?
- Is the provider allowed to process it?
- Is it logged?
- Is it retained?
- Can the user request deletion?
```

---

## 9. Retrieval security vectors

Em RAG, o vector database pode virar canal de vazamento se não houver controle de acesso.

Erro perigoso:

```text
User asks question
  ↓
Vector DB searches all documents
  ↓
LLM receives unauthorized context
  ↓
Data leak
```

Arquitetura segura:

```text
User asks question
  ↓
Auth context / user role
  ↓
Vector DB query with RBAC / ACL filters
  ↓
Only authorized documents retrieved
  ↓
LLM answers from allowed context
```

### RBAC para vector DB

O retrieval layer deve filtrar por:

- user ID;
- role;
- tenant;
- organization;
- project;
- document-level ACL;
- data classification;
- region/compliance constraints.

Frase boa:

> RAG security requires access control at retrieval time, not only at display time.

---

## 10. Active-active resilient clusters

Active-active significa múltiplos clusters ativos ao mesmo tempo.

```text
Region A cluster: active
Region B cluster: active
Region C cluster: active
```

Se uma região falha:

```text
EU cluster fails
   ↓
Global load balancer shifts traffic
   ↓
US / Asia clusters continue serving
```

### Aplicação em produção

Para sistemas com RAG, cada região deve ter seu próprio índice vetorial local.

```text
EU cluster -> local vector DB/index
US cluster -> local vector DB/index
Asia cluster -> local vector DB/index
```

E esses índices devem ser sincronizados continuamente:

```text
Document update
   ↓
Embedding pipeline
   ↓
Vector index update
   ↓
Replicate to EU / US / Asia indexes
```

### Consistência

| Estratégia | Significado | Trade-off |
|---|---|---|
| Strong consistency | Todas as regiões sempre iguais | Mais difícil, mais lento |
| Eventual consistency | Regiões sincronizam continuamente, mas podem diferir por pouco tempo | Mais escalável |

Para muitos sistemas RAG, eventual consistency é aceitável se houver monitoramento de staleness.

Frase boa:

> Active-active design means every region is ready to serve traffic, and every critical dependency, including the vector index, must be replicated or synchronized.

---

## 11. Real-time evaluation loops

Sistemas de IA em produção não devem esperar o usuário reportar erro.

É possível rodar checks em paralelo antes de retornar a resposta.

Checks possíveis:

- hallucination detection;
- grounding check;
- semantic drift detection;
- safety check;
- schema validation;
- policy check;
- contradiction detection;
- confidence thresholding.

Arquitetura:

```text
LLM response candidate
  ↓
Parallel evaluation checks
  ├─ factuality check
  ├─ safety check
  ├─ schema validation
  ├─ grounding check against retrieved docs
  └─ semantic drift check
  ↓
Pass? -> send to user
Fail? -> repair / retry / fallback / block
```

Para RAG:

```text
Generated answer
  ↓
Check whether answer is supported by retrieved context
  ↓
If unsupported -> flag hallucination
```

Frase boa:

> Real-time evaluation loops turn trust from post-hoc analysis into runtime protection.

---

## 12. Regulatory and compliance frameworks

Trust também envolve compliance.

Frameworks citados:

- EU AI Act alignment;
- GDPR for data privacy;
- sovereign audits.

---

### EU AI Act alignment

O EU AI Act segue uma abordagem baseada em risco.

Pontos importantes:

- classificação do sistema por risco;
- documentação;
- risk management;
- human oversight;
- monitoring;
- incident reporting;
- traceability;
- accountability.

### GDPR

GDPR trata de proteção de dados pessoais.

Pontos importantes:

- base legal para processamento;
- minimização de dados;
- purpose limitation;
- storage limitation;
- transparência;
- direitos do usuário;
- deleção;
- controle de acesso;
- restrições de transferência internacional;
- proteção de dados sensíveis.

### Sovereign audits

Sovereign audits significam capacidade de auditar o sistema internamente e provar controle sobre:

- modelos;
- versões;
- prompts;
- datasets;
- logs;
- retrieval indexes;
- access controls;
- safety filters;
- evaluation results;
- deployment environment;
- incident history.

Frase boa:

> Trustworthy AI is not only technically reliable; it must also be legally compliant, auditable, and governable.

---

# Core lessons from the class

## 1. Standardize early

Padronizar cedo evita caos operacional.

Sistemas de IA devem criar padrões para:

- inference pipelines;
- routing;
- batching;
- observability;
- model deployment;
- rollback;
- schema validation;
- guardrails;
- evaluation;
- fallback behavior.

Assim como Kubernetes padronizou deployment cloud-native, pipelines de inferência precisam ser padronizados para AI systems.

Frase boa:

> Standardization reduces operational chaos before the system reaches scale.

---

## 2. Design defensively

Não assumir que o modelo, provider ou output vão funcionar sempre.

Deployar cedo:

- semantic caches;
- adaptive circuit breakers;
- fallback models;
- schema validation;
- guardrails;
- monitoring;
- real-time evals;
- rate limiting;
- retries with backoff;
- degraded mode.

Frase boa:

> Defensive design turns model failure into controlled degradation instead of application failure.

---

## 3. Own the weights

Controlar pesos, modelos e deployment aumenta trust.

Priorizar:

- open-weight models;
- auditable architectures;
- private infrastructure;
- reproducible model versions;
- internal evaluation;
- transparent methodologies;
- controlled updates.

Frase boa:

> Owning the weights means owning the risk surface.

---

# Production checklist

## LLM serving checklist

- [ ] Medir TTFT, TPOT, throughput e end-to-end latency.
- [ ] Usar continuous batching para melhorar GPU utilization.
- [ ] Usar chunked prefill para evitar que prompts longos bloqueiem decode.
- [ ] Monitorar KV cache usage.
- [ ] Usar Paged Attention ou mecanismos similares para reduzir fragmentação.
- [ ] Avaliar tensor, pipeline e data parallelism conforme o tamanho do modelo.
- [ ] Fazer benchmark de FP16, INT8, FP8 e INT4.
- [ ] Usar autoscaling baseado em métricas reais de serving.
- [ ] Ter fallback entre modelos grandes, médios e locais.
- [ ] Usar semantic cache para reduzir custo e latência.

---

## Reliability checklist

- [ ] Implementar timeouts.
- [ ] Implementar retries com backoff.
- [ ] Implementar adaptive circuit breakers.
- [ ] Ter fallback chain.
- [ ] Não depender de um único provider ou endpoint.
- [ ] Validar structured outputs.
- [ ] Usar constrained decoding quando formato for crítico.
- [ ] Ter modo degradado.
- [ ] Monitorar malformed output rate.
- [ ] Monitorar provider outages e rate limits.
- [ ] Fazer regression tests com prompts fixos.

---

## Security and trust checklist

- [ ] Filtrar input antes do modelo principal.
- [ ] Detectar prompt injection.
- [ ] Detectar e mascarar PII.
- [ ] Não tratar documentos recuperados como instruções.
- [ ] Aplicar RBAC/ACL no vector DB.
- [ ] Garantir tenant isolation em RAG.
- [ ] Validar output antes de mostrar ao usuário.
- [ ] Rodar checks de hallucination e grounding.
- [ ] Registrar logs auditáveis sem vazar dados sensíveis.
- [ ] Controlar retenção de dados.
- [ ] Monitorar drift semântico.
- [ ] Preferir open weights quando auditabilidade e soberania forem críticas.

---

## Governance checklist

- [ ] Classificar o sistema conforme risco.
- [ ] Documentar modelo, versões, prompts e datasets.
- [ ] Registrar avaliações.
- [ ] Ter model registry.
- [ ] Ter prompt registry/versioning.
- [ ] Ter data lineage.
- [ ] Ter incident reporting.
- [ ] Ter human oversight quando necessário.
- [ ] Avaliar GDPR para dados pessoais.
- [ ] Avaliar EU AI Act para contexto europeu.
- [ ] Garantir auditoria de acesso e uso.

---

# Final synthesis

Os três pilares da aula podem ser resumidos assim:

| Pilar | Pergunta central | Resposta arquitetural |
|---|---|---|
| Scale | Como servir IA para muitos usuários com baixa latência e custo controlado? | Orquestração, batching, parallelism, quantization, edge/cloud split |
| Reliability | Como impedir que falhas de modelo ou provider quebrem a aplicação? | Circuit breakers, fallback chains, schema enforcement, monitoring |
| Trust | Como garantir controle, segurança, privacidade e auditabilidade? | Guardrails, RBAC, PII filtering, open weights, audits, compliance |

Frase final:

> Scalable and trustworthy AI systems are standardized, defensive, observable, and auditable by design.

Em português:

> Sistemas de IA escaláveis e confiáveis são padronizados, defensivos, observáveis e auditáveis por design.
