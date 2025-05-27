# **AI Brain Factory - Editor de modelos de IA - Assistente de edição de modelos distill pré treinados para processo educativo do funcionamento de treinamento com fine tunning e supervisão**  

**Transformando a educação através de diálogos inteligentes e personalizados para treinar modelos de AI**  

**OBS**.: O uso principal para este método de treinamento é para que alunos possam entender o passo a passo de como funciona o treinamento por fine tunning supervisionado de modelos distill, usando um chat, mas podem haver outras utilidades no futuro.

---

## ** Visão Geral**  
O **AI Brain Factory** é um projeto de IA generativa que utiliza o **DialoGPT** (modelo de linguagem da Microsoft) para criar assistentes de diálogo voltados para:  
- **Apoio à redação** (correção gramatical e estruturação de textos)  
- **Prática de conversação** (simulação de debates e perguntas-respostas)  
- **Aprendizado personalizado** (respostas adaptadas ao nível do aluno)  

** Repositório Oficial**: `github.com/Marcelo-S-Pereira/treinamento.interativo.de.modelo`  

---

## **Recursos**  
1 - **Chatbot Educativo** – Editor Diálogos contextualizados para aulas de português, história e ciências  
2 -  **Correção em Tempo Real** – Identifica erros gramaticais e sugere melhorias  
3 - **Modo Debate** – Simula discussões sobre temas atuais para treinar argumentação  
4 - **Avaliação por Recompensas** – Sistema de pontos por respostas coerentes  

---

## ** Como Usar**  

### **Pré-requisitos**  
- Python 3.8+  
- Biblioteca `transformers` (Hugging Face)  
- GPU (recomendado)
- CPU mínimo I5 ou FX6300
- RAM 16gb (ideal)
- Não funciona em computador com RAM menor que 10gb


### **Instalação**  
```bash
git clone https://github.com/seu-usuario/DialoEscola.git
cd DialoEscola
pip install -r requirements.txt
```  


## ** Estrutura do Projeto**  
```
DialoEscola/  
├── DialoGPT_HITL_RL_v1.py          # Script principal  
├── verificar.txt/                  # Letter de instruções básicas  
└── README.md                       # Este arquivo  
```  

---

## ** Métricas de Avaliação**  
O projeto utiliza:  
- **BLEU Score** (avaliação de qualidade textual)  
- **BERTScore** (similaridade semântica)
- **METEOR**
- **ROUGE-L**
- **Feedback de Professores ou Alunos** (Nota definitiva para decidir se o modelo precisa de treinamento)

 **Funcionamento das Métricas de Avaliação do DialoEscola** 

#### **1. BLEU Score – O "Corretor Ortográfico Robótico"**  
**Experimento mental:** Imagine que o BLEU Score é um **professor de português corrigindo uma cópia de texto**. Ele pega a resposta do aluno (gerada pela IA) e compara palavra por palavra com um gabarito ideal (resposta humana).  

- **Como funciona?**  
  - Se o aluno escreve *"O cachorro correu no parque"* e o gabarito era *"O cão correu no parque"*, o BLEU dá uma nota alta (as frases são quase iguais).  
  - Se o aluno escreve *"Um felino saltou"* para um gabarito sobre cachorros, a nota é baixa, por que não há nada parecido com isto na frase ensinada.  

**Limitação:**  
➠ É **rígido como um ditado escolar** – não entende sinônimos ou sentido semântico (por isso usamos também o BERTScore).  

---

#### **2. BERTScore – O "Professor de Redação que Entende Contexto"**  
**Experimento mental:** Se o BLEU é um corretor de ditado, o BERTScore é **aquele professor que avalia ideias, não só palavras exatas**. Ele usa um modelo de IA (BERT) para comparar **o significado** das respostas.  

- **Como funciona?**  
  - Se a IA diz *"A poluição prejudica o ar"* e o gabarito era *"A qualidade do ar piora com contaminantes"*, o BERTScore dá nota alta (mesmo sentido).  
  - Se a IA diz *"Plantas são verdes"* para uma pergunta sobre poluição, a nota cai.  

**Vantagem:**  
➠ **Entende paráfrases e contexto** – como um humano faria.  

---

#### **3. Feedback de Professores – O "Conselho Pedagógico Humano"**  
**Experimento mental:** Por melhor que sejam as métricas automáticas, elas são como **simulados de prova**. O feedback real de professores é a **prova final, aplicada por quem conhece os alunos**. Esta é a nota humana, de supervisão do desempenho do modelo o mesmo que se faz com alunos ao responder uma avaliação, então faz-se o mesmo com o modelo.

- **Como funciona?**  
  - Professores avaliam:  
    - **Clareza**: *"A resposta foi fácil de entender?"*  
    - **Relevância**: *"Respondeu ao que foi perguntado?"*  
    - **Pedagogica**: *"Essa resposta ajudaria um aluno de verdade?"*  

**Exemplo Prático:**  
➠ Se a IA corrige *"menas"* para *"menos"*, mas o aluno era disléxico e precisava de uma explicação visual, o professor sugere ao modelo que processa imagens o comando **incluir um exemplo com imagem**.  

---

### ** Comparação Final**  
| Métrica          | Analogia                     | Avalia                  | Ideal Para               |  
|------------------|-----------------------------|-------------------------|--------------------------|  
| **BLEU Score**   | Corretor de ditado          | Palavras exatas         | Exercícios formais (ex: gramática) |  
| **BERTScore**    | Professor de redação        | Significado             | Respostas criativas      |  
| **Feedback Humano** | Conselho pedagógico       | Necessidades reais      | Personalização           |  

---

### ** Exemplo Prático**  
**Pergunta:** *"Por que a água é importante?"*  

- **Resposta da IA:** *"Água é vital para a vida humana e animal."*  
  - BLEU: Nota 7/10 (falta "plantas")  
  - BERTScore: 9/10 (captou o essencial)  
  - Professores: 6/10 (*"Faltou exemplos concretos para crianças"*)  

**Resultado:** O modelo é ajustado para incluir *"Água ajuda plantas a crescerem, sacia a sede e mantém rios limpos"*.  
---

Termos técnicos no treinamento:

Epoch: Quantas vezes o modelo viu todo o conjunto de dados de treino. Como reler um livro 3 vezes para entender melhor.
Loss: Representa o valor da função de perda, e está em 3.24, é alto no início mas o importante é o quanto chegou no último epoch, ou seja, indica quão "erradas" estão as previsões do modelo em relação aos rótulos reais ou às saídas do modelo professor (no caso da destilação), então se caiu para 1.64, e isto é bom.

Grad_norm: Norma (magnitude) dos gradientes durante a atualização dos parâmetros do modelo, ou seja, indicam a direção e intensidade dos ajustes nos pesos do modelo para minimizar a loss. Maior que 100 o gradiente é instável, e menor que 0 indica que não aprendeu nada, está em 12 no primeiro epoch e depois caiu para 6, está regular, deveria ter ficado fixado, se ocorrer mais epochs pode ficar crítico, pois tende a cair sempre, mas não indicam sem explosão/vanishing.

Learning_rate: Taxa de aprendizagem (learning rate) usada pelo otimizador, ou seja, controla o tamanho dos passos que o modelo dá para ajustar seus parâmetros durante o treinamento. Vemos o valor de 3.33.10^5 (ou 0.0000333) é relativamente baixo, típico em etapas de ajuste fino (fine-tuning) ou treinamento estável. Indica que vai demorar treinar e muito. 

Train_runtime (43.39s): Tempo total que o treinamento levou para ser concluído. Como um cronômetro marcando a duração total, na verdade é maior, pois este cálculo é feito pelo modelo desconsiderando processamento dos dados, pre-processamento de tokens e embeddings e outros, e somente a entrada e a saída. Somando as 3 epochs chega a uns 2,17 por epoch o que seria 7 minutos em média.

Train_samples_per_second (0.069): Quantas amostras de dados o modelo consegue processar por segundo. Parecido com quantas páginas um escritor consegue revisar por minuto - quanto maior, mais rápido.

Train_steps_per_second (0.069): Quantos passos de atualização do modelo são feitos por segundo. Como um marcador de passos - cada passo é uma pequena melhoria no modelo.

Train_loss (2.275): Mede o erro do modelo durante o treino. Quanto menor, melhor o modelo está aprendendo. Como a pontuação de erros em um teste - você quer diminuir.

---

** Conclusão:**  
O DialoEscola (DialoGPT_HITL_RL_v1) combina **rigor técnico** (BLEU), **inteligência contextual** (BERTScore) e **sensibilidade humana** (professores humanos ou alunos) para criar um assistente educacional verdadeiramente eficaz.  

*"Um sistema de IA para educação é como uma sintaxe de idioma na lousa e um professor ensinando: precisa ser precisa (BLEU), semântica (BERTScore) e adaptável (Feedback Humano)."*

---

## ** Sugestão de Uso Real é a Didática e Experimentação**  
1. **Escolas Públicas** – Auxílio em aulas de Machine Learn interativo e imersivo, para introdução, e de maior compreensão, não seria necessário conhecimentos avançados para
2. **Escolas Técnicas** – Auxílio em aulas de Machine Learn interativo e imersivo, para introdução, e de maior compreensão, não seria necessário conhecimentos avançados para
3. **Universidade** - Auxílio em aulas de Machine Learn interativo e imersivo, para introdução nos primeiros períodos, e de maior compreensão, não seria necessário conhecimentos avançados para treinar o modelo.

---

## ** Como Contribuir**  
1. **Reporte bugs** (GitHub Issues)  
2. **Melhore o modelo** – Faça fine-tuning com novos dados, ou ajude no código a dimiuir overffiting.
3. **Desenvolver entradas em formulário Web** - Desenvolver a interface gráfica web.


## ** Licença**  
**MIT License** – Livre para uso acadêmico e comercial.  

--- 

** Saiba Mais**:  
- modelo usado: `huggingface.co/microsoft/DialoGPT-medium`  


** Contato**: `ds.marcelo.spereira@gmail.com`  

---  
<p align="center">  
   **Educação do futuro, construída hoje**  
</p>  

*(criado em: Maio de 2025)*  

--- 

### ** Próximos Passos**  
- [ ] Desenvolvimento da versão 2 deste editor para treinamento imersivo e interativo de modelos distill pré treinados. 
  

** Dê uma estrela no repositório se gostou e acredita nessa missão!**
