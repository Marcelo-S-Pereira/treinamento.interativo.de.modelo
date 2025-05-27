Dado técnicos

---

1. import json
   
import: Comando para carregar módulos/bibliotecas

json: Módulo para manipular dados no formato JSON (JavaScript Object Notation)

---

2. from transformers import AutoTokenizer, AutoModelForCausalLM
   
from: Especifica o módulo de origem

transformers: Biblioteca da Hugging Face para modelos de NLP

import: Traz classes específicas

AutoTokenizer: Classe que tokeniza textos automaticamente

AutoModelForCausalLM: Classe para modelos generativos (ex: GPT)

---

3. import torch
   
torch: Biblioteca PyTorch para computação tensorial e deep learning

---

4. from rouge_score import rouge_scorer
   
rouge_score: Pacote para métrica ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

rouge_scorer: Classe que calcula similaridade entre textos

---

5. from nltk.translate.bleu_score import sentence_bleu
   
nltk: Natural Language Toolkit

translate: Submódulo para tradução

bleu_score: Implementação da métrica BLEU

sentence_bleu: Função que calcula BLEU para frases individuais

---

6. from bert_score import score

bert_score: Métrica baseada em embeddings BERT

score: Função principal de avaliação

---

7. import argparse
   
argparse: Módulo para processar argumentos da linha de comando

---

8. import nltk
   
nltk: Biblioteca para processamento de linguagem natural

---

9. import re
    
re: Módulo para operações com expressões regulares

---

10. import time
    
time: Módulo para manipulação de tempo e pausas

---

11. nltk.download('wordnet')
    
nltk.download: Função para baixar recursos

'wordnet': Banco de dados léxico-semântico

---

12. nltk.download('omw-1.4')
    
'omw-1.4': Open Multilingual WordNet (extensão para múltiplos idiomas)

---

13. parser = argparse.ArgumentParser()
    
parser: Variável que armazena o parser

argparse.ArgumentParser: Classe que cria o parser de argumentos

---

14. parser.add_argument("--batch_size", type=int, default=1)
    
add_argument: Método para adicionar um argumento

"--batch_size": Nome do argumento (com dois hífens)

type=int: Especifica que o valor deve ser inteiro

default=1: Valor padrão se não for fornecido

---

15. parser.add_argument("--learning_rate", type=float, default=5e-5)
    
type=float: Valor deve ser decimal

default=5e-5: 0.00005 (notação científica)

---

16. parser.add_argument("--num_train_epochs", type=int, default=3)
    
"num_train_epochs": Número de passagens completas pelo dataset

---

17. parser.add_argument("--max_length", type=int, default=50)
    
"max_length": Comprimento máximo em tokens das respostas

---

18. parser.add_argument("--min_length", type=int, default=1)
    
"min_length": Comprimento mínimo em tokens

---

19. parser.add_argument("--do_sample", type=bool, default=True)
    
"do_sample": Habilita amostragem estocástica

type=bool: Valor booleano (True/False)

---

20. parser.add_argument("--temperature", type=float, default=0.7)
    
"temperature": Controla aleatoriedade (valores altos = mais criativo)

---

21. parser.add_argument("--top_k", type=int, default=50)
    
"top_k": Filtra apenas os K tokens mais prováveis

---

22. parser.add_argument("--top_p", type=float, default=0.9)
    
"top_p": Filtra tokens até acumular 90% de probabilidade

---

23. parser.add_argument("--repetition_penalty", type=float, default=1.2)
    
"repetition_penalty": Penaliza tokens repetidos (valor >1 = mais penalização)

---

24. parser.add_argument("--num_return_sequences", type=int, default=1)
    
"num_return_sequences": Quantas respostas gerar por entrada

---

25. parser.add_argument("--pad_token_id", type=int, default=50256)
    
"pad_token_id": ID numérico do token de preenchimento (padding)

---

26. parser.add_argument("--eos_token_id", type=int, default=50256)
    
"eos_token_id": ID do token de fim de sequência (End Of Sequence)

---

27. args = parser.parse_args()
    
args: Variável que armazena os argumentos processados

parse_args(): Método que extrai os valores fornecidos

---

28. modelo_path = r"C:\caminho\modelo"
    
modelo_path: String com caminho absoluto

r: Prefixo para string raw (ignora escapes como \n)

---

29. salvar_path = r"C:\caminho\salvar"
    
salvar_path: Caminho para salvar o modelo treinado

---

30. tokenizer = AutoTokenizer.from_pretrained(modelo_path)
    
from_pretrained: Método que carrega um tokenizador pré-treinado

---

31. model = AutoModelForCausalLM.from_pretrained(modelo_path)
    
AutoModelForCausalLM: Classe para modelos generativos

---

32. device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
torch.device: Configura dispositivo de processamento

"cuda": Dispositivo GPU (se disponível)

torch.cuda.is_available(): Verifica se há GPU compatível

else "cpu": Fallback para CPU

---

33. model.to(device)
to(device): Move o modelo para GPU/CPU

---
