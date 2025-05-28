import json
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from bert_score import score
import argparse
import nltk
import re
import time
import threading

# Baixar recursos do NLTK
nltk.download('wordnet')
nltk.download('omw-1.4')

# Configuração de hiperparâmetros
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--learning_rate", type=float, default=5e-5)
parser.add_argument("--num_train_epochs", type=int, default=3)
parser.add_argument("--max_length", type=int, default=50)
parser.add_argument("--min_length", type=int, default=1)
parser.add_argument("--do_sample", type=bool, default=True)
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--top_k", type=int, default=50)
parser.add_argument("--top_p", type=float, default=0.9)
parser.add_argument("--repetition_penalty", type=float, default=1.2)
parser.add_argument("--num_return_sequences", type=int, default=1)
parser.add_argument("--pad_token_id", type=int, default=50256)
parser.add_argument("--eos_token_id", type=int, default=50256)

args = parser.parse_args()

# Caminhos para salvar e carregar os modelos
modelo_path = r"C:\Users\cla_m\OneDrive\Área de Trabalho\conversação\modelos_AI\SALA_DE_AULA\DIALO_GPT_READ"
salvar_path = r"C:\Users\cla_m\OneDrive\Área de Trabalho\conversação\modelos_AI\SALA_DE_AULA\DIALO_GPT_WRITE"

# Carregar modelo e tokenizer
tokenizer = AutoTokenizer.from_pretrained(modelo_path)
model = AutoModelForCausalLM.from_pretrained(modelo_path)

# Configurar o token de padding, se necessário
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Usa o token de fim de sequência como padding

# Mover o modelo para o dispositivo (CPU ou GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Classe para o dataset de conversa
class ConversaDataset(Dataset):
    def __init__(self, inputs, masks, labels):
        self.inputs = inputs
        self.masks = masks
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            "input_ids": self.inputs[idx],
            "attention_mask": self.masks[idx],
            "labels": self.labels[idx]
        }

# Função para gerar resposta contextualizada
def gerar_resposta(contexto, max_new_tokens=100, temperature=0.7, top_k=50, top_p=0.9):
    input_ids = tokenizer.encode(contexto + tokenizer.eos_token, return_tensors="pt").to(device)
    
    # Verifica se o tokenizer tem um token de padding
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    resposta_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        pad_token_id=pad_token_id,  # Usa o token de padding ou EOS
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=args.repetition_penalty,
        attention_mask=input_ids.ne(pad_token_id).float().to(device)  # Usa o token de padding ou EOS
    )
    resposta = tokenizer.decode(resposta_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return resposta

# Função de tokenização detalhada
def tokenizar_frase(frase):
    tokens = re.findall(r"\w+|[^\w\s]", frase.lower())
    return tokens

# Função de análise comparativa
def analisar_tokens(tokens_esperados, tokens_gerados, max_pontuacao, meia_recompensa):
    recompensa_por_token = max_pontuacao / len(tokens_esperados) if tokens_esperados else 0
    penalidade = -recompensa_por_token
    
    tabela = []
    pontuacao_total = 0
    
    # Preencher tokens ausentes com strings vazias
    max_tokens = max(len(tokens_esperados), len(tokens_gerados))
    tokens_esperados += [''] * (max_tokens - len(tokens_esperados))
    tokens_gerados += [''] * (max_tokens - len(tokens_gerados))
    
    for i, (te, tg) in enumerate(zip(tokens_esperados, tokens_gerados)):
        if tg == te:
            status = f"posição {i+1} correta"
            recompensa = recompensa_por_token
        elif tg in tokens_esperados:
            pos_correta = tokens_esperados.index(tg)
            status = f"posição {i+1} incorreta (deveria ser {pos_correta+1})"
            recompensa = meia_recompensa
        else:
            status = "ausente"
            recompensa = penalidade
            
        pontuacao_total += recompensa
        tabela.append({
            'esperado': te,
            'gerado': tg,
            'status': status,
            'recompensa': round(recompensa, 4)
        })
    
    return pontuacao_total, tabela

# Função para calcular métricas
def calcular_metricas(resposta_gerada, resposta_esperada):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(resposta_esperada, resposta_gerada)
    smoothing_function = SmoothingFunction().method1
    bleu_score = sentence_bleu([resposta_esperada.split()], resposta_gerada.split(), smoothing_function=smoothing_function)
    meteor_score_value = meteor_score([resposta_esperada.split()], resposta_gerada.split())
    bert_score_value = score([resposta_gerada], [resposta_esperada], lang="en")[2].mean().item()
    return rouge_scores, bleu_score, meteor_score_value, bert_score_value

def truncar_contexto(contexto, max_tokens=1000):
    tokens = tokenizer.encode(contexto)
    if len(tokens) > max_tokens:
        tokens = tokens[-max_tokens:]  # Mantém os últimos `max_tokens` tokens
    return tokenizer.decode(tokens)

# Função para processar o treinamento
def processar_treinamento(contexto_acumulado):
    # Tokeniza o contexto acumulado
    inputs = tokenizer(contexto_acumulado, return_tensors="pt", padding=True, truncation=True).to(device)
    labels = inputs.input_ids.clone()  # Labels são iguais aos inputs para modelos de linguagem
    
    # Cria o dataset
    dataset = ConversaDataset([inputs.input_ids], [inputs.attention_mask], [labels])
    
    # Configurações de treinamento
    training_args = TrainingArguments(
        output_dir=salvar_path,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=1,
        save_strategy="no",
        remove_unused_columns=False
    )
    
    # Inicializa o Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )
    
    # Treinamento
    trainer.train()
    
    # Salva o modelo após o treinamento
    model.save_pretrained(salvar_path)
    tokenizer.save_pretrained(salvar_path)
    print(f"Modelo salvo em {salvar_path}")

# Função para coletar feedback humano
def coletar_feedback():
    while True:
        try:
            feedback = input("\nAvalie a resposta (0.0-1.0): ")
            if not feedback:  # Se o usuário pressionar Enter sem digitar nada
                print("Por favor, insira um valor entre 0.0 e 1.0.")
                continue
            feedback = float(feedback)
            if 0.0 <= feedback <= 1.0:
                return feedback
            else:
                print("Por favor, insira um valor entre 0.0 e 1.0.")
        except ValueError:
            print("Entrada inválida. Por favor, insira um número entre 0.0 e 1.0.")

# Variável global para controle de interrupção
interromper_treinamento = False

# Função principal
def treinamento_continuo():
    global interromper_treinamento
    
    num_turnos = int(input("Total de turnos: "))
    max_pontuacao = float(input("Pontuação máxima por turno (0.0-1.0): "))
    meia_recompensa = float(input("Meia recompensa (0.0-1.0): "))
    pontuacao_esperada = float(input("Pontuação esperada (0.0-1.0): "))
    total_treinamentos = int(input("Total de iterações de treinamento: "))
    
    contexto_acumulado = []
    
    for iteracao in range(total_treinamentos):
        if interromper_treinamento:
            break
        
        print(f"\n--- Iteração de Treinamento {iteracao+1}/{total_treinamentos} ---")
        contexto = ""
        pontuacoes = []
        metricas = {'rouge': [], 'bleu': [], 'meteor': [], 'bert': []}
        
        for turno in range(num_turnos):
            if interromper_treinamento:
                break
            
            print(f"\nTurno {turno + 1}")
            dialogo = input("Diálogo: ")
            intencao = input("Intenção (1 - Sugerir | 2 - Discordar | 3 - Perguntar | 4 - Concordar): ")
            emocao = input("Emoção (0 - Neutro | 1 - Feliz | 2 - Triste | 3 - Surpreso | 4 - Animado | 5 - Raiva | 6 - Medo): ")
            idioma = input("Idioma (en, pt, en-pt, pt-en): ")
            
            # Adiciona ao contexto acumulado
            contexto += dialogo + tokenizer.eos_token
            contexto_acumulado.append(dialogo)
            
            # Truncar o contexto se necessário
            contexto = truncar_contexto(contexto, max_tokens=1000)
    
            # Gerar resposta do modelo
            resposta = gerar_resposta(contexto)
            print(f"\nResposta do Modelo: {resposta}")
            
            # Atualiza o contexto com a resposta do modelo
            contexto += resposta + tokenizer.eos_token
            contexto_acumulado.append(resposta)
            
            # Tokeniza e analisa
            tokens_esperados = tokenizar_frase(dialogo)
            tokens_gerados = tokenizar_frase(resposta)
            pontuacao, tabela = analisar_tokens(tokens_esperados, tokens_gerados, max_pontuacao, meia_recompensa)
            pontuacoes.append(pontuacao)
            
            # Mostra tabela de análise
            print("\nTabela de Análise de Tokens")
            print("Token Esperado | Token Gerado | Status | Recompensa")
            print("-" * 60)
            for linha in tabela:
                print(f"{linha['esperado']:13} | {linha['gerado']:11} | {linha['status']:25} | {linha['recompensa']:8.4f}")
            
            # Calcular métricas
            rouge_scores, bleu_score, meteor_score_value, bert_score_value = calcular_metricas(resposta, dialogo)
            metricas['rouge'].append(rouge_scores['rougeL'].fmeasure)
            metricas['bleu'].append(bleu_score)
            metricas['meteor'].append(meteor_score_value)
            metricas['bert'].append(bert_score_value)
            
            # Exibir métricas e pontuação
            print("\nMétricas e Pontuação:")
            print(f"Pontuação: {pontuacao:.4f}")
            print(f"ROUGE-L: {rouge_scores['rougeL'].fmeasure:.4f}")
            print(f"BLEU: {bleu_score:.4f}")
            print(f"METEOR: {meteor_score_value:.4f}")
            print(f"BERTScore: {bert_score_value:.4f}")
            
            # Coletar feedback humano
            feedback = coletar_feedback()
            
            # Verificar se o feedback é satisfatório
            if feedback < pontuacao_esperada:
                print("Feedback insatisfatório. Reiniciando o processo...")
                
                # Processa o treinamento com o contexto acumulado
                processar_treinamento(" ".join(contexto_acumulado))
                
                # Contagem de 5 segundos
                print("\nPróxima iteração em 5 segundos... (pressione 's' para interromper)")
                interromper_treinamento = False
                
                def temporizador():
                    for i in range(6):
                        if interromper_treinamento:
                            break
                        print(f"Tempo: {i} segundos")
                        time.sleep(1)
                    if not interromper_treinamento:
                        print("Continuando para a próxima iteração...")
                
                timer_thread = threading.Thread(target=temporizador)
                timer_thread.start()
                
                # Aguarda input do usuário
                user_input = input()
                if user_input.lower() == 's':
                    interromper_treinamento = True
                    if input("Salvar progresso? (s/n): ").lower() == 's':
                        model.save_pretrained(salvar_path)
                        tokenizer.save_pretrained(salvar_path)
                    return
            else:
                print("Feedback satisfatório. Continuando...")
        
        # Calcular média de pontuação e métricas
        media_pontuacao = sum(pontuacoes) / len(pontuacoes)
        media_rouge = sum(metricas['rouge']) / len(metricas['rouge'])
        media_bleu = sum(metricas['bleu']) / len(metricas['bleu'])
        media_meteor = sum(metricas['meteor']) / len(metricas['meteor'])
        media_bert = sum(metricas['bert']) / len(metricas['bert'])
        
        print("\nMédias Finais:")
        print(f"Pontuação: {media_pontuacao:.4f}")
        print(f"ROUGE-L: {media_rouge:.4f}")
        print(f"BLEU: {media_bleu:.4f}")
        print(f"METEOR: {media_meteor:.4f}")
        print(f"BERTScore: {media_bert:.4f}")
        
        # Salva o modelo após cada iteração
        model.save_pretrained(salvar_path)
        tokenizer.save_pretrained(salvar_path)
        print(f"Modelo salvo em {salvar_path}")

# Executar
if __name__ == "__main__":
    treinamento_continuo()
    
    