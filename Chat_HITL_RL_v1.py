# CÉLULA 1: Importações e Configurações (MODIFICADO)
# ==============================================================================

import json
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from torch.utils.data import Dataset
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from bert_score import score
import argparse
import nltk, re, time
import keyboard 

# ==============================================================================

# CÉLULA 2: Baixar recursos do NLTK
nltk.download('wordnet')
nltk.download('omw-1.4')

# ==============================================================================


# CÉLULA 3: Configuração de hiperparâmetros
# ==============================================================================

parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=1e-5, help="Taxa de aprendizado.") #new
parser.add_argument("--num_train_epochs", type=int, default=5, help="Número de épocas.") #new
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

# ==============================================================================


# Caminhos para salvar e carregar os modelos
# ==============================================================================

#instrução: 
# 1 - Depois de baixar o modelo, crie uma pasta com nome de modelo_path
#         obs: a pasta modelo_path terá um modelo para leitura
# 2 - crie uma pasta com nome de salvar_path e coloque uma copia do modelo treinado
#         obs: a pasta salvar_path com o modelo baixado vai receber novos treinamentos
# 3 - após o treinamento copie o modelo treinado para a pasta modelo_path
#         obs: poderá observar as novas respostas e acumular treinamento desta maneira.

modelo_path = r"C:\Users\cla_m\OneDrive\Área de Trabalho\conversação\modelos_AI\SALA_DE_AULA\DIALO_GPT_READ"
salvar_path = r"C:\Users\cla_m\OneDrive\Área de Trabalho\conversação\modelos_AI\SALA_DE_AULA\DIALO_GPT_WRITE"

# ==============================================================================


# Carregar modelo e tokenizer
# ==============================================================================

tokenizer = AutoTokenizer.from_pretrained(modelo_path)
model = AutoModelForCausalLM.from_pretrained(modelo_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

# ==============================================================================

# Função para separar as palavras da pontuação
# ==============================================================================

# A função irá separar os pontos das palavras
# Gerar uma lista única com as palavras e pontos na ordem correta.

def separar_pontos_e_palavras(frase):
    palavras = frase.split()
    lista_final = []
    for palavra in palavras:
        # Separa a palavra dos pontos (se houver)
        if '.' in palavra:
            partes = palavra.split('.')
            lista_final.append(partes[0])  # Adiciona a palavra
            lista_final.append('.')        # Adiciona o ponto
        else:
            lista_final.append(palavra)    # Adiciona a palavra sem pontos
    return lista_final

# ==============================================================================

# Métodos de avaliação
# ==============================================================================
# Função que compara a frase gerada com a frase esperada e gera a tabela de comparação.

def comparar_frases(frase_esperada, frase_gerada, recompensa_maxima, penalizacao_maxima, meia_recompensa):
    # Separar pontos e palavras
    lista_esperada = separar_pontos_e_palavras(frase_esperada)
    lista_gerada = separar_pontos_e_palavras(frase_gerada)
    
    # Determinar o tamanho máximo das listas
    tamanho_maximo = max(len(lista_esperada), len(lista_gerada))
    
    # Preencher listas com valores vazios, se necessário
    lista_esperada += [''] * (tamanho_maximo - len(lista_esperada))
    lista_gerada += [''] * (tamanho_maximo - len(lista_gerada))
    
    # Inicializar variáveis
    pontuacao_total = 0
    detalhes = []
    
    for i in range(tamanho_maximo):
        palavra_esperada = lista_esperada[i]
        palavra_gerada = lista_gerada[i]
        
        if palavra_esperada == palavra_gerada:
            status = "posição correta"
            recompensa = recompensa_maxima
        elif palavra_gerada == '':
            status = "ausente"
            recompensa = penalizacao_maxima
        elif palavra_gerada in lista_esperada:
            status = "posição incorreta"
            distancia = abs(i - lista_esperada.index(palavra_gerada))
            recompensa = meia_recompensa * (1 - (distancia / len(lista_esperada)))
        else:
            status = "ausente"
            recompensa = penalizacao_maxima
        
        # Adicionar detalhes à tabela
        detalhes.append({
            "esperada": palavra_esperada,
            "gerada": palavra_gerada,
            "status": status,
            "recompensa": recompensa
        })
        pontuacao_total += recompensa
    
    return pontuacao_total, detalhes

# ==============================================================================


# Função para gerar resposta contextualizada
# ==============================================================================

def gerar_resposta(contexto, max_length=50, temperature=0.7, top_k=50, top_p=0.9):
    input_ids = tokenizer.encode(contexto + tokenizer.eos_token, return_tensors="pt").to(device)
    resposta_ids = model.generate(
        input_ids,
        max_length=max_length,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=args.repetition_penalty
    )
    resposta = tokenizer.decode(resposta_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return resposta

# Função de tokenização detalhada
def tokenizar_frase(frase):
    tokens = re.findall(r"\w+|[^\w\s]", frase.lower())
    return tokens

# Função de análise comparativa
def analisar_tokens(tokens_esperados, tokens_gerados, max_pontuacao):
    recompensa_por_token = max_pontuacao / len(tokens_esperados) if tokens_esperados else 0
    meia_recompensa = recompensa_por_token * 0.5
    penalidade = -recompensa_por_token
    
    tabela = []
    pontuacao_total = 0
    
    for i, (te, tg) in enumerate(zip(tokens_esperados, tokens_gerados)):
        if i >= len(tokens_gerados):
            tg = ''
            
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

# Função principal
def treinamento_continuo():
    num_turnos = int(input("Total de turnos: "))
    max_pontuacao = float(input("Pontuação máxima por turno (0.0-1.0): "))
    total_treinamentos = int(input("Total de iterações de treinamento: "))
    
    for iteracao in range(total_treinamentos):
        print(f"\n--- Iteração de Treinamento {iteracao+1}/{total_treinamentos} ---")
        contexto = ""
        
        for turno in range(num_turnos):
            print(f"\nTurno {turno + 1}")
            dialogo = input("Diálogo: ")
            contexto += dialogo + tokenizer.eos_token  # Adicionar ao contexto
            
            # Gerar resposta do modelo
            resposta = gerar_resposta(contexto)
            print(f"\nResposta do Modelo: {resposta}")
            
            # Atualizar contexto com a resposta do modelo
            contexto += resposta + tokenizer.eos_token
            
            # Tokenizar e analisar
            tokens_esperados = tokenizar_frase(dialogo)
            tokens_gerados = tokenizar_frase(resposta)
            pontuacao, tabela = analisar_tokens(tokens_esperados, tokens_gerados, max_pontuacao)
            
            # Mostrar tabela de análise
            print("\nTabela de Análise de Tokens")
            print("Token Esperado | Token Gerado | Status | Recompensa")
            print("-" * 60)
            for linha in tabela:
                print(f"{linha['esperado']:13} | {linha['gerado']:11} | {linha['status']:25} | {linha['recompensa']:8.4f}")
            
            # Coletar feedback humano
            feedback = float(input("\nAvalie a resposta (0.0-1.0): "))
            
            # Verificar se o treinamento deve continuar
            if abs(feedback - pontuacao) <= 0.05:
                print("Feedback dentro da margem aceitável!")
                if input("Salvar modelo? (s/n): ").lower() == 's':
                    model.save_pretrained("modelo_ajustado")
                    tokenizer.save_pretrained("modelo_ajustado")
                return
        
        # Controle de tempo
        print("\nPróxima iteração em 30 segundos... (pressione 's' para interromper)")
        start_time = time.time()
        while time.time() - start_time < 30:
            if input() == 's':
                if input("Salvar progresso? (s/n): ").lower() == 's':
                    model.save_pretrained(salvar_path)
                    tokenizer.save_pretrained(salvar_path)
                return

# Executar
if __name__ == "__main__":
    treinamento_continuo()