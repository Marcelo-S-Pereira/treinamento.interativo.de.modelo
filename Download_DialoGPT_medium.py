from transformers import AutoModelForCausalLM, AutoTokenizer

# Defina o diretório onde o modelo será salvo
MODEL_SAVE_PATH = r"C:\Users\cla_m\OneDrive\Área de Trabalho\conversação\modelos AI\DIALOGPT\dialogpt-medium"

# Nome do modelo no Hugging Face
MODEL_NAME = "microsoft/DialoGPT-medium"

# Baixar o tokenizador e o modelo
print(f"Baixando o modelo {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Salvar o modelo e tokenizador no diretório especificado
print(f"Salvando o modelo em {MODEL_SAVE_PATH}...")
tokenizer.save_pretrained(MODEL_SAVE_PATH)
model.save_pretrained(MODEL_SAVE_PATH)

print(f"Modelo {MODEL_NAME} baixado e salvo com sucesso em {MODEL_SAVE_PATH}!")