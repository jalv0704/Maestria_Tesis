import pandas as pd
import json
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset


import torch

# Verificar si hay GPU disponible
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")


# Cargar datos
def load_data(file_path):
    df = pd.read_excel(file_path)
    
    # Extraer texto de la conversación
    def extract_text(conv):
        try:
            conv_dict = json.loads(conv)
            mensajes = [turno.get("mensaje", "") for turno in conv_dict.get("conversacion", [])]
            return " ".join(mensajes)
        except json.JSONDecodeError:
            return ""
    
    df["Texto"] = df["Conversacion"].apply(extract_text)
    return df[["Texto", "CALIFICACIÓN"]]

# Dataset personalizado para PyTorch
class ConversationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

# Cargar archivo
file_path = "C:/Users/jdalv/OneDrive/Escritorio/Conversaciones.xlsx"
df = load_data(file_path)

# División de datos
X_train, X_test, y_train, y_test = train_test_split(df["Texto"], df["CALIFICACIÓN"], test_size=0.2, random_state=42)

# Tokenizador
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)

# Crear datasets
train_dataset = ConversationDataset(list(X_train), list(y_train), tokenizer)
test_dataset = ConversationDataset(list(X_test), list(y_test), tokenizer)

# Cargar modelo
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=11).to(device)


# Configurar entrenamiento
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Entrenar el modelo
trainer.train()

# Guardar modelo
model.save_pretrained("./bert_calificacion_model")
tokenizer.save_pretrained("./bert_calificacion_model")

print("Entrenamiento completado y modelo guardado.")
