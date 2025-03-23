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
    
    # Mapeo de calificaciones textuales a numéricas
    mapeo_calificaciones = {
        "MALA": 0,
        "REGULAR": 1,
        "BUENA": 2
    }
    
    # Extraer texto de la conversación
    def extract_text(conv):
        try:
            conv_dict = json.loads(conv)
            mensajes = [turno.get("mensaje", "") for turno in conv_dict.get("conversacion", [])]
            return " ".join(mensajes)
        except json.JSONDecodeError:
            return ""
    
    df["Texto"] = df["Conversacion"].apply(extract_text)
    
    # Limpiar y convertir calificaciones
    df["CALIFICACIÓN"] = df["CALIFICACIÓN"].str.strip().str.upper()
    df["CALIFICACIÓN"] = df["CALIFICACIÓN"].map(mapeo_calificaciones)
    
    # Eliminar filas con calificaciones no reconocidas
    df = df.dropna(subset=["CALIFICACIÓN"])
    df["CALIFICACIÓN"] = df["CALIFICACIÓN"].astype(int)
    
    return df[["Texto", "CALIFICACIÓN"]]

# Dataset personalizado para PyTorch (sin cambios en la clase)
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
    num_train_epochs=20,
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



import torch
print(torch.cuda.is_available())  # Debe ser True
print(torch.cuda.get_device_name(0))  # Debe mostrar el modelo de la GPU

import torch
print(torch.cuda.is_available())  # Debe ser True
print(torch.cuda.get_device_name(0))  # Debe mostrar el modelo de la GPU



from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# Función para evaluar el modelo
def evaluate_model(trainer, test_dataset):
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=1)  # Convertir logits a clases
    labels = predictions.label_ids

    # Calcular métricas
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

# Evaluar modelo después del entrenamiento
evaluate_model(trainer, test_dataset)


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Función para mostrar la matriz de confusión con etiquetas
def plot_confusion_matrix(labels, preds, class_names):
    cm = confusion_matrix(labels, preds)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_names, 
                yticklabels=class_names)
    
    plt.xlabel("Predicciones")
    plt.ylabel("Verdaderas Etiquetas")
    plt.title("Matriz de Confusión - Calificaciones de Conversaciones")
    plt.show()

# Obtener predicciones del modelo
predictions = trainer.predict(test_dataset)
preds = np.argmax(predictions.predictions, axis=1)
labels = predictions.label_ids

# Generar la matriz con nombres de clases
plot_confusion_matrix(labels, preds, class_names=["MALA", "REGULAR", "BUENA"])

