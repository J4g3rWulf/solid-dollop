# evaluate_model.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, precision_score, recall_score, f1_score

# Configuração (deve coincidir com o script de treinamento)
IMAGE_SIZE = (256, 256)         # Mesmo tamanho de imagem usado durante o treinamento
BATCH_SIZE = 24                 # Mesmo tamanho de lote usado no treinamento
TEST_DIR = './images/test/' 

# Carregar modelo treinado
model = tf.keras.models.load_model('trash_classifier_model_finetuned.keras', compile=False)  # Carregar o modelo salvo

# Carregar conjunto de teste
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,       
    batch_size=BATCH_SIZE,      
    shuffle=False                # Não embaralhar para manter ordem consistente
)

class_names = test_ds.class_names  

def evaluate_per_class(model, dataset, class_names):
    y_true = []         
    y_pred = []         
    y_pred_probs = [] 

    # Fazer previsões em cada lote
    for images, labels in dataset:
        preds = model.predict(images, verbose=0)      
        y_pred.extend(np.argmax(preds, axis=1))      
        y_pred_probs.extend(preds)          
        y_true.extend(labels.numpy()) 

    # Converter listas para arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_probs = np.array(y_pred_probs)

    print("\nAvaliação por categoria:")
    for i, class_name in enumerate(class_names):
        idxs = np.where(y_true == i)[0] 
        if len(idxs) == 0:
            print(f"Classe '{class_name}' não encontrada no conjunto de teste.")
            continue

        true_class = y_true[idxs]
        pred_class = y_pred[idxs]
        prob_class = y_pred_probs[idxs] 

        acc = accuracy_score(true_class, pred_class) 

        try:
            loss = log_loss(true_class, prob_class, labels=range(len(class_names)))  # Log loss
        except ValueError:
            loss = float('nan')

        print(f"Classe: {class_name}")
        print(f"   - Acurácia: {acc:.4f}")
        print(f"   - Perda:    {loss:.4f}")
        print("")

    # Métricas gerais
    overall_acc = accuracy_score(y_true, y_pred)
    overall_prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    overall_rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    overall_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print("Avaliação geral:")
    print(f"   - Acurácia:  {overall_acc:.4f}")
    print(f"   - Precisão:  {overall_prec:.4f}")
    print(f"   - Recall:    {overall_rec:.4f}")
    print(f"   - F1-score:  {overall_f1:.4f}")
    print("")

    # Visualização da matriz de confusão
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))  # Calcular matriz
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,              # Mostrar números nas células
        fmt='d',                 # Formato inteiro
        cmap='Blues',            # Esquema de cores
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        linecolor='gray',
        square=True
    )
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.title("Confusion Matrix")       # Título do gráfico
    plt.xlabel("Predicted")              # Rótulos previstos
    plt.ylabel("True")                  # Rótulos verdadeiros
    plt.tight_layout()
    plt.show()

# Executar avaliação
evaluate_per_class(model, test_ds, class_names)
