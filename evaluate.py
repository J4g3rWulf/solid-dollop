# evaluate_model.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, precision_score, recall_score, f1_score

# Configuration (must match training script)
IMAGE_SIZE = (256, 256)         # Same image size used during training
BATCH_SIZE = 24                 # Same batch size as training
TEST_DIR = './images/test/' 

# Load trained model
model = tf.keras.models.load_model('trash_classifier_model_finetuned.keras', compile=False)  # Load the saved model

# Load test dataset
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,       
    batch_size=BATCH_SIZE,      
    shuffle=False                
)

class_names = test_ds.class_names  

def evaluate_per_class(model, dataset, class_names):
    y_true = []         
    y_pred = []         
    y_pred_probs = [] 

    # 🔍 Predict on each batch
    for images, labels in dataset:
        preds = model.predict(images, verbose=0)      
        y_pred.extend(np.argmax(preds, axis=1))      
        y_pred_probs.extend(preds)          
        y_true.extend(labels.numpy()) 

    # 📊 Convert lists to arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_probs = np.array(y_pred_probs)

    print("\n📊 Evaluation per category:")
    for i, class_name in enumerate(class_names):
        idxs = np.where(y_true == i)[0] 
        if len(idxs) == 0:
            print(f"⚠️ Class '{class_name}' not found in test set.")
            continue

        true_class = y_true[idxs]
        pred_class = y_pred[idxs]
        prob_class = y_pred_probs[idxs] 

        acc = accuracy_score(true_class, pred_class) 

        try:
            loss = log_loss(true_class, prob_class, labels=range(len(class_names)))  # Log loss
        except ValueError:
            loss = float('nan')

        print(f"🧩 Class: {class_name}")
        print(f"   - Accuracy:  {acc:.4f}")
        print(f"   - Loss:      {loss:.4f}")
        print("")

    overall_acc = accuracy_score(y_true, y_pred)
    overall_prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    overall_rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    overall_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print("📊 Overall Evaluation:")
    print(f"   - Accuracy:  {overall_acc:.4f}")
    print(f"   - Precision: {overall_prec:.4f}")
    print(f"   - Recall:    {overall_rec:.4f}")
    print(f"   - F1-score:  {overall_f1:.4f}")
    print("")

    # 🔥 Confusion matrix visualization
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))  # Compute matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,              # Show numbers in cells
        fmt='d',                 # Integer format
        cmap='Blues',            # Color scheme
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        linecolor='gray',
        square=True
    )
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.title("Confusion Matrix")       # Title of the plot
    plt.xlabel("Predicted")             # Predicted labels
    plt.ylabel("True")                  # True labels
    plt.tight_layout()
    plt.show()

# 📈 Run evaluation
evaluate_per_class(model, test_ds, class_names)
