# train_model.py
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Número de threads usadas dentro de uma operação individual (como multiplicação de matrizes)
tf.config.threading.set_intra_op_parallelism_threads(8)

# Número de threads usadas entre operações independentes
tf.config.threading.set_inter_op_parallelism_threads(4)

import tensorflow as tf

# 📦 Configuração
IMAGE_SIZE = (256, 256)                     # Dimensões da imagem de entrada (altura, largura)
BATCH_SIZE = 24                             # Número de imagens processadas por passo de treinamento
EPOCHS_INITIAL = 70                         # Épocas para a fase inicial de treinamento
EPOCHS_FINE_TUNE = 35                       # Épocas para o ajuste fino com taxa de aprendizado menor
DATA_DIR = data_dir = './images/train'      # Caminho para as pastas de imagens de treinamento
VALIDATION_SPLIT_CF = 0.1                   # 10% dos dados usados para validação

def focal_loss_multiclass(y_true, y_pred, alpha=0.25, gamma=3.0):

    num_classes = tf.shape(y_pred)[-1]                       
    y_true_onehot = tf.one_hot(tf.cast(y_true, tf.int32),    
                               depth=num_classes)

    epsilon = tf.keras.backend.epsilon()                     
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon) 

    pt = tf.reduce_sum(y_true_onehot * y_pred, axis=-1)      
    modulating_factor = tf.pow(1. - pt, gamma)               
    ce = -tf.math.log(pt)                                    

    if isinstance(alpha, (float, int)):                      
        alpha_factor = alpha
    else:                                                    
        alpha_factor = tf.reduce_sum(y_true_onehot * alpha, axis=-1)

    loss = alpha_factor * modulating_factor * ce             
    return tf.reduce_mean(loss)                              

# 📁 Carregamento e pré-processamento do dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    validation_split=VALIDATION_SPLIT_CF,
    subset="training",           # Subconjunto de treinamento
    seed=123,                    # Semente para reprodutibilidade
    image_size=IMAGE_SIZE,       # Redimensionar imagens para o tamanho de entrada do modelo
    batch_size=BATCH_SIZE        # Tamanho do lote
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    validation_split=VALIDATION_SPLIT_CF,
    subset="validation",         # Subconjunto de validação
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names                      # Lista de rótulos de classes
AUTOTUNE = tf.data.AUTOTUNE                             # Ajuste automático de desempenho para o pipeline de dados

# 🏎️ Pipeline de treinamento com mapeamento multi-thread
train_ds = train_ds.cache()                             # Cache dos dados em memória para evitar recarregamento a cada época
train_ds = train_ds.shuffle(1000)                       # Embaralhar dados de treinamento para melhor generalização
train_ds = train_ds.map(lambda x, y: (x, y),            # Mapeamento identidade (placeholder para pré-processamento)
                        num_parallel_calls=AUTOTUNE)    # Usar múltiplas threads automaticamente
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)      # Sobrepor pré-processamento e execução do modelo para desempenho

# 🏎️ Pipeline de validação com mapeamento multi-thread
val_ds = val_ds.cache()                                 # Cache dos dados de validação em memória
val_ds = val_ds.map(lambda x, y: (x, y),                # Mapeamento identidade (placeholder para pré-processamento)
                    num_parallel_calls=AUTOTUNE)        # Chamadas paralelas multi-thread
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)          # Prefetch para carregamento eficiente

# 🎨 Aumento de dados (aplicado apenas durante o treinamento)
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),  
    layers.RandomRotation(0.2),                    
    layers.RandomTranslation(0.1, 0.1),            
    layers.RandomZoom(0.2),                        
    layers.RandomContrast(0.2),                    
    layers.RandomBrightness(0.2),                  
    layers.GaussianNoise(0.05)                     
])

# 🧠 Modelo CNN personalizado
model = models.Sequential([
    data_augmentation,                                      
    layers.Rescaling(1./255, input_shape=(IMAGE_SIZE, 3)),  # Normalizar valores de pixels

    layers.Conv2D(32, (3, 3), activation='relu'),           # Primeira camada convolucional
    layers.BatchNormalization(),                            # Normalização das ativações
    layers.MaxPooling2D(2, 2),                              # Redução de dimensionalidade

    layers.Conv2D(64, (3, 3), activation='relu'),           # Segunda camada convolucional
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu'),          # Terceira camada convolucional
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(256, (3,3), activation='relu'),           # Quarta camada convolucional
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(512, (3,3), activation='relu'),           # Quinta camada convolucional
    layers.MaxPooling2D(2, 2),

    layers.GlobalAveragePooling2D(),                        # Pooling global em vez de achatar

    layers.Dense(512, activation='relu',
                 kernel_regularizer=regularizers.l2(1e-4)), # Camada totalmente conectada
    layers.Dropout(0.4),                                    

    layers.Dense(len(class_names), activation='softmax')    # Camada de saída com probabilidades
])


# Ajuste dinâmico da taxa de aprendizado para treinamento
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',   
    factor=0.5,           
    patience=3,           
    min_lr=1e-6,          
    verbose=1             
)

# EarlyStopping para treinamento inicial
early_stop_initial = EarlyStopping(
    monitor='val_loss',           
    patience=10,                  
    restore_best_weights=True     
)

reduce_lr_secondary = ReduceLROnPlateau(
    monitor='val_loss',   
    factor=0.3,           
    patience=2,           
    min_lr=1e-7,          
    verbose=1             
)

early_stop_secondary = EarlyStopping(
    monitor='val_loss',           
    patience=5,                   
    restore_best_weights=True
)    

# Compilar modelo com perda focal em vez de entropia cruzada
model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4),
    loss=focal_loss_multiclass,
    metrics=['accuracy']
)

# Fase inicial de treinamento
model.fit(
    train_ds,                                   
    validation_data=val_ds,                     
    epochs=EPOCHS_INITIAL,                      
    callbacks=[early_stop_initial,              
               reduce_lr]                       
)

current_lr = float(model.optimizer.learning_rate.numpy())

model.compile(
    optimizer=tf.keras.optimizers.Adam(current_lr),  # Recompilar modelo com a taxa de aprendizado atual
    loss=focal_loss_multiclass,                      # Usar função de perda focal
    metrics=['accuracy']                             # Métrica de acurácia
)

# Treino secundário
model.fit(
    train_ds,                       
    validation_data=val_ds,         
    epochs=EPOCHS_FINE_TUNE,        
    callbacks=[early_stop_secondary, 
               reduce_lr_secondary]  
)

# 💾 Salvar modelo treinado
model.save('trash_classifier_model_finetuned.keras')
