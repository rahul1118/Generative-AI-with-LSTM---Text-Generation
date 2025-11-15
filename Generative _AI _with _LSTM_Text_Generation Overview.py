import os
import string
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

# -----------------------------
# Configuration
# -----------------------------
DATA_PATH = "Shakespeare’s_Complete _Works.txt"
SEQ_LENGTH = 40
EMBEDDING_DIM = 128
LSTM_UNITS = 256
BATCH_SIZE = 128
EPOCHS = 20
CHECKPOINT_DIR = "checkpoints"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# -----------------------------
# Load & Preprocess Dataset
# -----------------------------
with open(DATA_PATH, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read().lower()

text = text.translate(str.maketrans("", "", string.punctuation))

chars = sorted(list(set(text)))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for c, i in char_to_idx.items()}
vocab_size = len(chars)

X_data, y_data = [], []

for i in range(len(text) - SEQ_LENGTH):
    seq = text[i:i + SEQ_LENGTH]
    next_char = text[i + SEQ_LENGTH]
    X_data.append([char_to_idx[c] for c in seq])
    y_data.append(char_to_idx[next_char])

X = np.array(X_data)
y = to_categorical(y_data, num_classes=vocab_size)

# -----------------------------
# Train / Validation Split
# -----------------------------
split_index = int(0.9 * len(X))
X_train, X_val = X[:split_index], X[split_index:]
y_train, y_val = y[:split_index], y[split_index:]

# -----------------------------
# Model Architecture
# -----------------------------
model = Sequential([
    Embedding(vocab_size, EMBEDDING_DIM, input_length=SEQ_LENGTH),
    LSTM(LSTM_UNITS),
    Dense(vocab_size, activation="softmax")
])

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

checkpoint_path = os.path.join(CHECKPOINT_DIR, "model_best.h5")

callbacks = [
    ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1),
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
]

# -----------------------------
# Training
# -----------------------------
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks
)

# -----------------------------
# Text Generation Utilities
# -----------------------------
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return np.random.choice(len(preds), p=preds)

def generate_text(seed, length=300, temperature=0.7):
    seed = seed.lower()
    generated = seed
    seq = seed[-SEQ_LENGTH:]
    seq_idx = [char_to_idx[c] for c in seq]
    seq_idx = pad_sequences([seq_idx], maxlen=SEQ_LENGTH)

    for _ in range(length):
        preds = model.predict(seq_idx, verbose=0)[0]
        next_idx = sample(preds, temperature)
        next_char = idx_to_char[next_idx]
        generated += next_char
        seq_idx = np.roll(seq_idx, -1)
        seq_idx[0, -1] = next_idx
    
    return generated

# -----------------------------
# Example Generation
# -----------------------------
seed_text = "to be or not to be that is the question"
print("\n======= GENERATED TEXT =======\n")
print(generate_text(seed_text, length=400, temperature=0.8))
