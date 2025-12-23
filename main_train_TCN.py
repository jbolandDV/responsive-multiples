import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import ntpath
from tensorflow.keras.layers import Input, Dense, MultiHeadAttention, LayerNormalization, Dropout
from tensorflow.keras.models import Model, load_model
from tcn import TCN
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import time
import logging
from contextlib import redirect_stdout
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import json
from tensorflow.keras.models import load_model
import os, json, time, pickle, logging
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Dense, MultiHeadAttention, LayerNormalization, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

start_time = time.time()

# ---------------------------- resolve custom_folder_path ----------------------------
current_dir = os.path.dirname(__file__)
files_dir = os.path.join(current_dir, "Files")
path_file = os.path.join(files_dir, "custom_folder_path.txt")

def _read_first_path_line(txt_path: str) -> str:
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Missing {txt_path}")
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                return s
    raise RuntimeError(f"{txt_path} exists but has no non-empty lines.")

# 1) Env override (kept)
custom_folder_path = os.environ.get("CUSTOM_DATA_DIR")

if not custom_folder_path:
    # 2) Read Windows absolute path from first line
    source_path = _read_first_path_line(path_file)

    # Normalize: handle quotes, trailing slashes, and Windows backslashes
    source_path = source_path.strip().strip('"').strip("'").rstrip("\\/")

    # Use ntpath so Windows-style paths work on Linux too
    base = ntpath.basename(source_path)      # <-- key change
    # If base still looks like a drive-letter path (edge cases), fallback:
    if base.lower().endswith(":"):
        base = ""
    if not base:
        # as a safety, try splitdrive then basename of the remainder
        drive, tail = ntpath.splitdrive(source_path)
        base = ntpath.basename(tail)

    candidate_local = os.path.join(files_dir, base)  # ./Files/<folder_name> (mounted)

    if os.path.isdir(candidate_local):
        custom_folder_path = candidate_local
    elif os.path.isdir(source_path):
        # Running directly on Windows (no container)
        custom_folder_path = source_path
    else:
        raise FileNotFoundError(
            "Could not resolve data directory.\n"
            f"- Tried local: {candidate_local}\n"
            f"- Tried source: {source_path}\n"
            "Ensure your data folder exists under ./Files/ (mounted) or set CUSTOM_DATA_DIR."
        )

print(f"\nLoading the datasets from {custom_folder_path}\n")

# Ensure output folders exist at the resolved path
images_folder = os.path.join(custom_folder_path, "Images")
os.makedirs(images_folder, exist_ok=True)
classifier_folder = os.path.join(custom_folder_path, "Classifier Model")
os.makedirs(classifier_folder, exist_ok=True)

# ---------------------------- logging ----------------------------
log_file_path = os.path.join(classifier_folder, "Training_Testing_Results.txt")
with open(log_file_path, "w", encoding="utf-8") as f:
    f.write("")
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(log_file_path, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# ---------------------------- load config & datasets ----------------------------
config_file_path = os.path.join(custom_folder_path, "config.json")
with open(config_file_path, "r", encoding="utf-8") as f:
    config = json.load(f)

time_step = config["time_step"]
transaction_dataset_name = config["transaction_dataset_name"]
promo_dataset_name = config["promo_dataset_name"]
filter_unique_ids = config["filter_unique_ids"]
cut_off_date = config["cut_off_date"]
test_size = config["test_size"]
model_window_size = config["model_window_size"]
exclude_one_time_donors = config["exclude_one_time_donors"]
decay_coefficient = config["decay_coefficient"]

X_train_path = os.path.join(custom_folder_path, "X_train.pkl")
y_train_path = os.path.join(custom_folder_path, "y_train.pkl")
X_test_path  = os.path.join(custom_folder_path, "X_test.pkl")
y_test_path  = os.path.join(custom_folder_path, "y_test.pkl")

with open(X_train_path, "rb") as f:
    X_train_df = pickle.load(f)
with open(y_train_path, "rb") as f:
    y_train_df = pickle.load(f)
with open(X_test_path, "rb") as f:
    X_test_df = pickle.load(f)
with open(y_test_path, "rb") as f:
    y_test_df = pickle.load(f)

# Stack arrays
X_train = np.stack([df.values for df in X_train_df])   # (N, T, 7)
y_train = np.stack([s.values for s in y_train_df])     # (N, T)
X_test  = np.stack([df.values for df in X_test_df])    # (N, T, 7)
y_test  = np.stack([s.values for s in y_test_df])      # (N, T)

# ---------------------------- padded indices ----------------------------
padded_rows_indices = np.array([])
for i in range(X_test.shape[0]):
    for j in range(X_test.shape[1]):
        if X_test[i][j][-1] == 1:
            padded_rows_indices = np.append(padded_rows_indices, i * model_window_size + j)
padded_rows_indices = padded_rows_indices.astype(int)

padded_rows_indices_train = np.array([])
for i in range(X_train.shape[0]):
    for j in range(X_train.shape[1]):
        if X_train[i][j][-1] == 1:
            padded_rows_indices_train = np.append(padded_rows_indices_train, i * model_window_size + j)
padded_rows_indices_train = padded_rows_indices_train.astype(int)

print(f"\n{padded_rows_indices_train.shape[0]} padded rows found in the train set out of "
      f"{X_train.shape[0] * model_window_size} rows "
      f"({padded_rows_indices_train.shape[0]/(X_train.shape[0] * model_window_size)*100:.2f}%).\n")
print(f"{padded_rows_indices.shape[0]} padded rows found in the test set out of "
      f"{X_test.shape[0] * model_window_size} rows "
      f"({padded_rows_indices.shape[0]/(X_test.shape[0] * model_window_size)*100:.2f}%).\n")

# ---------------------------- features ----------------------------
def build_lagged_labels(X):
    """
    X: (N, T, 7). Add column: 1 when Recency==0 and is_padded==0, else 0. -> (N, T, 8)
    """
    N, T, _ = X.shape
    assert T == model_window_size, "Expecting model_window_size-sized windows"
    new_column = ((X[:, :, 2] == 0) & (X[:, :, 6] == 0)).astype("float32")
    return np.concatenate([X, new_column[..., np.newaxis]], axis=-1)

w0 = 1.0
w1 = (y_train == 0).sum() / (y_train == 1).sum()
sample_weight = np.where(y_train == 1, w1, w0).astype(np.float32)

y_train = y_train[..., np.newaxis]  # (N, T, 1)
y_test  = y_test[...,  np.newaxis]  # (N, T, 1)
X_train = build_lagged_labels(X_train)  # (N, T, 8)
X_test  = build_lagged_labels(X_test)
sample_weight = sample_weight[..., np.newaxis]  # (N, T, 1)

print(f"Training label set's shape: {y_train.shape}")
print(f"Testing  label set's shape: {y_test.shape}")
print(f"Training set's shape: {X_train.shape}")
print(f"Test set's shape: {X_test.shape}")
print(f"sample_weight.shape: {sample_weight.shape}")
print(f"Penalty weight for class 0: {w0}")
print(f"Penalty weight for class 1: {w1}\n")

model_save_path = os.path.join(classifier_folder, "RFM_TCN")

# ---------------------------- model ----------------------------
inp = Input(shape=(model_window_size, 8), name="features")
x = TCN(
    nb_filters=300,
    kernel_size=3,
    dilations=[1, 2, 4, 8],
    use_skip_connections=True,
    return_sequences=True
)(inp)

attn = MultiHeadAttention(num_heads=10, key_dim=64, dropout=0.1)(x, x, use_causal_mask=True)
x = LayerNormalization(epsilon=1e-6)(x + attn)  # this will serialize as an Add layer in Keras 2.x
x = Dropout(0.1)(x)
x = Dense(100, activation='relu')(x)
out = Dense(1, activation='sigmoid')(x)

model = Model(inp, out)
model.compile(optimizer=Adam(5.5e-4), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

print("\nTraining the TCN model...\n")

# ---------------------------- callbacks (defined for future use) ----------------------------
lr_cb = ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-5, verbose=1, monitor="val_loss")
es_cb = EarlyStopping(patience=15, restore_best_weights=True, monitor="val_loss", verbose=1)
mc_cb = ModelCheckpoint(
    filepath=model_save_path,          # CHANGED: .h5
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=False,           # CHANGED: save full model
    verbose=1
)

history = model.fit(
    x=X_train,
    y=y_train,
    sample_weight=sample_weight,
    validation_data=(X_test, y_test),
    epochs=300,
    batch_size=32,
    verbose=2,
    shuffle=True,
    callbacks=[mc_cb]
)

# ---------------------------- load best & plots ----------------------------
model = load_model(model_save_path, custom_objects={"TCN": TCN}, compile=False)
print(f"\nBest trained model loaded from {model_save_path}\n")

# Accuracy
train_acc = history.history.get('accuracy', [])
val_acc = history.history.get('val_accuracy', [])
epochs = range(1, len(train_acc) + 1)
plt.figure(figsize=(25, 5))
plt.plot(epochs, train_acc, label='Training Accuracy')
plt.plot(epochs, val_acc, label='Validation Accuracy')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Accuracy per Epoch')
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(os.path.join(images_folder, "accuracy_per_epoch.svg"), format='svg')
plt.close()

# Loss
train_loss = history.history.get('loss', [])
val_loss = history.history.get('val_loss', [])
plt.figure(figsize=(25, 5))
plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss per Epoch')
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(os.path.join(images_folder, "loss_per_epoch.svg"), format='svg')
plt.close()

print("\nTraining and validation accuracy and loss plots saved successfully!\n")

# ---------------------------- metrics & logs ----------------------------
logging.info("------------------------------------------------------------")
logging.info("                     Accuracy Results                       ")
logging.info("------------------------------------------------------------")

y_true = y_test.flatten()
y_true_train = y_train.flatten()
y_pred_probs = model.predict(X_test).flatten()
y_pred_probs_train = model.predict(X_train).flatten()
y_pred = (y_pred_probs > 0.5).astype(int)
y_pred_train = (y_pred_probs_train > 0.5).astype(int)

score_train_total = balanced_accuracy_score(y_true_train, y_pred_train)
logging.info("***Total*** Train Balanced Accuracy: %s\n", score_train_total)
score_total = balanced_accuracy_score(y_true, y_pred)
logging.info("***Total*** Test Balanced Accuracy: %s\n\n", score_total)

y_true_no_padded = np.delete(y_true, padded_rows_indices.astype(int))
y_true_no_padded_train = np.delete(y_true_train, padded_rows_indices_train.astype(int))
y_pred_probs_no_padded = np.delete(y_pred_probs, padded_rows_indices.astype(int))
y_pred_probs_no_padded_train = np.delete(y_pred_probs_train, padded_rows_indices_train.astype(int))
y_pred_no_padded = (y_pred_probs_no_padded > 0.5).astype(int)
y_pred_no_padded_train = (y_pred_probs_no_padded_train > 0.5).astype(int)

score_train_np = balanced_accuracy_score(y_true_no_padded_train, y_pred_no_padded_train)
score_np = balanced_accuracy_score(y_true_no_padded, y_pred_no_padded)
logging.info("Train ***Non-Padded*** Balanced Accuracy: %s\n", score_train_np)
logging.info("Test ***Non-Padded*** Balanced Accuracy: %s\n\n", score_np)

conf_matrix_train = confusion_matrix(y_true_no_padded_train, y_pred_no_padded_train)
logging.info("***Non-Padded*** Train Confusion Matrix:\n%s\n", conf_matrix_train)
conf_matrix = confusion_matrix(y_true_no_padded, y_pred_no_padded)
logging.info("***Non-Padded*** Test Confusion Matrix:\n%s\n\n", conf_matrix)

y_true_padded = y_true[padded_rows_indices.astype(int)]
y_pred_probs_padded = y_pred_probs[padded_rows_indices.astype(int)]
y_pred_padded = (y_pred_probs_padded > 0.5).astype(int)

y_true_padded_train = y_true_train[padded_rows_indices_train.astype(int)]
y_pred_probs_padded_train = y_pred_probs_train[padded_rows_indices_train.astype(int)]
y_pred_padded_train = (y_pred_probs_padded_train > 0.5).astype(int)

score_train_pad = balanced_accuracy_score(y_true_padded_train, y_pred_padded_train)
logging.info("Train ***Padded*** Balanced Accuracy: %s\n", score_train_pad)
score_pad = balanced_accuracy_score(y_true_padded, y_pred_padded)
logging.info("Test ***Padded*** Balanced Accuracy: %s\n\n", score_pad)

# Sample classification plots
if X_test.shape[0] < 1000:
    step = max(1, X_test.shape[0] // 10)
    range_plot = range(0, X_test.shape[0], step)
else:
    step = 1000
    range_plot = range(0, X_test.shape[0], step)

i = 0
flat_len = X_test.shape[0] * model_window_size
for idx in range_plot:
    i += 1
    ran = range(idx, min(idx + step, flat_len))
    y_true_plot = y_true[ran]
    y_pred_plot = y_pred_probs[ran]

    padded_indices_in_range = [k - idx for k in ran if k in padded_rows_indices]
    sections = []
    if padded_indices_in_range:
        start = padded_indices_in_range[0]
        for ii in range(1, len(padded_indices_in_range)):
            if padded_indices_in_range[ii] != padded_indices_in_range[ii - 1] + 1:
                sections.append((start, padded_indices_in_range[ii - 1]))
                start = padded_indices_in_range[ii]
        sections.append((start, padded_indices_in_range[-1]))

    plt.figure(figsize=(28, 8))
    plt.scatter(range(len(y_pred_plot)), y_pred_plot, alpha=0.6, label='Predicted Probabilities (y_pred)', color='red', s=10)
    plt.scatter(range(len(y_true_plot)), y_true_plot, alpha=0.6, label='True Labels (y_test)', color='blue', s=10)
    plt.axhline(y=0.5, color='black', linestyle='--', label='Threshold (0.5)')
    for start, end in sections:
        plt.axvspan(start, end, color='green', alpha=0.2, label='_nolegend_')
    plt.title('True Labels vs Predicted Probabilities')
    plt.xlabel('Sample Index'); plt.ylabel('Value'); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(images_folder, f"sample_test_classifications_{i}.svg"), format='svg')
    plt.close()

print("\nTest set's sample classification plots saved successfully!\n")

# Append model code + summary to the log
training_code_path = __file__
with open(training_code_path, "r", encoding="utf-8") as script_file:
    model_code = script_file.read()
with open(log_file_path, "a", encoding="utf-8") as f:
    f.write("\n------------------------------------------------------------\n")
    f.write("                     Model Architecture                     \n")
    f.write("------------------------------------------------------------\n")
    f.write(model_code)
    f.write("\n------------------------------------------------------------\n")
with open(log_file_path, "a", encoding="utf-8") as f:
    with redirect_stdout(f):
        model.summary()

end_time = time.time()
print(f"\nTotal time taken: {(end_time - start_time)/60:.2f} mins.\n")
