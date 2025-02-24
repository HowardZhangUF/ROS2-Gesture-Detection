import os
import numpy as np
import collections
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# 1) DATA LOADING & AUGMENTATION
# ----------------------------
DATA_DIR = "/home/bakerherrin/ROS2-Gesture-Detection/src/landmark_data_holistic"
actions = sorted(os.listdir(DATA_DIR))  # Action categories (folders)

action_to_label = {action: i for i, action in enumerate(actions)}
X, y = [], []
sequence_lengths = []

# Load .npy files
for action in actions:
    action_folder = os.path.join(DATA_DIR, action)
    if os.path.isdir(action_folder):
        for npy_file in os.listdir(action_folder):
            if npy_file.endswith('.npy'):
                npy_path = os.path.join(action_folder, npy_file)
                sequence = np.load(npy_path)

                # Flatten extra dims if needed
                if len(sequence.shape) > 2:
                    sequence = sequence.reshape(sequence.shape[0], -1)

                sequence_lengths.append(sequence.shape[0])
                X.append(sequence)
                y.append(action_to_label[action])

NUM_CLASSES = len(actions)
if not sequence_lengths:
    raise ValueError("No valid .npy sequences found. Check if the dataset is empty!")

MAX_SEQ_LEN = max(sequence_lengths)
print(f"Max Sequence Length: {MAX_SEQ_LEN}")

# Pad sequences manually
X_padded = []
for seq in X:
    seq_length = seq.shape[0]
    feature_dim = seq.shape[1]
    if seq_length < MAX_SEQ_LEN:
        pad_amount = MAX_SEQ_LEN - seq_length
        padded_seq = np.concatenate([seq, np.zeros((pad_amount, feature_dim))], axis=0)
    else:
        padded_seq = seq[:MAX_SEQ_LEN, :]
    X_padded.append(padded_seq)

X_padded = np.array(X_padded)
y = np.array(y)
print(f"Shape of X_padded: {X_padded.shape}, y shape: {y.shape}")

# Check class distribution
class_counts = collections.Counter(y)
print(f"Class distribution before augmentation: {class_counts}")

# Scale to [-1, 1]
N, S, F = X_padded.shape
scaler = MinMaxScaler(feature_range=(-1, 1))
X_reshaped = X_padded.reshape(N*S, F)
X_scaled = scaler.fit_transform(X_reshaped).reshape(N, S, F)
X_padded = X_scaled

# Convert y to one-hot (for augmentation logic)
y_onehot = np.eye(NUM_CLASSES)[y]

def augment_sequence(sequence, jitter_strength=0.05, time_warp_strength=0.2):
    """Jitter + Scaling + Time Warping."""
    # Jitter
    jitter = np.random.normal(loc=0, scale=jitter_strength, size=sequence.shape)
    seq_jittered = sequence + jitter

    # Scaling
    scale_factor = np.random.uniform(0.9, 1.1)
    seq_scaled = seq_jittered * scale_factor

    # Time Warping
    seq_length = seq_scaled.shape[0]
    warp_amount = int(seq_length * time_warp_strength)
    warp_indices = np.random.choice(seq_length, size=warp_amount, replace=False)
    seq_warped = seq_scaled.copy()
    for idx in warp_indices:
        if random.random() > 0.5:
            seq_warped[idx] *= 1.1
        else:
            seq_warped[idx] *= 0.9
    return seq_warped

# Augment underrepresented classes
X_augmented, y_augmented = [], []
min_samples_threshold = 10
for i in range(len(X_padded)):
    seq = X_padded[i]
    label_index = np.argmax(y_onehot[i])
    if class_counts[label_index] < min_samples_threshold:
        for _ in range(5):
            X_augmented.append(augment_sequence(seq))
            y_augmented.append(label_index)

X_augmented = np.array(X_augmented)
y_augmented = np.array(y_augmented)
y_augmented_onehot = np.eye(NUM_CLASSES)[y_augmented]

X_padded = np.concatenate((X_padded, X_augmented), axis=0)
y_onehot = np.concatenate((y_onehot, y_augmented_onehot), axis=0)

# Augment entire dataset 10×
X_extra_augmented, y_extra_augmented = [], []
for i in range(len(X_padded)):
    seq = X_padded[i]
    label_index = np.argmax(y_onehot[i])
    for _ in range(10):
        X_extra_augmented.append(augment_sequence(seq))
        y_extra_augmented.append(label_index)

X_extra_augmented = np.array(X_extra_augmented)
y_extra_augmented = np.array(y_extra_augmented)
y_extra_augmented_onehot = np.eye(NUM_CLASSES)[y_extra_augmented]

X_padded = np.concatenate((X_padded, X_extra_augmented), axis=0)
y_onehot = np.concatenate((y_onehot, y_extra_augmented_onehot), axis=0)

final_labels = np.argmax(y_onehot, axis=1)
aug_class_counts = collections.Counter(final_labels)
print(f"Class distribution after full augmentation: {aug_class_counts}")

# Train/test split
X_train, X_test, y_train_onehot, y_test_onehot = train_test_split(
    X_padded, y_onehot, test_size=0.2, random_state=42
)

y_train = np.argmax(y_train_onehot, axis=1)
y_test = np.argmax(y_test_onehot, axis=1)

print(f"Loaded {len(X_padded)} samples, {len(actions)} action classes.")
print(f"Training Samples: {len(X_train)}, Testing Samples: {len(X_test)}")

# ----------------------------
# 2) DEFINE A PYTORCH TRANSFORMER FOR CLASSIFICATION
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            -torch.arange(0, d_model, 2).float() * (np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :].to(x.device)
        return x

class ActionTransformer(nn.Module):
    def __init__(self, feature_dim, num_classes, embed_dim=64, num_heads=4, ff_dim=128, dropout=0.1):
        super(ActionTransformer, self).__init__()
        self.input_proj = nn.Linear(feature_dim, embed_dim)
        self.pos_encoding = PositionalEncoding(d_model=embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        x = self.transformer_encoder(x)
        x = torch.mean(x, dim=1)  # global average pooling
        x = self.dropout(x)
        logits = self.fc_out(x)
        return logits

# ----------------------------
# 3) PREPARE DATA FOR PYTORCH
# ----------------------------
X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)

X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)

# ----------------------------
# 4) INITIALIZE MODEL, OPTIMIZER, LOSS
# ----------------------------
model = ActionTransformer(
    feature_dim=X_train.shape[-1],
    num_classes=NUM_CLASSES,
    embed_dim=64,
    num_heads=4,
    ff_dim=128,
    dropout=0.1
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print(model)
print(f"Using device: {device}")

# ----------------------------
# 5) TRAINING LOOP
# ----------------------------
def evaluate_loss_and_accuracy(model, X_data, y_data):
    """
    Compute loss & accuracy over the entire dataset X_data, y_data.
    Returns (loss_value, accuracy_value).
    """
    model.eval()
    with torch.no_grad():
        outputs = model(X_data)
        loss_val = criterion(outputs, y_data).item()
        _, preds = torch.max(outputs, dim=1)
        correct = (preds == y_data).float().sum().item()
        accuracy_val = correct / y_data.shape[0]
    return loss_val, accuracy_val

def train_model(model, criterion, optimizer, X_train, y_train, X_test, y_test,
                epochs=5, batch_size=32):
    """
    Train and store metrics each epoch for plotting.
    """
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    dataset_size = X_train.shape[0]
    num_batches = int(np.ceil(dataset_size / batch_size))

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(dataset_size)
        epoch_loss = 0.0

        for i in range(num_batches):
            idx = perm[i*batch_size : (i+1)*batch_size]
            X_batch = X_train[idx]
            y_batch = y_train[idx]

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # End of epoch: compute average train loss
        epoch_loss /= num_batches

        # ---- Evaluate on full train set & test set ----
        train_loss, train_acc = evaluate_loss_and_accuracy(model, X_train, y_train)
        test_loss, test_acc = evaluate_loss_and_accuracy(model, X_test, y_test)

        # Store metrics
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{epochs}] | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    return train_losses, test_losses, train_accs, test_accs

# ----------------------------
# 6) RUN TRAINING
# ----------------------------
train_losses, test_losses, train_accs, test_accs = train_model(
    model, criterion, optimizer,
    X_train_t, y_train_t,
    X_test_t, y_test_t,
    epochs=5,
    batch_size=32
)
print("Training complete!")

# Save the PyTorch model
torch.save(model.state_dict(), "transformer_action_recognition.pth")
print("Model saved to transformer_action_recognition.pth")

# ----------------------------
# 7) PLOT LOSS & ACCURACY CURVES
# ----------------------------
plt.figure(figsize=(12, 5))

# -- Loss Curves --
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# -- Accuracy Curves --
plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Accuracy')
plt.plot(test_accs, label='Test Accuracy')
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ----------------------------
# 8) FINAL EVALUATION (CONFUSION MATRIX)
# ----------------------------
model.eval()
with torch.no_grad():
    outputs = model(X_test_t)
    _, preds = torch.max(outputs, dim=1)

preds_np = preds.cpu().numpy()
y_true_np = y_test_t.cpu().numpy()

conf_matrix = confusion_matrix(y_true_np, preds_np, normalize="true")

action_labels = sorted(os.listdir(DATA_DIR))
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt=".2f",
            xticklabels=action_labels, yticklabels=action_labels)
plt.title("Confusion Matrix Heatmap (PyTorch Transformer)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

acc = accuracy_score(y_true_np, preds_np)
print("Confusion Matrix:\n", conf_matrix)
print(f"Model Accuracy: {acc:.4f}")
