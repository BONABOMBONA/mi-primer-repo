# -*- coding: utf-8 -*-
# ===========================================================================
#  NUCLEO COMUN: utilidades del clasificador de noticias falsas con TextCNN.
#  (clean_text, vocabulario, GloVe, Dataset, TextCNN, entrenamiento, metricas)
# ===========================================================================
import os
import re
import math
import copy
import warnings
from collections import Counter

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # backend sin ventana: no bloquea al correr como script
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuracion editable
# ---------------------------------------------------------------------------
# El profesor coloca su CSV (con columnas title, text, label) en esta ruta,
# o simplemente lo deja como "WELFake_Dataset.csv" junto a este main.py.
DATA_PATH = "WELFake_Dataset.csv"
MAX_VOCAB = 20000
MAX_LEN = 200
BATCH_SIZE = 64

PAD_IDX, UNK_IDX = 0, 1
PAD_TOKEN, UNK_TOKEN = "<PAD>", "<UNK>"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_URL_RE = re.compile(r"http\S+|www\.\S+")
_HTML_RE = re.compile(r"<[^>]+>")
_NON_ALPHA_RE = re.compile(r"[^a-z\s]")
_MULTISPACE_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Limpieza de texto
# ---------------------------------------------------------------------------
def clean_text(text):
    """Minusculas, sin URLs/HTML/numeros; robusto ante None/NaN."""
    if text is None:
        return ""
    if isinstance(text, float):
        if math.isnan(text):
            return ""
        text = str(text)
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)
    text = _NON_ALPHA_RE.sub(" ", text)
    text = _MULTISPACE_RE.sub(" ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Vocabulario y GloVe
# ---------------------------------------------------------------------------
def build_vocabulary(texts, max_vocab=MAX_VOCAB):
    counter = Counter()
    for t in texts:
        if isinstance(t, str) and t:
            counter.update(t.split())
    word2idx = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    for word, _ in counter.most_common(max(0, int(max_vocab) - len(word2idx))):
        if word not in word2idx:
            word2idx[word] = len(word2idx)
    return word2idx


def load_glove(glove_path, word2idx, embed_dim):
    vocab_size = len(word2idx)
    rng = np.random.default_rng(42)
    mat = rng.normal(scale=0.6, size=(vocab_size, embed_dim)).astype(np.float32)
    mat[PAD_IDX] = 0.0
    if not glove_path or not os.path.exists(glove_path):
        return mat
    found = 0
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) != embed_dim + 1:
                continue
            idx = word2idx.get(parts[0])
            if idx is None:
                continue
            try:
                mat[idx] = np.asarray(parts[1:], dtype=np.float32)
                found += 1
            except ValueError:
                continue
    print(f"[GloVe] Cobertura: {found}/{vocab_size} ({found/max(vocab_size,1)*100:.1f}%)")
    return mat


# ---------------------------------------------------------------------------
# Dataset y DataLoaders
# ---------------------------------------------------------------------------
class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, word2idx, max_len=MAX_LEN):
        self.w2i = word2idx
        self.max_len = int(max_len)
        self.pad = word2idx.get(PAD_TOKEN, PAD_IDX)
        self.unk = word2idx.get(UNK_TOKEN, UNK_IDX)
        self.seqs = [self._encode(t) for t in texts]
        self.labels = [int(l) for l in labels]

    def _encode(self, text):
        if not isinstance(text, str):
            text = "" if text is None else str(text)
        ids = [self.w2i.get(w, self.unk) for w in text.split()][: self.max_len]
        if len(ids) < self.max_len:
            ids += [self.pad] * (self.max_len - len(ids))
        return ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return (torch.tensor(self.seqs[i], dtype=torch.long),
                torch.tensor(self.labels[i], dtype=torch.long))


def create_dataloaders(train, val, test, word2idx, batch_size=BATCH_SIZE, max_len=MAX_LEN):
    tr = FakeNewsDataset(train[0], train[1], word2idx, max_len)
    va = FakeNewsDataset(val[0], val[1], word2idx, max_len)
    te = FakeNewsDataset(test[0], test[1], word2idx, max_len)
    return (DataLoader(tr, batch_size=batch_size, shuffle=True),
            DataLoader(va, batch_size=batch_size, shuffle=False),
            DataLoader(te, batch_size=batch_size, shuffle=False))


# ---------------------------------------------------------------------------
# Modelo TextCNN
# ---------------------------------------------------------------------------
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, num_filters=100,
                 kernel_sizes=(3, 4, 5), dropout=0.5, num_classes=1,
                 pretrained_embeddings=None):
        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes]
        kernel_sizes = list(kernel_sizes)
        self._max_kernel = max(kernel_sizes)

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            w = torch.as_tensor(np.asarray(pretrained_embeddings), dtype=torch.float32)
            if tuple(w.shape) == (vocab_size, embed_dim):
                self.embedding.weight.data.copy_(w)
            with torch.no_grad():
                self.embedding.weight[0].fill_(0.0)

        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        emb = self.embedding(x).permute(0, 2, 1)           # (B, E, L)
        if emb.size(2) < self._max_kernel:                 # seguridad si L < kernel
            emb = F.pad(emb, (0, self._max_kernel - emb.size(2)))
        conv_outputs = [torch.relu(c(emb)).max(dim=2)[0] for c in self.convs]
        out = self.dropout(torch.cat(conv_outputs, dim=1))
        logits = self.fc(out)
        if logits.size(-1) == 1:
            logits = logits.squeeze(-1)
        return logits


# ---------------------------------------------------------------------------
# Entrenamiento y evaluacion
# ---------------------------------------------------------------------------
def _epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    tot, n, yt, yp = 0.0, 0, [], []
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).view(-1).float()
            if is_train:
                optimizer.zero_grad()
            logits = model(x).view(-1)
            loss = criterion(logits, y)
            if is_train:
                loss.backward()
                optimizer.step()
            tot += loss.item() * y.size(0)
            n += y.size(0)
            yt.extend(y.long().cpu().tolist())
            yp.extend((torch.sigmoid(logits) >= 0.5).long().cpu().tolist())
    f1 = f1_score(yt, yp, zero_division=0) if yt else 0.0
    return tot / max(n, 1), f1


def train_model(model, train_loader, val_loader, optimizer, criterion, device,
                num_epochs=10, patience=3):
    model.to(device)
    history = {"train_loss": [], "train_f1": [], "val_loss": [], "val_f1": []}
    best_f1, best_state, no_improve = -1.0, copy.deepcopy(model.state_dict()), 0
    for ep in range(1, num_epochs + 1):
        tl, tf = _epoch(model, train_loader, criterion, device, optimizer)
        vl, vf = _epoch(model, val_loader, criterion, device, None)
        history["train_loss"].append(tl); history["train_f1"].append(tf)
        history["val_loss"].append(vl); history["val_f1"].append(vf)
        print(f"  Epoca {ep:2d}/{num_epochs} | train_loss={tl:.4f} f1={tf:.4f}"
              f" | val_loss={vl:.4f} f1={vf:.4f}")
        if vf > best_f1:
            best_f1, best_state, no_improve = vf, copy.deepcopy(model.state_dict()), 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping en epoca {ep}.")
                break
    model.load_state_dict(best_state)
    return history


def train_proxy(model, train_loader, val_loader, device, epochs=3,
                data_fraction=0.3, lr=1e-3):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    n_batches = max(1, int(len(train_loader) * data_fraction))
    best = 0.0
    for _ in range(epochs):
        model.train()
        for i, (x, y) in enumerate(train_loader):
            if i >= n_batches:
                break
            x = x.to(device); y = y.to(device).view(-1).float()
            optimizer.zero_grad()
            loss = criterion(model(x).view(-1), y)
            loss.backward(); optimizer.step()
        _, vf = _epoch(model, val_loader, criterion, device, None)
        best = max(best, vf)
    return best


def evaluate_model(model, loader, criterion, device):
    model.to(device); model.eval()
    tot, n, yt, yp = 0.0, 0, [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device); y = y.to(device).view(-1).float()
            logits = model(x).view(-1)
            tot += criterion(logits, y).item() * y.size(0); n += y.size(0)
            yt.extend(y.long().cpu().tolist())
            yp.extend((torch.sigmoid(logits) >= 0.5).long().cpu().tolist())
    if not yt:
        return {"loss": 0.0, "accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    return {"loss": tot / max(n, 1),
            "accuracy": float(accuracy_score(yt, yp)),
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall": float(recall_score(yt, yp, zero_division=0))}


def get_predictions(model, loader, device):
    model.to(device); model.eval()
    yt, yp, ypb = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            proba = torch.sigmoid(model(x).view(-1))
            yt.extend(y.view(-1).long().cpu().tolist())
            yp.extend((proba >= 0.5).long().cpu().tolist())
            ypb.extend(proba.cpu().tolist())
    return np.array(yt), np.array(yp), np.array(ypb)


# ---------------------------------------------------------------------------
# Graficas (se GUARDAN como PNG; no abren ventana)
# ---------------------------------------------------------------------------
def save_confusion_matrix(y_true, y_pred, path, labels=("Real", "Fake")):
    labels = list(labels)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues"); fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Prediccion"); ax.set_ylabel("Real"); ax.set_title("Matriz de Confusion")
    th = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > th else "black")
    plt.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    print(f"  [figura] {path}")


def save_training_curves(history, path):
    ep = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot(ep, history["train_loss"], "o-", label="Train"); ax[0].plot(ep, history["val_loss"], "o--", label="Val")
    ax[0].set_title("Perdida"); ax[0].set_xlabel("Epoca"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(ep, history["train_f1"], "o-", label="Train"); ax[1].plot(ep, history["val_f1"], "o--", label="Val")
    ax[1].set_title("F1-Score"); ax[1].set_xlabel("Epoca"); ax[1].legend(); ax[1].grid(alpha=0.3)
    plt.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    print(f"  [figura] {path}")


# ---------------------------------------------------------------------------
# Carga de datos (real o sintetica de respaldo) + pipeline completo
# ---------------------------------------------------------------------------
def make_synthetic(n=1000, seed=0):
    """Genera un dataset de prueba si no hay CSV, para que el script SIEMPRE corra."""
    rng = np.random.default_rng(seed)
    fake = "shocking secret exposed conspiracy miracle hoax click outrage banned breaking".split()
    real = "according report officials study confirmed reuters announced policy committee economy".split()
    fill = "the a of to and in for on with as by from that this it they we you".split()
    titles, texts, labels = [], [], []
    for i in range(n):
        lab = int(rng.integers(0, 2))
        pool = fake if lab == 1 else real
        body = " ".join(rng.choice(pool + fill, size=int(rng.integers(20, 120))))
        title = "" if i % 50 == 0 else " ".join(rng.choice(pool, size=int(rng.integers(3, 9))))
        titles.append(title); texts.append(body); labels.append(lab)
    return pd.DataFrame({"title": titles, "text": texts, "label": labels})


def load_dataframe():
    candidates = [DATA_PATH, "WELFake_Dataset.csv",
                  os.path.join("..", "proyecto_semana6", "data", "raw", "WELFake_Dataset.csv"),
                  os.path.join("data", "WELFake_Dataset.csv")]
    for p in candidates:
        if os.path.exists(p):
            print(f"[datos] Cargando CSV real: {p}")
            return pd.read_csv(p)
    print("[datos] No se hallo WELFake_Dataset.csv -> se usan datos sinteticos (1000 filas).")
    return make_synthetic(1000)


def prepare_data(embed_dim=50, max_vocab=MAX_VOCAB, max_len=MAX_LEN, batch_size=BATCH_SIZE):
    """Carga, limpia, divide, vectoriza y devuelve todo lo necesario para entrenar."""
    df = load_dataframe()
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    for col in ("title", "text"):
        if col not in df.columns:
            df[col] = ""
    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")
    if "label" not in df.columns:
        raise ValueError("El CSV debe tener una columna 'label' (0=real, 1=fake).")
    df["label"] = df["label"].astype(int)
    df = df.drop_duplicates(subset=["title", "text"], keep="first").reset_index(drop=True)
    df["text_clean"] = df["text"].apply(clean_text)

    train_df, temp = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    val_df, test_df = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp["label"])

    word2idx = build_vocabulary(train_df["text_clean"].tolist(), max_vocab=max_vocab)
    idx2word = {v: k for k, v in word2idx.items()}

    glove_file = f"glove.6B.{embed_dim}d.txt"
    emb = load_glove(glove_file, word2idx, embed_dim) if os.path.exists(glove_file) else None

    loaders = create_dataloaders(
        (train_df["text_clean"].tolist(), train_df["label"].tolist()),
        (val_df["text_clean"].tolist(), val_df["label"].tolist()),
        (test_df["text_clean"].tolist(), test_df["label"].tolist()),
        word2idx, batch_size=batch_size, max_len=max_len)

    print(f"[datos] Vocab={len(word2idx)} | Train={len(train_df)} Val={len(val_df)} Test={len(test_df)}"
          f" | embeddings={'GloVe' if emb is not None else 'aleatorios'} | device={DEVICE}")

    return {"df": df, "train_df": train_df, "val_df": val_df, "test_df": test_df,
            "word2idx": word2idx, "idx2word": idx2word, "vocab_size": len(word2idx),
            "emb": emb, "loaders": loaders, "device": DEVICE}


# ===========================================================================
#  RETO 10 — Analisis Final, Visualizacion e Interpretabilidad
#  Entrena el mejor TextCNN, visualiza embeddings (t-SNE/PCA), identifica las
#  palabras mas influyentes (gradientes) y arma la tabla comparativa final.
# ===========================================================================
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

EMBED_DIM = 50
NUM_FILTERS = 100
KERNEL_SIZES = [3, 4, 5]
DROPOUT = 0.5
LR = 1e-3
NUM_EPOCHS = 10
PATIENCE = 3


def run():
    data = prepare_data(embed_dim=EMBED_DIM)
    train_loader, val_loader, test_loader = data["loaders"]
    V, emb, device = data["vocab_size"], data["emb"], data["device"]
    word2idx, idx2word = data["word2idx"], data["idx2word"]

    # ---- Entrenar modelo ----
    print("\n=== Entrenando TextCNN ===")
    torch.manual_seed(42)
    model = TextCNN(V, EMBED_DIM, NUM_FILTERS, KERNEL_SIZES, DROPOUT,
                    pretrained_embeddings=emb).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.BCEWithLogitsLoss()
    train_model(model, train_loader, val_loader, opt, crit, device, NUM_EPOCHS, PATIENCE)
    tm = evaluate_model(model, test_loader, crit, device)

    # ---- Tabla comparativa final ----
    final_table = pd.DataFrame({
        "Modelo": ["TF-IDF + LogReg (ref.)", "TF-IDF + SVM (ref.)",
                   "TextCNN (este modelo)", "BERT fine-tuned (ref.)"],
        "Accuracy": ["~0.93", "~0.94", f"{tm['accuracy']:.4f}", "~0.987"],
        "F1-Score": ["~0.93", "~0.94", f"{tm['f1']:.4f}", "~0.988"],
    })
    print("\n" + "=" * 60)
    print("   TABLA COMPARATIVA FINAL")
    print("=" * 60)
    print(final_table.to_string(index=False))
    print("(Los valores de referencia son aproximados de la literatura.)")

    # ---- Visualizacion t-SNE / PCA de los embeddings ----
    print("\n=== Visualizacion de embeddings (t-SNE / PCA) ===")
    weights = model.embedding.weight.data.cpu().numpy()
    np.random.seed(42)
    hi = min(V, 5000)
    n_sample = min(800, max(2, hi - 2))
    if hi - 2 >= 2:
        sample_idx = np.random.choice(range(2, hi), size=min(n_sample, hi - 2), replace=False)
        vecs = weights[sample_idx]
        perp = max(5, min(30, len(sample_idx) - 1))
        tsne = TSNE(n_components=2, random_state=42, perplexity=perp, max_iter=500)
        tsne_xy = tsne.fit_transform(vecs)
        pca = PCA(n_components=2)
        pca_xy = pca.fit_transform(vecs)

        fig, ax = plt.subplots(1, 2, figsize=(16, 7))
        ax[0].scatter(tsne_xy[:, 0], tsne_xy[:, 1], alpha=0.3, s=6, c="steelblue")
        ax[0].set_title("t-SNE de embeddings aprendidos")
        for w in ["trump", "president", "election", "war", "fake", "news", "report", "official"]:
            if w in word2idx and word2idx[w] in sample_idx:
                j = int(np.where(sample_idx == word2idx[w])[0][0])
                ax[0].annotate(w, (tsne_xy[j, 0], tsne_xy[j, 1]), color="red", fontweight="bold")
        ax[1].scatter(pca_xy[:, 0], pca_xy[:, 1], alpha=0.3, s=6, c="coral")
        ax[1].set_title(f"PCA ({pca.explained_variance_ratio_[:2].sum()*100:.1f}% var)")
        plt.tight_layout(); fig.savefig("embeddings_tsne_pca.png", dpi=120); plt.close(fig)
        print("  [figura] embeddings_tsne_pca.png")
    else:
        print("  Vocabulario muy pequeno para t-SNE; se omite la grafica.")

    # ---- Palabras mas influyentes (analisis de gradientes) ----
    print("\n=== Top palabras influyentes (gradientes) ===")
    model.eval()
    word_imp = np.zeros(V); word_cnt = np.zeros(V)
    for bidx, (texts, labels) in enumerate(test_loader):
        if bidx >= 10:
            break
        texts = texts.to(device)
        e = model.embedding(texts); e.retain_grad()
        x = e.permute(0, 2, 1)
        if x.size(2) < model._max_kernel:
            x = F.pad(x, (0, model._max_kernel - x.size(2)))
        conv_out = [torch.relu(c(x)).max(dim=2)[0] for c in model.convs]
        x = model.dropout(torch.cat(conv_out, dim=1))
        out = model.fc(x)
        model.zero_grad()
        out.sum().backward()
        grad_mag = e.grad.norm(dim=2).detach().cpu().numpy()
        tok = texts.cpu().numpy()
        for i in range(len(tok)):
            for j in range(len(tok[i])):
                tid = tok[i][j]
                if tid > 1:
                    word_imp[tid] += grad_mag[i][j]; word_cnt[tid] += 1
    mask = word_cnt > 0
    avg = np.zeros(V); avg[mask] = word_imp[mask] / word_cnt[mask]
    top = np.argsort(avg)[-20:][::-1]
    print("Top 20 palabras mas influyentes:")
    for rank, idx in enumerate(top, 1):
        print(f"  {rank:2d}. {idx2word.get(idx, f'idx_{idx}'):15s} ({avg[idx]:.4f})")

    top_words = [idx2word.get(i, f"idx_{i}") for i in top]
    top_scores = [avg[i] for i in top]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(top_words)), top_scores[::-1], color="steelblue", edgecolor="black")
    ax.set_yticks(range(len(top_words))); ax.set_yticklabels(top_words[::-1])
    ax.set_xlabel("Importancia (magnitud del gradiente)")
    ax.set_title("Top 20 palabras mas influyentes")
    plt.tight_layout(); fig.savefig("palabras_influyentes.png", dpi=120); plt.close(fig)
    print("  [figura] palabras_influyentes.png")


if __name__ == "__main__":
    run()
