import torch
import torch.nn as nn
import random
from create_lc import *

# ----------------------------------------------------------------------------
# ConvShapeletFilter + Transformer 
# ----------------------------------------------------------------------------
class ConvShapeletFilter(nn.Module):
    def __init__(self, shapelet_length, num_shapelets, eps=1e-6):
        super().__init__()
        self.shapelet_length = shapelet_length
        self.num_shapelets = num_shapelets
        self.eps = eps

        kernel = torch.randn(num_shapelets, shapelet_length) * 0.1
        self.shapelets = nn.Parameter(kernel)
        self.last_best_positions = None

    def forward(self, x):
        # x: [B, T]
        B, T = x.shape
        L = self.shapelet_length        
        x_win = x.unfold(1, L, 1)
        NT = x_win.size(1)
        
        mu_x = x_win.mean(dim=2, keepdim=True)
        std_x = x_win.std(dim=2, unbiased=False, keepdim=True) + self.eps
        x_norm = (x_win - mu_x) / std_x
        
        S = self.shapelets  # [K, L]
        mu_s = S.mean(dim=1, keepdim=True)
        std_s = S.std(dim=1, unbiased=False, keepdim=True) + self.eps
        S_norm = (S - mu_s) / std_s
        corr = torch.einsum('btl,kl->btk', x_norm, S_norm)
        
        max_vals, max_idx = corr.max(dim=1) 
        self.last_best_positions = max_idx
        return max_vals

class GenericTransformer(nn.Module):
    def __init__(self, input_channels, d_gen, kernel_size=8, n_heads=4, num_layers=1):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, d_gen, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(d_gen)
        self.conv2 = nn.Conv1d(d_gen, d_gen, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(d_gen)
        self.act = nn.GELU()
        encoder = nn.TransformerEncoderLayer(d_model=d_gen, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder, num_layers=num_layers)

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.conv2(out)))
        out = out.transpose(1, 2)
        enc = self.transformer(out)
        return enc.mean(dim=1)

class ShapeFormerConv(nn.Module):
    def __init__(self, input_channels, num_shapelets, shapelet_length, d_gen, num_classes):
        super().__init__()
        self.shapelet_filter = ConvShapeletFilter(shapelet_length, num_shapelets)
        self.generic = GenericTransformer(input_channels, d_gen)
        self.fc = nn.Linear(num_shapelets + d_gen, num_classes)

    def forward(self, x):
        # x: [B, T]
        shape_scores = self.shapelet_filter(x)
        gen_feat = self.generic(x.unsqueeze(1))
        fused = torch.cat([shape_scores, gen_feat], dim=1)
        return self.fc(fused)


def init_conv_shapelets(X, shapelet_length, num_shapelets):
    N, seq_len = X.size()
    kernels = []
    for _ in range(num_shapelets):
        seq = X[random.randrange(N)]
        pos = torch.argmin(seq).item()
        start = max(0, pos - shapelet_length//2)
        end = start + shapelet_length
        if end > seq_len:
            end = seq_len; start = end - shapelet_length
        
        kernels.append(seq[start:end])
    
    return torch.stack(kernels, dim=0)



def train_model(model, optimizer, criterion, 
                X_train, y_train, X_val, y_val,
                epochs=50, batch_size=32, patience=5, min_delta=0.0):
    best_vl, no_imp, best_state = float('inf'), 0, None
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": []
    }

    for ep in range(1, epochs+1):
        
        model.train()
        tl = 0
        perm = torch.randperm(len(X_train))
        for i in range(0, len(X_train), batch_size):
            idx = perm[i:i+batch_size]
            xb, yb = X_train[idx], y_train[idx]
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tl += loss.item() * xb.size(0)
        tr_loss = tl / len(X_train)

        
        model.eval()
        with torch.no_grad():
            vl = criterion(model(X_val), y_val).item()
            logits = model(X_val)
            preds = logits.argmax(dim=1)
            correct = (preds == y_val).float().sum().item()
            val_acc = correct / len(y_val)

        
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl)
        history["val_acc"].append(val_acc)

        
        print(f"Epoch {ep}: Train Loss {tr_loss:.4f}, Val Loss {vl:.4f}, Val Acc {val_acc:.4f}")

        
        if vl + min_delta < best_vl:
            best_vl, no_imp, best_state = vl, 0, model.state_dict()
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"Early stop at epoch {ep}")
                break

    
    if best_state:
        model.load_state_dict(best_state)
    return model, history



def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        preds = logits.argmax(dim=1)
        acc = (preds == y_test).float().mean().item()
    return acc, preds