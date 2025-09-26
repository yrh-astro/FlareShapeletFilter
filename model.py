import torch
import torch.nn as nn
import random
from create_lc import *

class ConvShapeletFilterClassWise(nn.Module):
    def __init__(self, shapelet_length, num_shapelets, num_classes, eps=1e-6):
        super().__init__()
        assert num_shapelets % num_classes == 0, "num_shapelets must be divisible by num_classes"
        self.shapelet_length = shapelet_length
        self.num_classes = num_classes
        self.Kc = num_shapelets // num_classes
        self.eps = eps
        
        self.shapelets = nn.Parameter(
            torch.randn(num_classes, self.Kc, shapelet_length) * 0.1
        )
        self.last_best_positions = None

    def forward(self, x):
        # x: [B, T]
        B, T = x.shape
        L = self.shapelet_length
        
        x_win = x.unfold(1, L, 1)
        mu_x = x_win.mean(dim=2, keepdim=True)
        std_x = x_win.std(dim=2, unbiased=False, keepdim=True) + self.eps
        x_norm = (x_win - mu_x) / std_x  # [B, NT, L]

        outputs = []
        all_pos = []
        
        for c in range(self.num_classes):
            S = self.shapelets[c]  # [Kc, L]
            mu_s = S.mean(dim=1, keepdim=True)
            std_s = S.std(dim=1, unbiased=False, keepdim=True) + self.eps
            S_norm = (S - mu_s) / std_s  # [Kc, L]
            
            corr = torch.einsum('btl,kl->btk', x_norm, S_norm) / L  # [B, NT, Kc]
            max_vals, max_idx = corr.max(dim=1)  # [B, Kc], [B, Kc]
            outputs.append(max_vals)
            all_pos.append(max_idx)

        
        self.last_best_positions = torch.stack(all_pos, dim=1)  # [B, C, Kc]
        return torch.cat(outputs, dim=1)  # [B, C*Kc]


class GenericTransformer(nn.Module):
    def __init__(self, input_channels, d_model,
                 kernel_size=8, n_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, d_model, kernel_size, padding=kernel_size//2)
        self.bn1   = nn.BatchNorm1d(d_model)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size//2)
        self.bn2   = nn.BatchNorm1d(d_model)
        self.act   = nn.GELU()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            batch_first=True, dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x: [B, C_in, T]
        y = self.act(self.bn1(self.conv1(x)))
        y = self.act(self.bn2(self.conv2(y)))    # [B, d_model, T]
        y = y.transpose(1, 2)                     # [B, T, d_model]
        y = self.transformer(y)                   # [B, T, d_model]
        return y.mean(dim=1)                      # [B, d_model]


class ShapeFormerConv(nn.Module):
    def __init__(self, input_channels, num_shapelets,
                 shapelet_length, d_gen, num_classes):
        super().__init__()
        self.num_shapelets = num_shapelets
        self.d_gen         = d_gen
        self.num_classes   = num_classes

        self.shapelet_filter = ConvShapeletFilterClassWise(
            shapelet_length, num_shapelets, num_classes
        )
        self.generic         = GenericTransformer(input_channels, d_gen)
        self.dropout         = nn.Dropout(0.2)

        
        self.gate = nn.Sequential(
            nn.Linear(num_shapelets + d_gen, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.fc = nn.Linear(max(num_shapelets, d_gen), num_classes)

    def forward(self, x):
        # x: [B, T]
        s = self.shapelet_filter(x)           # [B, K]
        t = self.generic(x.unsqueeze(1))      # [B, d_gen]
        feat = torch.cat([s, t], dim=1)       # [B, K + d_gen]

        α = self.gate(feat)                   # [B, 1]

        
        if self.num_shapelets != self.d_gen:
            d = max(self.num_shapelets, self.d_gen)
            s = F.pad(s, (0, d - self.num_shapelets))
            t = F.pad(t, (0, d - self.d_gen))

        fused = α * s + (1 - α) * t           # [B, d]
        fused = self.dropout(fused)
        return self.fc(fused)                 # [B, num_classes]


def split_data(X, y, n_tr, n_val, n_te):
    idx = np.arange(len(X)); np.random.shuffle(idx)
    X, y = X[idx], y[idx]
    X_tr, y_tr = X[:n_tr], y[:n_tr]
    X_va, y_va = X[n_tr:n_tr+n_val], y[n_tr:n_tr+n_val]
    X_te, y_te = X[n_tr+n_val:n_tr+n_val+n_te], y[n_tr+n_val:n_tr+n_val+n_te]
    return (torch.tensor(X_tr, dtype=torch.float),
            torch.tensor(y_tr, dtype=torch.long),
            torch.tensor(X_va, dtype=torch.float),
            torch.tensor(y_va, dtype=torch.long),
            torch.tensor(X_te, dtype=torch.float),
            torch.tensor(y_te, dtype=torch.long))

def train_model(model, optimizer, criterion,
                X_tr, y_tr, X_va, y_va,
                epochs=50, batch_size=32, patience=5):
    best_vl, no_imp, best_state = float('inf'), 0, None
    for ep in range(1, epochs+1):
        
        model.train(); perm = torch.randperm(len(X_tr)); tloss=0.0
        for i in range(0, len(X_tr), batch_size):
            idx = perm[i:i+batch_size]
            xb, yb = X_tr[idx], y_tr[idx]
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tloss += loss.item() * xb.size(0)
        tr_loss = tloss / len(X_tr)

        
        model.eval(); 
        with torch.no_grad():
            vl = criterion(model(X_va), y_va).item()
            preds = model(X_va).argmax(dim=1)
            va_acc = (preds == y_va).float().mean().item()

        print(f"Epoch {ep}: Train {tr_loss:.4f}, Val {vl:.4f}, Val Acc {va_acc:.4f}")
        if vl < best_vl:
            best_vl, no_imp, best_state = vl, 0, model.state_dict()
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"[EarlyStop] Epoch {ep}")
                break

    if best_state:
        model.load_state_dict(best_state)
    return model

def evaluate_model(model, X, y):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean().item()
    return acc, preds
