import torch
import torch.nn as nn

class ConvShapeletFilter(nn.Module):
    def __init__(self, shapelet_length, num_shapelets, eps=1e-6, k_top=5):
        super().__init__()
        self.shapelet_length = shapelet_length
        self.num_shapelets = num_shapelets
        self.eps = eps
        self.k_top = k_top
        
        kernel = torch.randn(num_shapelets, shapelet_length) * 0.1
        self.shapelets = nn.Parameter(kernel)
        
        self.num_output_features = 4
        self.last_topk_positions = None

    def forward(self, x):
        B, T = x.shape
        L = self.shapelet_length
        
        x_win = x.unfold(1, L, 1)
        N = x_win.size(1)

        mu_x = x_win.mean(dim=2, keepdim=True)
        x_norm = x_win - mu_x 

        S = self.shapelets
        mu_s = S.mean(dim=1, keepdim=True)
        S_norm = S - mu_s 

        corr = torch.einsum('bnl,kl->bnk', x_norm, S_norm)

        k_val = min(self.k_top, N)
        topk_vals, topk_idx = torch.topk(corr, k=k_val, dim=1)
        
        if not self.training:
             self.last_topk_positions = topk_idx

        p1 = topk_vals[:, 0, :]
        
        if k_val >= 2:
            p2 = topk_vals[:, 1, :]
        else:
            p2 = p1 

        p_mean = topk_vals.mean(dim=1)
        dominance = torch.relu(p1 - p2)

        fused_features = torch.cat([p1, p_mean, p2, dominance], dim=1)
        
        return fused_features

class GenericTransformer(nn.Module):
    def __init__(self, input_channels, d_gen, kernel_size=8,
                 n_heads=4, num_layers=1, max_len=1024,
                 shapelet_feat_dim=120):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, d_gen, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(d_gen)
        self.conv2 = nn.Conv1d(d_gen, d_gen, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(d_gen)
        self.act = nn.GELU()
        
        self.d_gen = d_gen
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_gen))
        self.shapelet_proj = nn.Linear(shapelet_feat_dim, d_gen)
        self.pos_emb = nn.Embedding(max_len + 2, d_gen)
        
        encoder = nn.TransformerEncoderLayer(d_model=d_gen, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder, num_layers=num_layers)

    def forward(self, x_time, x_shapelet):
        out_time = self.act(self.bn1(self.conv1(x_time)))
        out_time = self.act(self.bn2(self.conv2(out_time)))
        out_time = out_time.transpose(1, 2)
        B, T, C = out_time.shape 

        out_shape = self.shapelet_proj(x_shapelet) 
        out_shape = out_shape.unsqueeze(1)

        cls_tokens = self.cls_token.expand(B, -1, -1)

        out = torch.cat((cls_tokens, out_shape, out_time), dim=1)
        
        total_len = T + 2
        pos = torch.arange(total_len, device=out.device).unsqueeze(0).expand(B, total_len)
        
        if total_len > self.pos_emb.num_embeddings:
            pos = pos[:, :self.pos_emb.num_embeddings]
            out = out[:, :self.pos_emb.num_embeddings, :]
            
        out = out + self.pos_emb(pos)
        
        enc = self.transformer(out)
        
        return enc[:, 0, :]

class ShapeFormerConv(nn.Module):
    def __init__(self, input_channels, num_shapelets, shapelet_length,
                 d_gen, num_classes, max_seq_len=1024):
        super().__init__()
        
        self.shapelet_filter = ConvShapeletFilter(
            shapelet_length, num_shapelets, k_top=5
        )
        
        shapelet_output_dims = num_shapelets * self.shapelet_filter.num_output_features
        
        self.generic = GenericTransformer(
            input_channels=input_channels,
            d_gen=d_gen,
            max_len=max_seq_len,
            shapelet_feat_dim=shapelet_output_dims
        )

        self.fc_class = nn.Linear(d_gen, num_classes)
        self.fc_peak = nn.Linear(d_gen, 1)

    def forward(self, x):
        x_flux_only = x[:, 0, :] 
        shape_scores = self.shapelet_filter(x_flux_only)
        
        gen_feat = self.generic(x, shape_scores)
        
        out_class = self.fc_class(gen_feat) 
        
        out_peak = torch.sigmoid(self.fc_peak(gen_feat)).squeeze(-1)
        
        return out_class, out_peak