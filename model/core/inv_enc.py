import torch
import torch.nn as nn
from model.core.vae import EncoderCNN, EncoderRNN, AbstractEncoder
from model.core.gru_encoder import GRUEncoder
import torch.nn.functional as F

class INV_ENC(nn.Module):
    def __init__(self, task, vae_enc=None, cnn_filt=8, modulator_dim=0, content_dim=0, rnn_hidden=10, T_inv=10, device='cpu'):
        super(INV_ENC, self).__init__()
        self.modulator_dim = modulator_dim
        self.content_dim = content_dim
        if task=='rot_mnist' or task=='rot_mnist_ou':
            self.inv_encoder = InvariantEncoderCNN(task=task, out_distr='dirac', enc_out_dim=modulator_dim+content_dim, n_filt=cnn_filt, T_inv=T_inv).to(device)
        elif task=='bb':
            self.inv_encoder = InvariantEncoderCNNMLP(vae_enc, T_inv, H=64, out_dim=modulator_dim+content_dim)
        elif task=='sin' or task=='lv' or 'mocap' in task:
            if task=='sin':
                data_dim = 1
            elif task=='lv':
                data_dim = 2
            elif 'mocap' in task:
                data_dim = 50
            #self.inv_encoder = InvariantEncoderRNN(data_dim, T_inv=T_inv, rnn_hidden=rnn_hidden, enc_out_dim=modulator_dim+content_dim, out_distr='dirac').to(device)
            self.inv_encoder = InvariantPriorEncoder(input_dim=data_dim, latent_dim=modulator_dim+content_dim, H1=10, H2=10, ns=8).to(device) #*#
    def kl(self):
        return torch.zeros(1) * 0.0

    def forward(self, X, L=1):
        '''
            X is [N,T,nc,d,d] or [N,T,q] 
            returns [L,N,T,q]
        '''
        c = self.inv_encoder(X) # N,Tinv,q or N,ns,q
       # return c.repeat([L, 1, 1, 1])  # L,N,T,q#
        return c.repeat([L, 1, 1]) #*#

class InvariantEncoderRCNN(nn.Module):
    def __init__(self, task, out_distr='dirac', enc_out_dim=16, n_filt=8, n_in_channels=1, T_inv=15, vae_enc=None):
        super().__init__()
        self.enc_out_dim = enc_out_dim
        if vae_enc is None:
            self.cnn_enc = EncoderCNN(task, out_distr=out_distr,  enc_out_dim=enc_out_dim, n_filt=8, n_in_channels=1)
        else:
            self.cnn_enc = nn.Sequential(vae_enc.cnn, nn.Linear(vae_enc.in_features, enc_out_dim))
        self.rnn_enc = EncoderRNN(enc_out_dim, rnn_hidden=10, enc_out_dim=enc_out_dim, out_distr=out_distr) 
        self.T_inv = T_inv

    def forward(self, X, ns=5):
        [N, T, nc, d, d] = X.shape
        z = self.cnn_enc(X.reshape(N*T, nc, d, d)).reshape(N, T, -1)  # N,T,n
        T_inv = min(self.T_inv, T)
        z   = z.repeat([ns, 1, 1])
        t0s = torch.randint(0, T-T_inv+1, [ns*N]) 
        z   = torch.stack([z[n, t0:t0+T_inv] for n, t0 in enumerate(t0s)])  # ns*N,T_inv,d
        X_out = self.rnn_enc(z)  # ns*N,enc_out_dim
        return X_out.reshape(ns, N, self.enc_out_dim).permute(1, 0, 2)  # N,ns,enc_out_dim

class InvariantEncoderCNNMLP(nn.Module):
    def __init__(self, vae_enc, T_inv, H, out_dim):
        super().__init__()
        self.vae_enc = vae_enc
        self.out_dim = out_dim
        self.mlp = nn.Sequential(nn.Linear(vae_enc.H*T_inv, H), nn.ReLU(), nn.Linear(H, out_dim))
        self.T_inv = T_inv
    def forward(self, X, ns=5):
        [N,T,nc,d,d] = X.shape
        z = self.vae_enc.backbone(X.reshape(N*T,nc,d,d)).reshape(N,T,-1) # N,T,n
        T_inv = min(self.T_inv,T)
        z     = z.repeat([ns,1,1])
        t0s   = torch.randint(0,T-T_inv+1,[ns*N]) 
        z     = torch.stack([z[n,t0:t0+T_inv] for n,t0 in enumerate(t0s)]) # ns*N,T_inv,d
        X_out = self.mlp(z.reshape(ns*N, T_inv*z.shape[-1])) # ns*N,enc_out_dim
        return X_out.reshape(ns,N,self.out_dim).permute(1,0,2) # N,ns,enc_out_dim
    
class InvariantEncoderCNN(EncoderCNN):
    def __init__(self, task, out_distr='dirac', enc_out_dim=16, n_filt=8, n_in_channels=1, T_inv=15):
        super().__init__(task, out_distr=out_distr, enc_out_dim=enc_out_dim, n_filt=n_filt, n_in_channels=n_in_channels)
        self.T_inv = T_inv

    def forward(self, X):
        [N, T, nc, d, d] = X.shape
        T_inv = T//2 if self.T_inv is None else self.T_inv
        T_inv = min(T_inv, T)
        t     = torch.stack([torch.randperm(T)[:T_inv] for _ in range(N)], 1).to(X.device)  # T_inv,N
        index = torch.arange(N).repeat(T_inv, 1).to(X.device)  # T_inv,N
        X     = X[index.view(-1), t.view(-1)].view(T_inv * N, nc, d, d)        
        X_out = super().forward(X)  # N*T,_
        return X_out.reshape(T_inv, N, self.enc_out_dim).permute(1, 0, 2)

class InvariantEncoderRNN(EncoderRNN):
    def __init__(self, input_dim, T_inv=None, rnn_hidden=10, enc_out_dim=16, out_distr='dirac'):
        super(InvariantEncoderRNN, self).__init__(input_dim, rnn_hidden=rnn_hidden, enc_out_dim=enc_out_dim, out_distr=out_distr)
        self.T_inv = T_inv

    def forward(self, X, ns=5):
        [N, T, d] = X.shape
        T_inv = T//2 if self.T_inv is None else self.T_inv
        T_inv = min(T_inv, T)
        X   = X.repeat([ns, 1, 1])
        t0s = torch.randint(0, T-T_inv+1, [ns*N])
        X   = torch.stack([X[n, t0:t0+T_inv] for n, t0 in enumerate(t0s)])  # ns*N,T_inv,d
        X_out = super().forward(X)  # ns*N,enc_out_dim
        return X_out.reshape(ns, N, self.enc_out_dim).permute(1, 0, 2)  # N,ns,enc_out_dim



class InvariantPriorEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, H1, H2, ns=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.ns = ns

        # Mean network
        self.mean_nn = nn.Sequential(nn.Linear(input_dim, H1), nn.ReLU(True), nn.Linear(H1, latent_dim))
        self.cov_nn = nn.Sequential(nn.Linear(2*input_dim, H2), nn.ReLU(True), nn.Linear(H2, latent_dim))

    def mean_est(self, X):
        N, q, D = X.shape
        _X = self.mean_nn(X)
        return _X.mean(dim=1)  # (N, q)
    
    def cov_est(self, X):
        """
        X: Tensor of shape (N, q, D)
        Returns:
            cov: Tensor of shape (N, q, q)
        """
        N, q, D = X.shape
        device = X.device

        # Expand for pairwise concatenation
        Xi = X.unsqueeze(2).expand(N, q, q, D)  # (N, q, q, D)
        Xj = X.unsqueeze(1).expand(N, q, q, D)  # (N, q, q, D)

        # Concatenate along feature dimension
        Xpair = torch.cat([Xi, Xj], dim=-1)     # (N, q, q, 2D)

        # Flatten for network
        Xpair = Xpair.reshape(N * q * q, 2 * D)

        # Compute L entries
        L_entries = self.cov_nn(Xpair)          # (N*q*q, q)

        # Reshape
        L_entries = L_entries.view(N, q, q, q)

        # Extract diagonal-aligned values to build L
        # L[n, i, j] = output[n, i, j, j]
        L = torch.zeros(N, q, q, device=device)
        for j in range(q):
            L[:, :, j] = L_entries[:, :, j, j]

        # Optional: enforce lower-triangular structure
        L = torch.tril(L)

        # Covariance
        cov = L @ L.transpose(-1, -2)  # (N, q, q)

        return cov
    
    def forward(self, X):
        """
        X: [N, T, d]
        """
        eps=1e-6
        N, T, d = X.shape
        T_inv = self.latent_dim
        ns = self.ns
        device = X.device

        X_rep = X.repeat(ns, 1, 1)  # [ns*N, T, d]
        t0s = torch.randint(
            0, T - T_inv + 1, (ns * N,), device=device
        )

        X_win = torch.stack(
            [X_rep[i, t0:t0 + T_inv] for i, t0 in enumerate(t0s)],
            dim=0
        )  # [ns*N, T_inv, d]

        mu = self.mean_est(X_win)  
        cov = self.cov_est(X_win)


        mu = mu.view(ns, N, self.latent_dim).mean(dim=0)  # [N, q]
        cov = cov.view(ns, N, self.latent_dim, self.latent_dim).mean(dim=0)  # [N, q, q]
        device = mu.device

        # Stabilize covariance (important!)
        cov = cov + eps * torch.eye(self.latent_dim, device=device).unsqueeze(0)

        # Cholesky factor
        L = torch.linalg.cholesky(cov)  # (N, q, q)

        # Standard normal
        eps_z = torch.randn(N, self.latent_dim, device=device)

        # Reparameterization
        z = mu + torch.bmm(L, eps_z.unsqueeze(-1)).squeeze(-1)
        return z