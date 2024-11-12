import torch
import torch.nn as nn
import torch.nn.functional as F
class condNet(nn.Module):
    def __init__(self,cond_dim=73,multires=4):
        super(condNet, self).__init__()
        d_in = 3
        self.multires = multires
        self.cond_dim = cond_dim
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, d_in)
            self.embed_fn = embed_fn
        self.cond_mlp1 = nn.Linear(cond_dim,128)
        self.cond_mlp2 = nn.Linear(128,128)
        self.cond_mlp3 = nn.Linear(128,input_ch)
        self.softplus = torch.nn.Softplus(beta=100)

    def forward(self, x,cond):
        B,N,_ =x.shape
        input_emb = self.embed_fn(x)
        cond = cond.unsqueeze(1).repeat(1,N,1)
        body_emb = self.softplus(self.cond_mlp1(cond))
        body_emb = self.softplus(self.cond_mlp2(body_emb))
        body_emb = self.cond_mlp3(body_emb)
        cond1 = input_emb*body_emb
        inputs = torch.cat([cond1,body_emb], dim=-1)
        inputs = inputs.reshape(B,N,-1)
        return inputs
    
""" Positional encoding embedding. Code was taken from https://github.com/bmild/nerf. """

class Embedder:

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0**torch.linspace(0.0, max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(
                    lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, d_in):
    embed_kwargs = {
        "include_input": True,
        "input_dims": d_in,
        "max_freq_log2": multires - 1,
        "num_freqs": multires,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)

    def embed(x, eo=embedder_obj):
        return eo.embed(x)

    return embed, embedder_obj.out_dim 

 
