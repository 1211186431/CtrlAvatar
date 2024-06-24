import torch
import torch.nn as nn
import torch.nn.functional as F

class colorNet(nn.Module):
    def __init__(self,output_dim=3,multires=10,d_in=3):
        super(colorNet, self).__init__()
        self.d_in = d_in
        self.output_dim = output_dim
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, self.d_in)
            self.embed_fn = embed_fn
        else:
            input_ch = d_in
        
        self.fc1 = nn.Linear(input_ch, 256)  
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256-input_ch)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, output_dim) 
        self.softplus = torch.nn.Softplus(beta=100)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        n_batch, n_point, _ = x.shape
        x = x.reshape(-1,self.d_in)
        input_emb = self.embed_fn(x)
        x = input_emb
        x = self.softplus(self.fc1(x))
        x = self.softplus(self.fc2(x))
        x = self.softplus(self.fc3(x))
        x = self.softplus(self.fc4(torch.cat([x, input_emb], dim=-1)))
        x = self.softplus(self.fc5(x))
        x = self.sigmoid(self.fc6(x))
        x = x.reshape(n_batch, n_point, -1)
        return x


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

if "__main__" == __name__:
    model = colorNet().to('cuda:0')
    x = torch.randn(1, 166475, 3).to('cuda:0')
    output = model(x)
    print(output.shape)  # torch.Size([1, 6890, 3]) 