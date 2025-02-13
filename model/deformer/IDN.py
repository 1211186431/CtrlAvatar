import torch
import torch.nn as nn
from .mlp import get_embedder,ImplicitNetwork
class CondNet(nn.Module):
    def __init__(self,cond_dim=73,multires=4):
        super(CondNet, self).__init__()
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

    def forward(self, x, cond):
        B, N, _ = x.shape
        input_emb = self.embed_fn(x)
        body_emb = self.softplus(self.cond_mlp1(cond))
        body_emb = self.softplus(self.cond_mlp2(body_emb))
        body_emb = self.cond_mlp3(body_emb)
        body_emb = body_emb.unsqueeze(1).repeat(1,N,1)
        cond1 = input_emb*body_emb
        inputs = torch.cat([cond1,body_emb], dim=-1)
        return inputs
    
class IDN(nn.Module):
    def __init__(self,cond_dim=73,multires=4):
        super(IDN, self).__init__()
        self.cond_net = CondNet(cond_dim=cond_dim,multires=multires)
        self.delta_net = ImplicitNetwork(d_in=54,d_out=3,skip_layer=[3],depth=4,width=256,multires=0,geometric_init=False)
        
    def forward(self, x, cond, inverse=False):
        """
        IDN Model
        
        Args:
            x (Tensor): (batch_size, num_vertices, 3).
            cond (Tensor): (batch_size, 73).
            inverse (bool): when is lbs forward, inverse=False, else inverse=True. Training is inverse=True, inference is inverse=False.
            
        Returns:
            Tensor: (batch_size, num_vertices, 3).
        """
        emb = self.cond_net(x,cond)
        x_delta = self.delta_net(emb,return_feature=False)
        if not inverse:
            x = x - x_delta * 0.1
        else:
            x = x + x_delta * 0.1
        return x
    
    def load_cond_weights(self, pretrained_model):
        cond_state_dict = {key: value for key, value in pretrained_model['state_dict'].items() if key.startswith('cond_net')}
        adjusted_state_dict = {key.replace('cond_net.', ''): value for key, value in cond_state_dict.items()}
        self.cond_net.load_state_dict(adjusted_state_dict)
        
    def load_delta_weights(self, pretrained_model):
        delta_state_dict = {key: value for key, value in pretrained_model['state_dict'].items() if key.startswith('delta_net')}
        adjusted_state_dict = {key.replace('delta_net.', ''): value for key, value in delta_state_dict.items()}
        self.delta_net.load_state_dict(adjusted_state_dict)
        
    def load_IDN_weights(self, pretrained_path):
        pretrained_model = torch.load(pretrained_path, map_location=torch.device('cuda:0'))
        self.load_cond_weights(pretrained_model)
        self.load_delta_weights(pretrained_model)
    