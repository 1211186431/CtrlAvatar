import torch.nn as nn
from .network import get_encoding ,get_mlp

class ColorNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_input_dims = config.n_input_dims
        self.n_feature_dims = config.n_output_dims
        self.encoding = get_encoding(self.n_input_dims, config.encoder)
        self.feature_network = get_mlp(self.encoding.n_output_dims, self.n_feature_dims, config.feature_mlp)

    def forward(self, x):
        points = x
        enc = self.encoding(points.view(-1, self.n_input_dims))
        features = self.feature_network(enc).view(
            *points.shape[:-1], self.n_feature_dims
        )
        return {"features": features}
