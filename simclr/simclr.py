import torch.nn as nn

from simclr.modules.identity import Identity

class SimCLR(nn.Module):
    def __init__(self, encoder, projection_dim, n_features):
        super().__init__()

        self.encoder = encoder
        self.n_features = n_features

        # 最后fc层换了
        self.encoder.fc = Identity

        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, projection_dim, bias=False)
        )

    def forwar(self, x_i, x_j):
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)

        z_i = self.projector(h_i)
        z_j = self.projector(h_j)
        return h_i, h_j, z_i, z_j



