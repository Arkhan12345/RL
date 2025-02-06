import torch
import torch.nn as nn


class QRCNN(nn.Module):
    """
    A 3-layer CNN for (84,84,1) grayscale Atari frames,
    producing (batch_size, num_actions, num_quantiles).
    """
    def __init__(self, in_channels=1, num_actions=4, num_quantiles=51):
        super().__init__()
        self.num_actions = num_actions
        self.num_quantiles = num_quantiles

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),  # 84->20
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),           # 20->9
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),           # 9->7
            nn.ReLU(),
            nn.Flatten()
        )
        # final feature size: 64*7*7 = 3136
        self.fc = nn.Linear(3136, 512)
        self.v = nn.Linear(512, num_actions * num_quantiles)

    def forward(self, x):
        """
        x shape: (batch_size, in_channels=1, 84, 84)
        returns: (batch_size, num_actions, num_quantiles)
        """
        x = self.features(x)
        x = torch.relu(self.fc(x))
        x = self.v(x)
        x = x.view(-1, self.num_actions, self.num_quantiles)
        return x