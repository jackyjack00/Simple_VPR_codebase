import torch
import torch.nn as nn
import torchvision.models

class GeMPooling(nn.Module):
    def __init__(self, feature_size, pool_size=7, init_norm=3.0, eps=1e-6, normalize=False, **kwargs):
        super(GeMPooling, self).__init__(**kwargs)
        self.feature_size = feature_size  # Final layer channel size, the pow calc at -1 axis
        self.pool_size = pool_size
        self.init_norm = init_norm
        #Use a learnable parameter p to compute not the standard average but a generalized where data is ^p and the sum is rooted by p
        #We want to train a p for each channel of the result of CNN --> p has dim of last channel, then it is broadcasted to the img dimension
        self.p = torch.nn.Parameter(torch.ones(self.feature_size) * self.init_norm, requires_grad=True)
        self.p.data.fill_(init_norm)
        self.normalize = normalize
        self.avg_pooling = nn.AvgPool2d((self.pool_size, self.pool_size))
        self.eps = eps

    def forward(self, features):
        # filter invalid value: set minimum to 1e-6
        # features-> (B, H, W, C)
        # feature tensor size is [n_batch , n_channels ,H , W] , p tensor should be broadcastable to feature size
        # first it substitues all values < eps with eps than it computes ^p
        # features is [batch , 512 , 7 , 7] p is [512] so manually reshape it to manage the broadcasting
        
        #ones = torch.ones( (512,7,7)).to(device = "cuda")
        #my_p = self.p.reshape((512,1,1)) * ones
        my_p = self.p.reshape((512,1,1))
        
        features = features.clamp(min=self.eps).pow(my_p)
        #features = features.permute((0, 3, 1, 2))
        #standard avg pooling operation
        features = self.avg_pooling(features)
        features = torch.squeeze(features)
        #^1/p
        features = torch.pow(features, (1.0 / self.p))
        
        #if you want to normalize output featurs to a unit vector 
        if self.normalize:
            features = F.normalize(features, dim=-1, p=2)
        return features
