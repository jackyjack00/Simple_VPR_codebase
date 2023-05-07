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

########################################################################################################################################
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


class FeatureMixerLayer(nn.Module):
    def __init__(self, in_dim, mlp_ratio=1):
        super().__init__()
        # Mixer Inner Structure: Norm , Linear , ReLu, Linear
        self.mix = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, int(in_dim * mlp_ratio)),
            nn.ReLU(),
            nn.Linear(int(in_dim * mlp_ratio), in_dim),
        )
        # Initialization of each Linear layer with normal distributed weights N(mean = 0 , std = 0.02) and bias = 0
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Forward uses the Mixer and a skip connection 
        return x + self.mix(x)


class MixVPR(nn.Module):
    def __init__(self,
                 in_channels=1024,
                 in_h=20,
                 in_w=20,
                 out_channels=512,
                 mix_depth=1,
                 mlp_ratio=1,
                 out_rows=4,
                 ) -> None:
        super().__init__()

        self.in_h = in_h  # height of input feature maps
        self.in_w = in_w  # width of input feature maps
        self.in_channels = in_channels  # depth of input feature maps

        self.out_channels = out_channels  # depth wise projection dimension
        self.out_rows = out_rows  # row wise projection dimesion

        self.mix_depth = mix_depth  # L the number of stacked FeatureMixers
        self.mlp_ratio = mlp_ratio  # ratio of the mid projection layer in the mixer block

        hw = in_h * in_w
        self.mix = nn.Sequential(*[
            FeatureMixerLayer(in_dim=hw, mlp_ratio=mlp_ratio)
            for _ in range(self.mix_depth)
        ])
        self.channel_proj = nn.Linear(in_channels, out_channels)
        self.row_proj = nn.Linear(hw, out_rows)

    def forward(self, x):
        x = x.flatten(2)
        x = self.mix(x)
        x = x.permute(0, 2, 1)
        x = self.channel_proj(x)
        x = x.permute(0, 2, 1)
        x = self.row_proj(x)
        x = F.normalize(x.flatten(1), p=2, dim=-1)
        return x
    
########################################################################################################################################    
import faiss 
import torch
import torch.nn as nn

class ProxyAccumulator:
    def __init__(self, tensor = None, n = 0, dim = 512):
        # Initialize the tensor and the counter of how many proxy is storing
        if tensor is None:
            # Tensor
            self.proxy_sum = torch.zeros(dim)
            # Counter
            self.n = n
        else:
            self.proxy_sum = tensor
            self.n = n
        
    def get_avg(self):
        return self.proxy_sum / self.n

    def __add__(self, other):
        return ProxyAccumulator(tensor = self.proxy_sum + other.proxy_sum, n = self.n + other.n)

class ProxyHead(nn.Module):
    def __init__(self, in_dim , proxy_dim = 512):
        super().__init__()
        # Setting starting space dimension as in_dim and output space dimension as proxy_dim
        self.projector = nn.Linear(in_dim, proxy_dim)
        
    def forward(self , x):
        # Projection from R_in_dim to R_proxy_dim
        out = self.projector(x)
        # L2 norm of output
        out = F.normalize(out, p=2)
        return out
    
class ProxyBank():
    def __init__(self, batch_size , proxy_dim = 512):
        # Set the size of batches we want to generate
        self.batch_size = batch_size
        # Initialize an index containing vectors of dim equals to the proxy 
        self.__index = faiss.IndexFlatL2( proxy_dim )
        # Initialize a dictionary to summarize the proxy-place_label relation
        self.__bank = {}
    
    def update_bank(self, proxies, labels):
        # TODO: call at epoch_end
        # Iterate over each pair proxy-label where proxy is the result of projection done by ProxyHead
        for proxy, label in zip(proxies , labels):
            # Create or Update the content of the bank dictionary
            if label not in self.bank.keys():
                self.__bank[label] = ProxyAccumulator( tensor = proxy , n = 1 )
            else:
                self.__bank[label] = ProxyAccumulator( tensor = proxy , n = 1 ) + self.__bank[label]
        #TODO: once all is stored, then remember to use ProxyAccumulator.get_avg()
        #TODO: reset and update the faiss index, possibly with a retrival of both vectors and labels
    
    #TODO: understand once the indexx is complete how to generate the batches usefull for the sampler and how to pass the results to it
    def proxy_batch_sampling(self , batch_dim):
        pass
       
########################################################################################################################################    

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import faiss
import numpy as np


class NetVLAD(nn.Module):
    def __init__(self, num_clusters=64, dim=512, normalize_input=True):
        super().__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=False)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
    
    def __forward_vlad__(self, x_flatten, soft_assign, N, D):
        vlad = torch.zeros([N, self.num_clusters, D], dtype=x_flatten.dtype, device=x_flatten.device)
        for D in range(self.num_clusters):  # Slower than non-looped, but lower memory usage
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                    self.centroids[D:D+1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual = residual * soft_assign[:, D:D+1, :].unsqueeze(2)
            vlad[:, D:D+1, :] = residual.sum(dim=-1)
        vlad = F.normalize(vlad, p=2, dim=2)
        vlad = vlad.view(N, -1)
        vlad = F.normalize(vlad, p=2, dim=1)
        return vlad
    
    def forward(self, x):
        N, D, H, W = x.shape[:]
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)
        x_flatten = x.view(N, D, -1)
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)
        vlad = self.__forward_vlad__(x_flatten, soft_assign, N, D)
        return vlad
    
########################################################################################################################################à
#TODO: DEBUG THE PATCH_NETVLAD, SOMETHING IS WRONG  
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import faiss
import numpy as np


def get_integral_feature(feat_in):
    """
    Input/Output as [N,D,H,W] where N is batch size and D is descriptor dimensions
    For VLAD, D = K x d where K is the number of clusters and d is the original descriptor dimensions
    """
    feat_out = torch.cumsum(feat_in, dim=-1)
    feat_out = torch.cumsum(feat_out, dim=-2)
    feat_out = torch.nn.functional.pad(feat_out, (1, 0, 1, 0), "constant", 0)
    return feat_out


def get_square_regions_from_integral(feat_integral, patch_size, patch_stride):
    """
    Input as [N,D,H+1,W+1] where additional 1s for last two axes are zero paddings
    regSize and regStride are single values as only square regions are implemented currently
    """
    N, D, H, W = feat_integral.shape

    if feat_integral.get_device() == -1:
        conv_weight = torch.ones(D, 1, 2, 2)
    else:
        conv_weight = torch.ones(D, 1, 2, 2, device=feat_integral.get_device())
    conv_weight[:, :, 0, -1] = -1
    conv_weight[:, :, -1, 0] = -1
    feat_regions = torch.nn.functional.conv2d(feat_integral, conv_weight, stride=patch_stride, groups=D, dilation=patch_size)
    return feat_regions / (patch_size ** 2)


class PatchNetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=128, normalize_input=True, vladv2=False, use_faiss=True,
                 patch_sizes='4', strides='1'):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
            vladv2 : bool
                If true, use vladv2 otherwise use vladv1
            use_faiss: bool
                Default true, if false don't use faiss for similarity search
            patch_sizes: string
                comma separated string of patch sizes
            strides: string
                comma separated string of strides (for patch aggregation)
        """
        super().__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = 0
        self.vladv2 = vladv2
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=vladv2)
        # noinspection PyArgumentList
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self.use_faiss = use_faiss
        self.padding_size = 0
        patch_sizes = patch_sizes.split(",")
        strides = strides.split(",")
        self.patch_sizes = []
        self.strides = []
        for patch_size, stride in zip(patch_sizes, strides):
            self.patch_sizes.append(int(patch_size))
            self.strides.append(int(stride))

    def init_params(self, clsts, traindescs):
        if not self.vladv2:
            clsts_assign = clsts / np.linalg.norm(clsts, axis=1, keepdims=True)
            dots = np.dot(clsts_assign, traindescs.T)
            dots.sort(0)
            dots = dots[::-1, :]  # sort, descending

            self.alpha = (-np.log(0.01) / np.mean(dots[0, :] - dots[1, :])).item()
            # noinspection PyArgumentList
            self.centroids = nn.Parameter(torch.from_numpy(clsts))
            # noinspection PyArgumentList
            self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha * clsts_assign).unsqueeze(2).unsqueeze(3))
            self.conv.bias = None
        else:
            if not self.use_faiss:
                knn = NearestNeighbors(n_jobs=-1)
                knn.fit(traindescs)
                del traindescs
                ds_sq = np.square(knn.kneighbors(clsts, 2)[1])
                del knn
            else:
                index = faiss.IndexFlatL2(traindescs.shape[1])
                # noinspection PyArgumentList
                index.add(traindescs)
                del traindescs
                # noinspection PyArgumentList
                ds_sq = index.search(clsts, 2)[1]
                del index

            self.alpha = (-np.log(0.01) / np.mean(ds_sq[:, 1] - ds_sq[:, 0])).item()
            # noinspection PyArgumentList
            self.centroids = nn.Parameter(torch.from_numpy(clsts))
            del clsts, ds_sq

            # noinspection PyArgumentList
            self.conv.weight = nn.Parameter(
                (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
            )
            # noinspection PyArgumentList
            self.conv.bias = nn.Parameter(
                - self.alpha * self.centroids.norm(dim=1)
            )

    def forward(self, x):
        N, C, H, W = x.shape

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, H, W)
        soft_assign = F.softmax(soft_assign, dim=1)

        # calculate residuals to each cluster
        store_residual = torch.zeros([N, self.num_clusters, C, H, W], dtype=x.dtype, layout=x.layout, device=x.device)
        for j in range(self.num_clusters):  # slower than non-looped, but lower memory usage
            residual = x.unsqueeze(0).permute(1, 0, 2, 3, 4) - \
                self.centroids[j:j + 1, :].expand(x.size(2), x.size(3), -1, -1).permute(2, 3, 0, 1).unsqueeze(0)

            residual *= soft_assign[:, j:j + 1, :].unsqueeze(2)  # residual should be size [N K C H W]
            store_residual[:, j:j + 1, :, :, :] = residual

        vlad_global = store_residual.view(N, self.num_clusters, C, -1)
        vlad_global = vlad_global.sum(dim=-1)
        store_residual = store_residual.view(N, -1, H, W)

        ivlad = get_integral_feature(store_residual)
        vladflattened = []
        for patch_size, stride in zip(self.patch_sizes, self.strides):
            vladflattened.append(get_square_regions_from_integral(ivlad, int(patch_size), int(stride)))

        vlad_local = []
        for thisvlad in vladflattened:  # looped to avoid GPU memory issues with certain config combinations
            thisvlad = thisvlad.view(N, self.num_clusters, C, -1)
            thisvlad = F.normalize(thisvlad, p=2, dim=2)
            thisvlad = thisvlad.view(x.size(0), -1, thisvlad.size(3))
            thisvlad = F.normalize(thisvlad, p=2, dim=1)
            vlad_local.append(thisvlad)

        vlad_global = F.normalize(vlad_global, p=2, dim=2)
        vlad_global = vlad_global.view(x.size(0), -1)
        vlad_global = F.normalize(vlad_global, p=2, dim=1)
        
        print("\n\nSome details of returned netvlad stuff")
        print( vlad_local )
        print( len(vlad_local) )
        print("\n\n Global")
        print( vlad_global )
        print( len(vlad_global) )
        
        #return vlad_local, vlad_global  # vlad_local is a list of tensors
        return vlad_global
