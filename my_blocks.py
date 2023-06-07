import torch
import torch.nn as nn
import torchvision.models

class GeMPooling(nn.Module):
    def __init__(self, feature_size, pool_size=7, init_norm=3.0, eps=1e-6, normalize=False, **kwargs):
        # Initialization of super class
        super(GeMPooling, self).__init__(**kwargs)
        # Final layer channel size, the pow calc at -1 axis
        self.feature_size = feature_size
        # Dimension to pool, if equal to h and w output will be 1 dim for each channel
        self.pool_size = pool_size
        # Use a learnable parameter p to compute not the standard average but a generalized one where data is ^p and the sum is rooted by p
        # We want to train a p for each channel of the result of CNN --> p has dim of last channel, then it is broadcasted to the img dimension
        self.p = torch.nn.Parameter(torch.ones(self.feature_size) * init_norm, requires_grad=True)
        # Define the avg_pooling layer, it takes in [n_batch, 512, 7, 7] and outputs [n_batch, 512, 1, 1]
        self.avg_pooling = nn.AvgPool2d((self.pool_size, self.pool_size))
        # Parameter for clamp
        self.eps = eps
        # Decide if perform final normalization or not
        self.normalize = normalize

    def forward(self, features):
        # Feature tensor size is [n_batch , n_channels ,H , W] , p tensor should be broadcastable to feature size
        # Features is [batch , 512 , 7 , 7] p is [512] so manually reshape it to manage the broadcasting
        my_p = self.p.reshape((512,1,1))
        # Put a min value possible in the tensor, then computes the "to the power of my_p (p broadcasted)"
        features = features.clamp(min=self.eps).pow(my_p)
        # Standard avg pooling operation, takes in [n_batch, 512, 7, 7] and outputs [n_batch, 512, 1, 1]
        features = self.avg_pooling(features)
        # Eliminates the uselss dimensions of 1, therefore from [n_batch, 512, 1, 1] to [n_batch, 512]
        features = torch.squeeze(features)
        # Root of power 1/p, this time broadcast is correctly done automatically
        features = torch.pow(features, (1.0 / self.p))
        
        # If you want to normalize output featurs to a unit vector 
        if self.normalize:
            features = F.normalize(features, dim=-1, p=2)
        return features
########################################################################################################################################
class CosPlaceAggregator(nn.Module):
    
    def __init__ (self , in_dim , out_dim ):
        super(CosPlaceAggregator,self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.gem = GeMPooling( feature_size = self.in_dim )
        self.projector = nn.Linear( in_dim , out_dim )
    
    def forward(self , x):
        # L2 normalization done on each feature map
        x = F.normalize(x , p = 2.0 , dim = 1)
        # Generalized Mean Pooling
        x = self.gem( x )
        # Flatten to [batch_size , 512]
        x = x.flatten( dim = 1 )
        # Change dimensionality from in_dim -> out_dim
        x = self.projector( x )
        # L2 Normalization done on the resulting feature vectors
        x = F.normalize(x , p = 2.0 , dim = 1)
        return x
    
#######################################################################################################################################
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
        # Forward uses a skip connection and the Mixer layer defined above
        return x + self.mix(x)


class MixVPR(nn.Module):
    def __init__(self,
                 in_channels=512,
                 in_h=7,
                 in_w=7,
                 out_channels=512,
                 mix_depth=4,
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
        # Build the L MixerLayer in a cascade architecture
        self.mix = nn.Sequential(*[
            FeatureMixerLayer(in_dim=hw, mlp_ratio=mlp_ratio)
            for _ in range(self.mix_depth)
        ])
        self.channel_proj = nn.Linear(in_channels, out_channels)
        self.row_proj = nn.Linear(hw, out_rows)

    def forward(self, x):
        # x is [batch_ize, 512, 7, 7] and flattened to [batch_size, 512, 49] from now on we refer as h*w = 49 dimension as "row"
        x = x.flatten(2)
        # mix layer preserves dimension, so it is still [batch_size, 512 , 49]
        x = self.mix(x)
        # Change the order of last two dimension of x from [batch_size, 512 , 49] to [batch_size, 49, 512]
        x = x.permute(0, 2, 1)
        # Reduce dimensionality of channels via Linear Layer from [batch_size , 49, 512] to [batch_size, 49, out_channels]
        x = self.channel_proj(x)
        # Come back to original order of dimension [batch_size, out_channels, 49]
        x = x.permute(0, 2, 1)
        # Reduce dimensionality of h*w called "row" via Linear Layer from [batch_size, out_channels, 49] to [batch_size, out_channels, out_rows]
        x = self.row_proj(x)
        # Produces an output of shape [batch_size, out_channels*out_channels]
        x = F.normalize(x.flatten(1), p=2, dim=-1)
        return x
    
########################################################################################################################################    
import random
import faiss 
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from torch.utils.data.sampler import RandomSampler

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
            
    # Compute the average proxy representation of all the proxy accumulated
    def get_avg(self):
        return torch.Tensor( self.proxy_sum / self.n )
    
    # Manually define how the "+" operation between ProxyAccumulator is done
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

#TODO: initialize it in main and pass it to get_dataloaders (then pass it to dataloader batch_sampler) and to the model (bank_update() at end epoch)
class ProxyBank():
    def __init__(self, proxy_dim = 512):
        self.proxy_dim = proxy_dim
        # Initialize an index containing vectors of dim equals to the proxy
        self.__base_index = faiss.IndexFlatL2( self.proxy_dim )
        # Wrap it in order to use user defined indeces (place labels)
        self.__index = faiss.IndexIDMap( self.__base_index )
        # Initialize a dictionary to summarize the proxy-place_label relation
        self.__bank = {}
        
    # Given the Proxies computed by ProxyHead and their lables
    def update_bank(self, proxies, labels):
        # Iterate over each pair proxy-label where proxy is the result of projection done by ProxyHead
        for proxy, label in zip(proxies , labels):
            # From Tensor to int to be usable in dictionary as key
            label = int(label)
            # Create or Update the content of the bank dictionary
            if label not in self.__bank.keys():
                self.__bank[label] = ProxyAccumulator( tensor = proxy , n = 1 )
            else:
                self.__bank[label] = ProxyAccumulator( tensor = proxy , n = 1 ) + self.__bank[label]
    
    # You first popolate the ProxyBank and then popolate the index for retrieval
    def update_index(self):
        # Override the index at each epoch. Do not stuck info from differet epoch (training should handle this aspect)
        self.__index.reset()
        for label, proxy_acc in self.__bank.items():
            # Use get_avg() to go from accumulator to average and compute the global proxy for each place
            self.__index.add_with_ids( proxy_acc.get_avg().reshape(1,-1).detach().cpu() , label )
    
    # Empty all the dictionaries and indeces created so far
    def reset(self):
        del self.__bank
        del self.__index
        self.__bank = {}
        self.__base_index = faiss.IndexFlatL2( self.proxy_dim )
        self.__index = faiss.IndexIDMap( self.__base_index )
    
    # Generates the batch using the information inside the bank: proxy that are near each other are sampled in the same batch
    def batch_sampling(self , batch_dim):
        batches = []
        # While places are enough to generate the KNN
        while len(self.__bank) >= batch_dim:
            # Extract from bank a random label-proxy related to a place
            rand_index = random.randint( 0 , len(self.__bank) - 1 )
            rand_bank_item = list( self.__bank.items() )[rand_index]
            # Inside bank i have ProxyAccumulator --> get_avg gives me the Proxy
            starting_proxy = rand_bank_item[1].get_avg()
            # Compute the batch_size_Nearest_Neighbours with faiss_index w.r.t. the extracted proxy
            distances, batch_of_labels = self.__index.search( starting_proxy.reshape(1,-1).detach().cpu(), batch_dim )
            # Faiss return a row per query in a multidim np.array, extract the one row
            batch_of_labels = batch_of_labels.flatten()
            # Add the new generated batch the one alredy created. KNN contains the starting proxy itself. Labels is the new Batch
            batches.append( batch_of_labels )
            # Remove all the already picked places from the index and the bank (no buono)
            for key_to_del in batch_of_labels:
                del self.__bank[ key_to_del ]
            self.__index.remove_ids( batch_of_labels )
        #TODO: bring this at epoch start
        # Call a reset in order to fully empty the stored elements
        self.reset()
        # Output the batches
        return batches 

# Initialized in get_dataloader and passed to dataloader of train dataset as batch_sampler
# At epoch 0 return random sampled minibatches
# Next epochs it uses the bank to compute the batches
class ProxyBankBatchMiner(Sampler):
    def __init__(self, dataset, batch_size, bank , num_workers = 1):
        # Epoch counter
        self.is_first_epoch = True
        # Save dataset
        self.dataset = dataset
        # Set dim of batch
        self.batch_size = batch_size
        # Compute the floor of the length of the iterable
        self.iterable_size = len(dataset) // batch_size
        # This is our ProxyBank
        self.bank = bank
        # Workaround, because pytorch lightning call 2 times iter at each epoch
        self.num_workers = num_workers
        self.counter = 0
        self.batch_iterable = []
        
    # Return an iterable over a list of groups of indeces (list of batches)
    def __iter__(self): 
        # Epoch 0 case
        if self.is_first_epoch and self.counter % self.num_workers == 0:
        #if self.is_first_epoch:
            # Change flag, first epoch is done
            self.is_first_epoch = False
            # Generate a random order of the indeces of the dataset, inside the parentesis there is the len of the dataset
            random_indeces_perm = torch.randperm( len( self.dataset ) )
            # Generate a fixed size partitioning of indeces
            batches =  torch.split( random_indeces_perm , self.batch_size )
            self.batch_iterable = iter(batches)
        # Epochs where Bank is informative, after epoch 0
        elif self.counter % self.num_workers == 0:
        #else:
            # Generate batches from ProxyBank
            batches = self.bank.batch_sampling( self.batch_size )
            self.batch_iterable = iter(batches)
        self.counter += 1
        return  self.batch_iterable
    
    # Return the length of the generated iterable, the one over the batches
    def __len__(self):
        return self.iterable_size

# A dummy class to understand how batch_sampler of DataLoader works
class MyRandomSampler(Sampler):
    def __init__(self, dataset, batch_size):
        # Save dataset
        self.dataset = dataset
        # Set dim of batch
        self.batch_size = batch_size
        # Compute the floor of the length of the iterable
        self.iterable_size = len(dataset) // batch_size
        
    # Return an iterable over a list of groups of indeces (list of batches_idx)
    def __iter__(self): 
        # Generate a random order of the indeces of the dataset, inside the parentesis there is the len of the dataset
        random_indeces_perm = torch.randperm( len( self.dataset ) )
        # Generate a fixed size partitioning of indeces
        batches =  torch.split( random_indeces_perm , self.batch_size )
        batches_iterable = iter(batches)
        return batches_iterable
    
    # Return the length of the generated iterable, the one over the batches
    def __len__(self):
        return self.iterable_size
