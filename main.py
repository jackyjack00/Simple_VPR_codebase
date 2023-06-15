import torch
import numpy as np
import torchvision.models
import pytorch_lightning as pl
from torchvision import transforms as tfm
from pytorch_metric_learning import losses , miners
from torch.utils.data.dataloader import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint

import utils
import parser
from datasets.test_dataset import TestDataset
from datasets.train_dataset import TrainDataset
import my_blocks

class LightningModel(pl.LightningModule):
    def __init__(self, val_dataset, test_dataset, descriptors_dim=512, num_preds_to_save=0, save_only_wrong_preds=True , last_pooling_layer = "default", optimizer_str = "default" , lr_scheduler_str = "default", bank = None, proxy_dim = 512):
        # Initialization of class pl.LightningModule, we hinerit from it
        super().__init__()
        # Dataset Parameters
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        # Proxy parameter
        self.bank = bank
        self.proxy_dim = proxy_dim
        # Flag to handle corretly the reset of the bank, exactly once at each epoch start
        #self.epoch_is_starting = True
        # Visualization Parameters
        self.num_preds_to_save = num_preds_to_save
        self.save_only_wrong_preds = save_only_wrong_preds
        # Architecture Parameters
        self.optimizer_str = optimizer_str.lower()
        self.lr_scheduler_str = lr_scheduler_str.lower()
        self.pooling_str = last_pooling_layer.lower()
        # Use as backbone the Resnet18 pretrained on IMAGENET
        self.model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
 
        # Very good code to unpack the resnet and play with it
        #TODO: look if it is nicer to refactor the code by adding here the pooling layer
        #self.model_layers = list( self.model.children() )[:-2]
        #self.model_without_pooling = torch.nn.Sequential( *[ _ for _ in self.model_layers ] )
        
        #  Change the model's pooling layer according to the command line parameter, "default" is avg_pooling of Resnet18
        if self.pooling_str == "gem":
            self.model.avgpool = my_blocks.GeMPooling(feature_size = self.model.fc.in_features , pool_size = 7, init_norm = 3.0, eps = 1e-6, normalize = False)
        elif self.pooling_str == "cosplace":
            #changed to a version found in prof repo
            self.model.avgpool = my_blocks.CosPlaceAggregator( in_dim = self.model.fc.in_features , out_dim = descriptors_dim)
        elif self.pooling_str == "mixvpr":
            #lowered dimensions to 256, 4 from 512, 1
            self.mixvpr_out_channels = 256 
            self.mixvpr_out_rows = 4
            # MixVPR works with an input of dimension [batch_size, 512, 7,7] == [batch_size, in_channels, in_h, in_w]
            self.model.avgpool = my_blocks.MixVPR( in_channels = self.model.fc.in_features, in_h = 7, in_w = 7, out_channels = self.mixvpr_out_channels , out_rows =  self.mixvpr_out_rows )
        
        # Initialize output dim as the standard one of CNN
        self.aggregator_out_dim = self.model.fc.in_features
        # Change the output of the FC layer to the desired descriptors dimension
        if self.pooling_str == "cosplace":
            self.aggregator_out_dim = descriptors_dim
            self.model.fc = torch.nn.Linear(self.aggregator_out_dim , descriptors_dim) 
        elif self.pooling_str == "mixvpr":
            # MixVPR take as input the final activation map of dim [n_batch,512,7,7] and outputs a feature vector for each batch [n_batch, out_channels * out_rows]
            self.aggregator_out_dim  = self.mixvpr_out_channels * self.mixvpr_out_rows
            self.model.fc =   torch.nn.Linear(self.aggregator_out_dim, descriptors_dim) 
        else:
            # Simply map the output of Resnet18 avg_pooling to a desired dimension
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, descriptors_dim) 
                                                
        # Define the ProxyHead Layer
        self.proxy_head = my_blocks.ProxyHead( descriptors_dim , proxy_dim )
        # Define an indipendent Loss for this Layer
        self.loss_head = losses.MultiSimilarityLoss( alpha=1, beta=50, base=0.0 )
        #self.loss_head = losses.ContrastiveLoss(pos_margin=0.0, neg_margin=1)
        
        # Set a miner
        #self.miner_fn = None
        #self.miner_fn = miners.TripletMarginMiner(margin=0.2, type_of_triplets="hard")
        #self.miner_fn = miners.PairMarginMiner(pos_margin=0.2, neg_margin=0.8)
        self.miner_fn = miners.MultiSimilarityMiner( epsilon=0.1 )
        # Set the loss function
        #self.loss_fn = losses.ContrastiveLoss(pos_margin=0.0, neg_margin=1)
        #self.loss_fn = losses.TripletMarginLoss(margin=0.05)
        #self.loss_fn = losses.MultiSimilarityLoss( alpha=2, beta=50, base=0.5 )
        self.loss_fn = losses.MultiSimilarityLoss( alpha=1, beta=50, base=0.0 )

    def forward(self, images):
        descriptors = self.model(images)
        if bank is not None:
            proxies = self.proxy_head(descriptors)
        else:
            proxies = None
        return descriptors , proxies

    def configure_optimizers(self):
        if self.optimizer_str == "default":
            optimizer = torch.optim.SGD(self.parameters(), lr=0.001, weight_decay=0.001, momentum=0.9)
        elif self.optimizer_str == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        elif self.optimizer_str == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
        elif  self.optimizer_str == "nadam":
            optimizer = torch.optim.NAdam(self.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, momentum_decay=0.004)
            
        if self.lr_scheduler_str == "exponential" :
            scheduler = [ torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = .95) ]
        else :
            scheduler = []
        return [optimizer] , scheduler

    # The loss function call (this method will be called at each training iteration)
    def loss_function(self, descriptors, labels):
        # Loss with miner
        if self.miner_fn is not None:
            # Compute the pairs to use in the loss using the miner function specified in __init__ 
            miner_output = self.miner_fn(descriptors , labels)
            # Compute the loss using the loss function specified in __init__ and the miner output
            loss = self.loss_fn(descriptors, labels, miner_output)
        # Loss without any miner
        else:
            # Compute the loss using the loss function specified in __init__
            loss = self.loss_fn(descriptors, labels)
        return loss

    # This is the training step that's executed at each iteration
    def training_step(self, batch, batch_idx):
        # Modify dataset dimension to have [all_images_in_batch, C, H, W]
        images, labels = batch
        num_places, num_images_per_place, C, H, W = images.shape
        images = images.view(num_places * num_images_per_place, C, H, W)
        labels = labels.view(num_places * num_images_per_place)

        # Feed forward the batch to the model
        descriptors, proxies = self(images)  # Here we are calling the method forward that we defined above
        
        # Call the loss_function of the main architecture  
        loss = self.loss_function(descriptors, labels)
        
        # Update the bank and compute the loss for training the ProxyHead
        if self.bank is not None:
            self.bank.update_bank(proxies , labels)
            loss_head = self.loss_head(proxies, labels)
            loss = loss + loss_head
        
        # Log the result
        self.log('loss', loss.item(), logger=True)
        return {'loss': loss}

    # For validation and test, we iterate step by step over the validation set
    def inference_step(self, batch):
        images, _ = batch
        descriptors, proxy = self(images)
        return descriptors.cpu().numpy().astype(np.float32)

    def validation_step(self, batch, batch_idx):
        return self.inference_step(batch)

    def test_step(self, batch, batch_idx):
        return self.inference_step(batch)

    def validation_epoch_end(self, all_descriptors):
        return self.inference_epoch_end(all_descriptors, self.val_dataset)

    def test_epoch_end(self, all_descriptors):
        return self.inference_epoch_end(all_descriptors, self.test_dataset, self.num_preds_to_save)

    def inference_epoch_end(self, all_descriptors, inference_dataset, num_preds_to_save=0):
        # Update the bank index at the end of each epoch
        if self.bank is not None:
            self.bank.update_index()
        
        """all_descriptors contains database then queries descriptors"""
        all_descriptors = np.concatenate(all_descriptors)
        queries_descriptors = all_descriptors[inference_dataset.database_num : ]
        database_descriptors = all_descriptors[ : inference_dataset.database_num]

        recalls, recalls_str = utils.compute_recalls(
            inference_dataset, queries_descriptors, database_descriptors,
            trainer.logger.log_dir, num_preds_to_save, self.save_only_wrong_preds
        )
        print(recalls_str)
        self.log('R@1', recalls[0], prog_bar=False, logger=True)
        self.log('R@5', recalls[1], prog_bar=False, logger=True)

def get_datasets_and_dataloaders(args, bank = None):
    # Define Transformation to apply
    #TODO: augment contrast
    train_transform = tfm.Compose([
        tfm.RandAugment(num_ops=3),
        tfm.ToTensor(),
        tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # Define Datasets
    train_dataset = TrainDataset(
        dataset_folder=args.train_path,
        img_per_place=args.img_per_place,
        min_img_per_place=args.min_img_per_place,
        transform=train_transform
    )
    val_dataset = TestDataset(dataset_folder=args.val_path)
    test_dataset = TestDataset(dataset_folder=args.test_path)
    # Define dataloaders, train one has with proxy and without proxy case
    if bank is not None:
        # Define Proxy Sampler that uses ProxyBank
        my_proxy_sampler = my_blocks.ProxyBankBatchMiner( train_dataset, args.batch_size , bank , num_workers = args.num_workers )
        train_loader = DataLoader(dataset=train_dataset, batch_sampler = my_proxy_sampler, num_workers=args.num_workers)
    else:
        my_random_sampler = my_blocks.MyRandomSampler( train_dataset,  args.batch_size )
        train_loader = DataLoader(dataset=train_dataset, batch_sampler = my_random_sampler, num_workers=args.num_workers)
        #train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Parse the arguments
    args = parser.parse_arguments()
    # Define the bank
    if args.proxy is not None:
        proxy_dim = 512
        bank = my_blocks.ProxyBank(proxy_dim)
    else:
        bank = None
    # Compute all the data related object
    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = get_datasets_and_dataloaders(args, bank)
    # Load the chekpoint if available or else rebuild it
    
    if args.ckpt_path is not None:
      model_args = {
        "val_dataset" : val_dataset,
        "test_dataset" : test_dataset,
        "last_pooling_layer" : args.pooling_layer,
        "optimizer_str" : args.optimizer, 
        "lr_scheduler_str" : args.lr_scheduler, 
        "bank" : bank
        }
      model = LightningModel.load_from_checkpoint(args.ckpt_path, **model_args)
    else:
      model_args = {
        "last_pooling_layer" : args.pooling_layer,
        "optimizer_str" : args.optimizer,
        "lr_scheduler_str" : args.lr_scheduler
        }
      model = LightningModel(val_dataset, test_dataset, args.descriptors_dim, args.num_preds_to_save, args.save_only_wrong_preds, bank = bank, **model_args)
    
    # Model params saving using Pytorch Lightning. Save the best 3 models according to Recall@1
    checkpoint_cb = ModelCheckpoint(
        monitor='R@1',
        filename='_epoch({epoch:02d})_step({step:04d})_R@1[{val/R@1:.4f}]_R@5[{val/R@5:.4f}]',
        auto_insert_metric_name=False,
        save_weights_only=True,
        save_top_k=3,
        mode='max'
    )

    # Instantiate a trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[0],
        default_root_dir='./LOGS',  # Tensorflow can be used to viz
        num_sanity_val_steps=0,  # runs a validation step before stating training
        precision=16,  # we use half precision to reduce  memory usage
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=1,  # run validation every epoch
        callbacks=[checkpoint_cb],  # we only run the checkpointing callback (you can add more)
        reload_dataloaders_every_n_epochs=1,  # we reload the dataset to shuffle the order
        log_every_n_steps=20,
    )
    
    # Train only if specified, else test only with a pretrained model
    if args.only_test is None:
      trainer.validate(model=model, dataloaders=val_loader)
      trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(model=model, dataloaders=test_loader)

