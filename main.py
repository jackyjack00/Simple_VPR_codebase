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
    def __init__(self, val_dataset, test_dataset, descriptors_dim=512, num_preds_to_save=0, save_only_wrong_preds=True , last_pooling_layer = "default", optimizer_str = "default", bank = None, proxy_dim = 512):
        # Initialization of class pl.LightningModule, we hinerit from it
        super().__init__()
        # Dataset Parameters
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        # Proxy parameter
        self.bank = bank
        self.proxy_dim = proxy_dim
        # Visualization Parameters
        self.num_preds_to_save = num_preds_to_save
        self.save_only_wrong_preds = save_only_wrong_preds
        # Architecture Parameters
        self.optimizer_str = optimizer_str.lower()
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
        elif self.pooling_str == "netvlad":
            #changed to a version found in prof repo
            self.model.avgpool = my_blocks.NetVLAD(num_clusters = 64, dim = self.model.fc.in_features)
        elif self.pooling_str == "mixvpr":
            #TODO: lower dimension like 256, 4
            self.mixvpr_out_channels = 512
            self.mixvpr_out_rows = 4
            # MixVPR works with an input of dimension [n_batch, 512, 7,7] == [n_batch, in_channels, in_h, in_w]
            self.model.avgpool = my_blocks.MixVPR( in_channels = self.model.fc.in_features, in_h = 7, in_w = 7, out_channels = self.mixvpr_out_channels , out_rows =  self.mixvpr_out_rows )
        
        # Initialize output dim as the standard one of CNN
        self.aggregator_out_dim = self.model.fc.in_features
        # Change the output of the FC layer to the desired descriptors dimension
        if self.pooling_str == "netvlad":
            # VLAD like architecture generates in_features*n_clusters outputs
            self.aggregator_out_dim = self.model.fc.in_features * 64
            self.model.fc = torch.nn.Linear(self.aggregator_out_dim , descriptors_dim)
        elif self.pooling_str == "mixvpr":
            # MixVPR take as input the final activation map of dim [n_batch,512,7,7] and outputs a feature vector for each batch [n_batch, out_channels * out_rows]
            self.aggregator_out_dim  = self.mixvpr_out_channels * self.mixvpr_out_rows
            self.model.fc = torch.nn.Linear(self.aggregator_out_dim, descriptors_dim)
        else:
            # Simply map the output of Resnet18 avg_pooling to a desired dimension
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, descriptors_dim)
            
        # Define ProxyHead if necessary
        if self.bank is not None:
            print("Ciao, inizializzo il proxy")
            self.proxy_head = my_blocks.ProxyHead( descriptors_dim , proxy_dim )
        
        # Set a miner
        # self.miner_fn = miners.PairMarginMiner(pos_margin=0.2, neg_margin=0.8)
        self.miner_fn = miners.MultiSimilarityMiner( epsilon=0.1 )
        # Set the loss function
        #self.loss_fn = losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
        #self.loss_fn = losses.MultiSimilarityLoss( alpha=2, beta=50, base=0.5 )
        self.loss_fn = losses.MultiSimilarityLoss( alpha=1, beta=50, base=0.0 )

    def forward(self, images):
        descriptors = self.model(images)
        proxies = self.proxy_head(descriptors)
        return descriptors , proxies

    def configure_optimizers(self):
        if self.optimizer_str == "default":
            optimizers = torch.optim.SGD(self.parameters(), lr=0.001, weight_decay=0.001, momentum=0.9)
        elif self.optimizer_str == "adam":
            optimizers = torch.optim.Adam(self.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        elif self.optimizer_str == "adamw":
            optimizers = torch.optim.AdamW(self.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
        return optimizers

    # The loss function call (this method will be called at each training iteration)
    def loss_function(self, descriptors, labels):
        # Include a miner for loss'pair selection
        miner_output = self.miner_fn(descriptors , labels)
        # Compute the loss using the loss function specified in __init__ and the miner output ?instead of all possible batch pairs?
        loss = self.loss_fn(descriptors, labels, miner_output)
        return loss

    # This is the training step that's executed at each iteration
    def training_step(self, batch, batch_idx):
        images, labels = batch
        num_places, num_images_per_place, C, H, W = images.shape
        images = images.view(num_places * num_images_per_place, C, H, W)
        labels = labels.view(num_places * num_images_per_place)

        # Feed forward the batch to the model
        descriptors, proxies = self(images)  # Here we are calling the method forward that we defined above
        # Update the bank
        if self.bank is not None:
            self.bank.update_bank(proxies , labels)
        # Call the loss_function we defined above  
        loss = self.loss_function(descriptors, labels)  
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

def get_datasets_and_dataloaders(args, bank=None):
    # Define Transformation to apply
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
        my_proxy_sampler = my_blocks.ProxyBankBatchMiner( train_dataset, args.batch_size , bank )
        train_loader = DataLoader(dataset=train_dataset, batch_sampler = my_proxy_sampler, num_workers=args.num_workers)
    else:
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)
    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Parse the arguments
    args = parser.parse_arguments()
    # Define the bank
    proxy_dim = 512
    bank = my_blocks.ProxyBank(proxy_dim)
    # Compute all the data related object
    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = get_datasets_and_dataloaders(args, bank)
    # Load the chekpoint if available or else rebuild it
    
    if args.ckpt_path is not None:
      model_args = {
        "val_dataset" : val_dataset,
        "test_dataset" : test_dataset,
        "last_pooling_layer" : args.pooling_layer,
        "optimizer_str" : args.optimizer
        }
      model = LightningModel.load_from_checkpoint(args.ckpt_path, **model_args)
    else:
      model_args = {
        "last_pooling_layer" : args.pooling_layer,
        "optimizer_str" : args.optimizer
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

