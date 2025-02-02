
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=64,
                        help="The number of places to use per iteration (one place is N images)")
    parser.add_argument("--img_per_place", type=int, default=4,
                        help="The effective batch size is (batch_size * img_per_place)")
    parser.add_argument("--min_img_per_place", type=int, default=4,
                        help="places with less than min_img_per_place are removed")
    parser.add_argument("--max_epochs", type=int, default=20,
                        help="stop when training reaches max_epochs")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="number of processes to use for data loading / preprocessing")

    # Architecture parameters
    parser.add_argument("--descriptors_dim", type=int, default=512,
                        help="dimensionality of the output descriptors")
    
    parser.add_argument("--pooling_layer", type = str, default="default",
                        help="change the last pooling layer.Supported so far:\n'default'\n'GeM'\n'MixVPR\nCosPlace")
    
    parser.add_argument("--optimizer", type = str, default="default",
                        help="change the optimizer for the parameters update:\n'default'\n'Adam'\n'AdamW'\n'Nadam")
    
    parser.add_argument("--lr_scheduler", type = str, default="default",
                        help="change the learning rate scheduler for the parameters update:'\n'Exponential'")
    
    parser.add_argument("--proxy", type = str, default = None,
                        help="Set to True to enable the proxy mining operation to generate batches")
    
    parser.add_argument("--loss", type=str, default="contrastive",
                        help="loss function:'\n'Contrastive'\n'Triplet'\n'Multisimilarity")
    
    # Visualizations parameters
    parser.add_argument("--num_preds_to_save", type=int, default=0,
                        help="At the end of training, save N preds for each query. "
                        "Try with a small number like 3")
    parser.add_argument("--save_only_wrong_preds", action="store_true",
                        help="When saving preds (if num_preds_to_save != 0) save only "
                        "preds for difficult queries, i.e. with uncorrect first prediction")

    # Paths parameters
    parser.add_argument("--train_path", type=str, default="data/gsv_xs/train",
                        help="path to train set")
    parser.add_argument("--val_path", type=str, default="data/sf_xs/val",
                        help="path to val set (must contain database and queries)")
    parser.add_argument("--test_path", type=str, default="data/sf_xs/test",
                        help="path to test set (must contain database and queries)")

    # Checkpoint parameters
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="path to retrive the chekpoint of the model")

    # Only test 
    parser.add_argument("--only_test", type=str, default=None,
                        help="Set to True if you want to avoid the train phase and test on --test_path")
    
    args = parser.parse_args()
    return args

