import sys
import os

# Add necessary paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Specific site-packages for dependencies if needed
sys.path.append(r"C:\Users\monch\AppData\Local\Programs\Python\Python313\Lib\site-packages")

from argparse import ArgumentParser
import traceback
import utils
import torch
from models.basic_model import CDEvaluator

"""
quick start

sample files in ./samples

save prediction files in the ./samples/predict

"""

def get_args():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--project_name', default='BIT_LEVIR', type=str)
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--checkpoint_root', default='checkpoints', type=str)
    parser.add_argument('--output_folder', default='samples/predict', type=str)

    # data
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--dataset', default='CDDataset', type=str)
    parser.add_argument('--data_name', default='quick_start', type=str)

    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--split', default="demo", type=str)
    parser.add_argument('--img_size', default=256, type=int)

    # model
    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--net_G', default='base_transformer_pos_s4_dd8_dedim8', type=str,
                        help='base_resnet18 | base_transformer_pos_s4_dd8 | base_transformer_pos_s4_dd8_dedim8|')
    parser.add_argument('--checkpoint_name', default='best_ckpt.pt', type=str)

    args = parser.parse_args()
    
    # Set up the checkpoint directory
    args.checkpoint_dir = os.path.join(args.checkpoint_root, "C:\\Users\\monch\\OneDrive\\Документы\\diploma_bit_cd\\Diploma_BiT_CD\\BIT_CD\\checkpoints\\BIT_LEVIR")
    
    return args


if __name__ == '__main__':
   
    args = get_args()
    utils.get_device(args)
    device = torch.device("cuda:%s" % args.gpu_ids[0]
                          if torch.cuda.is_available() and len(args.gpu_ids)>0
                        else "cpu")
    args.checkpoint_dir = os.path.join(args.checkpoint_root, "C:\\Users\\monch\\OneDrive\\Документы\\diploma_bit_cd\\Diploma_BiT_CD\\BIT_CD\\checkpoints\\BIT_LEVIR")
    os.makedirs(args.output_folder, exist_ok=True)

    log_path = os.path.join(args.output_folder, 'log_vis.txt')

    data_loader = utils.get_loader(args.data_name, img_size=args.img_size,
                                   batch_size=args.batch_size,
                                   split=args.split, is_train=False)

    model = CDEvaluator(args)
    model.load_checkpoint(args.checkpoint_name)
    model.eval()

    for i, batch in enumerate(data_loader):
        name = batch['name']
        print('process: %s' % name)
        score_map = model._forward_pass(batch)
        model._save_predictions()
