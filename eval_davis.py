"""
Evaluate ViT+LiFT performance on DAVIS video segmentation. Based on https://github.com/davisvideochallenge/davis2017-evaluation and https://github.com/facebookresearch/dino/blob/main/eval_video_segmentation.py

Code adapted by: Saksham Suri and Matthew Walmer

PART 1: First run the below to generate outputs
python eval_davis.py --dataroot /YOUR/PATH/TO/DAVIS --model_type dino_vits16 --lift_path pretrained/lift_dino_vits16.pth --output_dir davis_outputs/lift_dino_vits16 --imsize 224 
PART 2: Then the below command computes the scores
python davis2017-evaluation/evaluation_method.py --davis_path /YOUR/PATH/TO/DAVIS --task semi-supervised --results_path davis_outputs/lift_dino_vits16/davis_vidseg_224_layer11 --imsize 224
"""
import os
import argparse
import time
import math

import torch
import torch.multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFile, ImageColor
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from urllib.request import urlopen

from lift_extractor import ViTLiFTExtractor
from lift_utils import infer_settings
from eval_video_segmentation import eval_video_tracking_davis, read_seg, read_frame_list

torch.multiprocessing.set_sharing_strategy('file_system')


def run_video_segmentation(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor = ViTLiFTExtractor(args.model_type, args.lift_path, args.channel, args.patch, args.stride, args.layer, args.facet, device=device)
    extractor.eval()
    mean = (0.485, 0.456, 0.406) if "dino" in args.model_type else (0.5, 0.5, 0.5)
    std = (0.229, 0.224, 0.225) if "dino" in args.model_type else (0.5, 0.5, 0.5)
    transform = transforms.Compose([
            transforms.Resize((args.imsize, args.imsize), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    color_palette = []
    for line in urlopen("https://raw.githubusercontent.com/Liusifei/UVC/master/libs/data/palette.txt"):
        color_palette.append([int(i) for i in line.decode("utf-8").split('\n')[0].split(" ")])
    color_palette = np.asarray(color_palette, dtype=np.uint8).reshape(-1,3)
    video_list = open(os.path.join(args.dataroot, "ImageSets/2017/val.txt")).readlines()
    for i, video_name in enumerate(video_list):
        video_name = video_name.strip()
        print(f'[{i}/{len(video_list)}] Begin to segmentate video {video_name}.')
        video_dir = os.path.join(args.dataroot, "JPEGImages/480p/", video_name)
        frame_list = read_frame_list(video_dir)
        seg_path = frame_list[0].replace("JPEGImages", "Annotations").replace("jpg", "png")
        if args.lift_path is not None:
            if('16' in args.model_type and args.imsize == 56):
                first_seg, seg_ori = read_seg(seg_path, args.patch//2, scale_size=[args.imsize-4, args.imsize-4])
            else:
                first_seg, seg_ori = read_seg(seg_path, args.patch//2, scale_size=[args.imsize, args.imsize])
        else:
            if(args.stride!=args.patch):
                op_size = (args.imsize - args.patch)//args.stride + 1
                first_seg, seg_ori = read_seg(seg_path, math.ceil(args.patch*(args.stride/args.patch)), scale_size=[args.imsize, args.imsize], custom=op_size)
            else:
                first_seg, seg_ori = read_seg(seg_path, args.patch, scale_size=[args.imsize, args.imsize])
        eval_video_tracking_davis(args, extractor, transform, frame_list, video_dir, first_seg, seg_ori, color_palette)


def parse_args():
    parser = argparse.ArgumentParser('DAVIS Object Segmentation Evaluation')
    ### BACKBONE ###
    parser.add_argument('--model_type', default='dino_vits8', type=str,
                        help="""type of model to extract. 
                        Choose from [dino_vits8 | dino_vits16 | dino_vitb8 | dino_vitb16 | vit_small_patch8_224 | 
                        vit_small_patch16_224 | vit_base_patch8_224 | vit_base_patch16_224]""")
    parser.add_argument('--facet', default='key', type=str, help="""facet to create descriptors from. 
                        options: ['key' | 'query' | 'value' | 'token']""")
    parser.add_argument('--channel', default=None, type=int, help='backbone output channels (default: inferred from --model_type)')
    parser.add_argument('--patch', default=None, type=int, help='backbone patch size (default: inferred from --model_type)')
    parser.add_argument('--stride', default=None, type=int, help='stride of first convolution layer. small stride -> higher resolution. (default: equal to --patch)')
    parser.add_argument('--layer', default=None, type=int, help='layer to create descriptors from. (default: last layer)')
    ### LIFT ###
    parser.add_argument('--lift_path', default=None, type=str, help='path of pretrained LiFT model to use. If not given, lift LiFT is not used')
    ### DATASET ###
    parser.add_argument('--dataroot', default='path/to/DAVIS')
    parser.add_argument('--output_dir', default='temp_davis_test', help='dir to save metric plots to')
    parser.add_argument('--imsize', default=224, type=int, help='image resize size')
    parser.add_argument("--n_last_frames", type=int, default=7, help="number of preceeding frames")
    parser.add_argument("--size_mask_neighborhood", default=12, type=int,
                        help="We restrict the set of source nodes considered to a spatial neighborhood of the query node")
    parser.add_argument("--topk", type=int, default=5, help="accumulate label from top k neighbors")
    ###
    args = parser.parse_args()
    infer_settings(args)
    return args


if __name__ == '__main__':
    args = parse_args()
    start_time = time.time()
    run_video_segmentation(args)
    print(args.model_type, args.imsize, args.stride)
    # print total time taken in minutes
    print(f'Total time taken: {(time.time() - start_time) / 60:.2f} minutes')
    # print max GPU memory used in GB
    print(f'Max GPU memory used: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB')