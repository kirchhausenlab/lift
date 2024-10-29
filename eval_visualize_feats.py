"""
Visualize ViT+LiFT features through feature self-similarity

Code adapted by: Saksham Suri and Matthew Walmer

Sample Usage:
python eval_visualize_feats.py --dataroot /YOUR/PATH/TO/imagenet/val --model dino_vits16 --lift_path pretrained/lift_dino_vits16.pth --output_dir vis/lift_dino_vits16/ --imsize 224
python eval_visualize_feats.py --dataroot /YOUR/PATH/TO/imagenet/val --model dino_vits8 --lift_path pretrained/lift_dino_vits8.pth --output_dir vis/lift_dino_vits8/ --imsize 224
python eval_visualize_feats.py --dataroot /YOUR/PATH/TO/imagenet/val --model dino_vitb16 --lift_path pretrained/lift_dino_vitb16.pth --output_dir vis/lift_dino_vitb16/ --imsize 224
python eval_visualize_feats.py --dataroot /YOUR/PATH/TO/imagenet/val --model dino_vitb8 --lift_path pretrained/lift_dino_vitb8.pth --output_dir vis/lift_dino_vitb8/ --imsize 224
"""
import os
import argparse
import random
from glob import glob

import cv2
import natsort
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from PIL import Image
from torchvision import datasets
from torch.utils.data import Dataset

from lift_extractor import ViTLiFTExtractor
from lift_utils import infer_settings, convert_shape


def compute_cosine_similarity(feat_center, feat):
    cos_sim = torch.zeros(feat.shape[-2], feat.shape[-1])
    for i in range(feat.shape[-2]):
        for j in range(feat.shape[-1]):
            cos_sim[i, j] = F.cosine_similarity(feat_center, feat[:, :, i, j])
    return cos_sim


def make_vis(feat):
    feat_center = feat[:, :, feat.shape[-1]//2, feat.shape[-1]//2]
    cos_sim_feat = compute_cosine_similarity(feat_center, feat)
    cos_sim_feat = cos_sim_feat.cpu().numpy()
    cos_sim_feat = (cos_sim_feat - np.min(cos_sim_feat)) / (np.max(cos_sim_feat) - np.min(cos_sim_feat))
    return cos_sim_feat


def main(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    L = int(args.imsize / args.patch)

    os.makedirs(args.output_dir, exist_ok=True)

    # prep extractor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor = ViTLiFTExtractor(args.model_type, args.lift_path, args.channel, args.patch, args.stride, args.layer, args.facet, device=device)
    extractor.eval()
    extractor_no_lift = ViTLiFTExtractor(args.model_type, None, args.channel, args.patch, args.stride, args.layer, args.facet, device=device, silent=True)
    extractor_no_lift.eval()
    
    # prep data loader
    mean = (0.485, 0.456, 0.406) if "dino" in args.model_type else (0.5, 0.5, 0.5)
    std = (0.229, 0.224, 0.225) if "dino" in args.model_type else (0.5, 0.5, 0.5)
    train_dataset = torchvision.datasets.ImageFolder(args.dataroot, transform=transforms.Compose([
            transforms.ToTensor(),
        ]))
    doub_size = transforms.Compose([
            transforms.Resize((args.imsize*2, args.imsize*2)),
            transforms.Normalize(mean, std),
        ])
    full_size = transforms.Compose([
            transforms.Resize((args.imsize, args.imsize)),
            transforms.Normalize(mean, std),
        ])
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)
    
    unorm = transforms.Normalize(
        mean=[-m / s for m, s in zip(mean, std)],
        std=[1 / s for s in std]
    )
    
    for batch_index, image_batch in enumerate(dataloader):
        if batch_index == args.limit: break
        image_batch = image_batch[0].to(device)
        img_1x = full_size(image_batch)
        img_2x = doub_size(image_batch)
        with torch.no_grad():
            ret = extractor.extract_descriptors_iterative_lift(img_1x, lift_iter=4, return_inter=True)
            ret2x = extractor_no_lift.extract_descriptors(img_2x)
        # DINO 14x14
        feat2 = ret['back']
        feat2 = convert_shape(feat2, L, L)
        cos_sim_feat2 = make_vis(feat2)
        # DINO+Bilinear 14x14 -> 28x28
        feat2_resized = torch.nn.functional.interpolate(feat2, size=(feat2.shape[-2]*2, feat2.shape[-1]*2), mode='bilinear', align_corners=False)
        cos_sim_feat2_resized = make_vis(feat2_resized)
        # DINO 28x28
        feat1 = ret2x
        feat1 = convert_shape(feat1, L*2, L*2)
        cos_sim_feat1 = make_vis(feat1)
        # DINO+LiFT 14x14 -> 28x28
        gen_1 = ret['lift_1']
        gen_1 = convert_shape(gen_1, L*2, L*2)
        cos_sim_gen1 = make_vis(gen_1)
        # DINO+LiFT_2x 14x14 -> 56x56
        gen_2 = ret['lift_2']
        gen_2 = convert_shape(gen_2, L*4, L*4)
        cos_sim_gen2 = make_vis(gen_2)
        # DINO+LiFT_3x 14x14 -> 112x112
        gen_3 = ret['lift_3']
        gen_3 = convert_shape(gen_3, L*8, L*8)
        cos_sim_gen3 = make_vis(gen_3)
        # DINO+LiFT_4x 14x14 -> 224x224
        gen_4 = ret['lift_4']
        gen_4 = convert_shape(gen_4, L*16, L*16)
        cos_sim_gen4 = make_vis(gen_4)
        
        img_1x = unorm(img_1x)

        ##### PLOT 1 - LiFT and Alternatives

        # visualize cos_sim_feat1 as heatmap and original image side by side
        fig, ax = plt.subplots(1, 6)
        fig.set_size_inches(30, 10)

        ax[0].imshow(cos_sim_feat2)
        if not args.hide_marker: ax[0].scatter(cos_sim_feat2.shape[-1]//2, cos_sim_feat2.shape[-1]//2, s=100, c='r', marker='s')

        ax[1].imshow(cos_sim_feat2_resized)
        if not args.hide_marker: ax[1].scatter(cos_sim_feat2_resized.shape[-1]//2, cos_sim_feat2_resized.shape[-1]//2, s=100, c='r', marker='s')

        ax[2].imshow(cos_sim_feat1)
        if not args.hide_marker: ax[2].scatter(cos_sim_feat1.shape[-1]//2, cos_sim_feat1.shape[-1]//2, s=100, c='r', marker='s')

        ax[3].imshow(cos_sim_gen1)
        if not args.hide_marker: ax[3].scatter(cos_sim_gen1.shape[-1]//2, cos_sim_gen1.shape[-1]//2, s=100, c='r', marker='s')

        ax[4].imshow(cos_sim_gen4)
        if not args.hide_marker: ax[4].scatter(cos_sim_gen4.shape[-1]//2, cos_sim_gen4.shape[-1]//2, s=100, c='r', marker='s')

        ax[5].imshow(img_1x[0].cpu().numpy().transpose(1, 2, 0))
        if not args.hide_marker: ax[5].scatter(img_1x.shape[-1]//2, img_1x.shape[-1]//2, s=100, c='r', marker='s')

        # markers
        ax[0].set_title('DINO\n'r'$14{\times}14$', fontsize=args.fontsize)
        ax[1].set_title('DINO + Bilinear\n'r'$14{\times}14\rightarrow28{\times}28$', fontsize=args.fontsize)
        ax[2].set_title('DINO\n'r'$28{\times}28$', fontsize=args.fontsize)
        ax[3].set_title('DINO + LiFT\n'r'$14{\times14}\rightarrow28{\times}28$', fontsize=args.fontsize)
        ax[4].set_title('DINO + LiFT(4x)\n'r'$14{\times14}\rightarrow224{\times}224$', fontsize=args.fontsize)
        ax[5].set_title('Original\nImage', fontsize=args.fontsize)
        # remove ticks
        for i in range(6):
            ax[i].set_xticks([])
            ax[i].set_yticks([])
        fig.tight_layout()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0.05)

        outname = os.path.join(args.output_dir, 'cos_sim_%05i.png'%batch_index)
        print(outname)
        plt.savefig(outname, bbox_inches='tight', pad_inches=0)
        plt.close()

        ##### PLOT 2 - Iterative LiFT

        fig, ax = plt.subplots(1, 6)
        fig.set_size_inches(30, 10)

        ax[0].imshow(cos_sim_feat2)
        if not args.hide_marker: ax[0].scatter(cos_sim_feat2.shape[-1]//2, cos_sim_feat2.shape[-1]//2, s=100, c='r', marker='s')

        ax[1].imshow(cos_sim_gen1)
        if not args.hide_marker: ax[1].scatter(cos_sim_gen1.shape[-1]//2, cos_sim_gen1.shape[-1]//2, s=100, c='r', marker='s')

        ax[2].imshow(cos_sim_gen2)
        if not args.hide_marker: ax[2].scatter(cos_sim_gen2.shape[-1]//2, cos_sim_gen2.shape[-1]//2, s=100, c='r', marker='s')

        ax[3].imshow(cos_sim_gen3)
        if not args.hide_marker: ax[3].scatter(cos_sim_gen3.shape[-1]//2, cos_sim_gen3.shape[-1]//2, s=100, c='r', marker='s')

        ax[4].imshow(cos_sim_gen4)
        if not args.hide_marker: ax[4].scatter(cos_sim_gen4.shape[-1]//2, cos_sim_gen4.shape[-1]//2, s=100, c='r', marker='s')

        ax[5].imshow(img_1x[0].cpu().numpy().transpose(1, 2, 0))
        if not args.hide_marker: ax[5].scatter(img_1x.shape[-1]//2, img_1x.shape[-1]//2, s=100, c='r', marker='s')

        # markers
        ax[0].set_title('DINO\n'r'$14{\times}14$', fontsize=args.fontsize)
        ax[1].set_title('DINO + LiFT(1x)\n'r'$14{\times14}\rightarrow28{\times}28$', fontsize=args.fontsize)
        ax[2].set_title('DINO + LiFT(2x)\n'r'$14{\times14}\rightarrow56{\times}56$', fontsize=args.fontsize)
        ax[3].set_title('DINO + LiFT(3x)\n'r'$14{\times14}\rightarrow112{\times}112$', fontsize=args.fontsize)
        ax[4].set_title('DINO + LiFT(4x)\n'r'$14{\times14}\rightarrow224{\times}224$', fontsize=args.fontsize)
        ax[5].set_title('Original Image\n'r'$224{\times}224$', fontsize=args.fontsize)
        # remove ticks
        for i in range(6):
            ax[i].set_xticks([])
            ax[i].set_yticks([])
        fig.tight_layout()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0.05)

        outname = os.path.join(args.output_dir, 'iter_cos_sim_%05i.png'%batch_index)
        print(outname)
        plt.savefig(outname, bbox_inches='tight', pad_inches=0)
        plt.close()


def parse_args():
    parser = argparse.ArgumentParser('Visual ViT+LiFT feature maps through self-similarity')
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
    ### INPUTS / OUTPUTS ###
    parser.add_argument('--dataroot', default='path/to/data/dir')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--output_dir', default='visualizations', help='dir to save metric plots to')
    parser.add_argument('--imsize', default=224, type=int, help='image resize size')
    ### VIS SETTINGS ###
    parser.add_argument('--hide_marker', action='store_true', help='hide red marker in the center of the image')
    parser.add_argument('--fontsize', default=36, type=int, help='label font size in plots')
    parser.add_argument('--limit', default=20, type=int, help='limit visualization count')
    ###
    args = parser.parse_args()
    infer_settings(args)
    return args
    

if __name__ == '__main__':
    args = parse_args()
    main(args)
    # print max GPU memory used in GB
    print(f'Max GPU memory used: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB')