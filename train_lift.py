"""
Train a LiFT module for a particular ViT backbone on ImageNet or an image folder dataset you specify.

Code by: Saksham Suri and Matthew Walmer

Sample Usage:
python train_lift.py --dataroot /YOUR/PATH/TO/imagenet/train/ --model_type dino_vits16 --save_every 1 --epochs 5 --lr 0.001 --augment --loss cosine --batch_size 256 --output_dir lift_imagenet_trains
python train_lift.py --dataroot /YOUR/PATH/TO/imagenet/train/ --model_type dino_vits8 --save_every 1 --epochs 5 --lr 0.001 --augment --loss cosine --batch_size 64 --output_dir lift_imagenet_trains
"""

import os
import argparse
import json
import random
import time
from glob import glob

import natsort
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from PIL import Image

from extractor import ViTExtractor
from lift import LiFT
from lift_utils import infer_settings, convert_shape


def my_extractor(args):
    """
    To train LiFT for your own ViT backbone, fill out the code here to load your model.

    This function must return a feature extractor whose forward function takes an
    input image with shape [B,3,H,W] and returns a feature array of shape [B,T,C]
    where T represents the token dimension.

    IMPORTANT: The extractor must be able to handle variable input resolutions, similar
    to the ViT implementation of https://github.com/facebookresearch/dino. If your ViT
    does not natively support variable input size, see extractor.py for an example of
    loading a non-DINO ViT's weights into a DINO ViT implementation.
    """
    my_extractor = None

    assert my_extractor is not None
    return my_extractor


def main(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    output_dir = (
        args.output_dir + "/" + args.model_type + "_" + str(args.lr) + "_" + args.loss
    )
    if args.augment:
        output_dir = output_dir + "_aug"
    else:
        output_dir = output_dir + "_noaug"
    output_dir = output_dir + "_" + str(args.imsize)

    os.makedirs(output_dir, exist_ok=True)
    print("Output dir: ", output_dir)

    # prepare backbone and lift
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.my_extractor:
        extractor = my_extractor(args)
    else:
        extractor = ViTExtractor(args.model_type, args.patch, device=device)
    lift = LiFT(args.channel, args.patch, pre_shape=False, post_shape=False).to(device)
    img_paths = glob(os.path.join(args.dataroot, "*.jpg"))

    # feature array sizes
    L1 = int(args.imsize / args.patch)
    L2 = int(L1 / 2)
    L4 = int(L1 / 4)

    mean = (0.485, 0.456, 0.406) if "dino" in args.model_type else (0.5, 0.5, 0.5)
    std = (0.229, 0.224, 0.225) if "dino" in args.model_type else (0.5, 0.5, 0.5)
    if "imagenet" in args.dataroot and not args.augment:
        train_dataset = torchvision.datasets.ImageFolder(
            args.dataroot,
            transform=transforms.Compose(
                [
                    transforms.Resize((args.imsize, args.imsize)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            ),
        )
    else:
        train_dataset = torchvision.datasets.ImageFolder(
            args.dataroot,
            transform=transforms.Compose(
                [
                    transforms.Resize((args.imsize, args.imsize)),
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            ),
        )
    train_dataset = torch.utils.data.Subset(
        train_dataset, torch.randperm(len(train_dataset)).tolist()
    )
    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16
    )

    # define optimizer
    optimizer = torch.optim.Adam(lift.parameters(), lr=args.lr)

    # define reconstruction loss
    if args.loss == "l1":
        criterion = torch.nn.L1Loss()
    elif args.loss == "l2":
        criterion = torch.nn.MSELoss()
    else:
        criterion = torch.nn.CosineEmbeddingLoss()

    # if model exists in dir load the latest epoch model
    epoch_start = 0
    if len(glob(os.path.join(output_dir, "*.pth"))) > 0:
        model_paths = glob(os.path.join(output_dir, "*.pth"))
        model_paths = natsort.natsorted(model_paths)
        lift.load_state_dict(torch.load(model_paths[-1]))
        epoch_start = int(model_paths[-1].split("_")[-1].split(".")[0])
        print("Loaded model from epoch {}".format(epoch_start))

    # train
    for epoch in range(epoch_start, args.epochs):
        epoch_start_time = time.time()
        for batch_index, image_batch in enumerate(dataloader):
            batch_start_time = time.time()
            if "imagenet" in args.dataroot:
                image_batch = image_batch[0].to(device)
            else:
                image_batch = image_batch.to(device)

            # multi-scale backbone features
            with torch.no_grad():
                if args.my_extractor:
                    feat1 = extractor(image_batch)
                    feat2 = extractor(image_batch[:, :, ::2, ::2])
                    feat3 = extractor(image_batch[:, :, ::4, ::4])
                else:
                    feat1 = extractor.extract_descriptors(
                        image_batch, args.layer, args.facet
                    )[:, 0, :, :]
                    feat2 = extractor.extract_descriptors(
                        image_batch[:, :, ::2, ::2], args.layer, args.facet
                    )[:, 0, :, :]
                    feat3 = extractor.extract_descriptors(
                        image_batch[:, :, ::4, ::4], args.layer, args.facet
                    )[:, 0, :, :]
                feat1 = convert_shape(feat1, L1, L1)
                feat2 = convert_shape(feat2, L2, L2)
                feat3 = convert_shape(feat3, L4, L4)

            gen_1 = lift(image_batch[:, :, ::2, ::2], feat2)
            gen_2 = lift(image_batch[:, :, ::4, ::4], feat3)

            if args.loss == "cosine":
                gen_1 = gen_1.permute(0, 2, 3, 1)
                gen_1 = gen_1.reshape(-1, gen_1.shape[-1])
                gen_2 = gen_2.permute(0, 2, 3, 1)
                gen_2 = gen_2.reshape(-1, gen_2.shape[-1])
                feat1 = feat1.permute(0, 2, 3, 1)
                feat1 = feat1.reshape(-1, feat1.shape[-1])
                feat2 = feat2.permute(0, 2, 3, 1)
                feat2 = feat2.reshape(-1, feat2.shape[-1])
                loss = criterion(
                    gen_1, feat1, torch.ones(gen_1.shape[0]).to(device)
                ) + criterion(gen_2, feat2, torch.ones(gen_2.shape[0]).to(device))
            else:
                loss = criterion(gen_1, feat1) + criterion(gen_2, feat2)

            print(
                "Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}".format(
                    epoch + 1,
                    args.epochs,
                    batch_index + 1,
                    len(dataloader),
                    loss.item(),
                ),
                flush=True,
            )
            print(
                "Time taken for 1 batch: {} seconds".format(
                    time.time() - batch_start_time
                )
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(
            "Time taken for 1 epoch: {} seconds".format(time.time() - epoch_start_time)
        )
        if (epoch + 1) % args.save_every == 0:
            torch.save(
                lift.state_dict(),
                os.path.join(output_dir, "lift_{}.pth".format(epoch + 1)),
            )

    # save final model
    torch.save(lift.state_dict(), os.path.join(output_dir, "lift.pth"))


def parse_args():
    parser = argparse.ArgumentParser("Train a LiFT model")
    ### BACKBONE ###
    parser.add_argument(
        "--model_type",
        default="dino_vits8",
        type=str,
        help="""type of model to extract. 
                        Choose from [dino_vits8 | dino_vits16 | dino_vitb8 | dino_vitb16 | vit_small_patch8_224 | 
                        vit_small_patch16_224 | vit_base_patch8_224 | vit_base_patch16_224 ]""",
    )
    parser.add_argument(
        "--facet",
        default="key",
        type=str,
        help="""facet to create descriptors from. 
                        options: ['key' | 'query' | 'value' | 'token']""",
    )
    parser.add_argument(
        "--channel",
        default=None,
        type=int,
        help="backbone output channels (default: inferred from --model_type)",
    )
    parser.add_argument(
        "--patch",
        default=None,
        type=int,
        help="backbone patch size (default: inferred from --model_type)",
    )
    parser.add_argument(
        "--stride",
        default=None,
        type=int,
        help="stride of first convolution layer. small stride -> higher resolution. (default: equal to --patch)",
    )
    parser.add_argument(
        "--layer",
        default=None,
        type=int,
        help="layer to create descriptors from. (default: last layer)",
    )
    ### LiFT TRAINING ###
    parser.add_argument(
        "--imsize", default=448, type=int, help="largest image size during training"
    )
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
    parser.add_argument("--epochs", default=100, type=int, help="number of epochs")
    parser.add_argument("--batch_size", default=64, type=int, help="batch size")
    parser.add_argument("--save_every", default=1, type=int, help="save_every n epochs")
    parser.add_argument("--augment", action="store_true", help="augment data or not")
    parser.add_argument(
        "--loss", default="cosine", type=str, help="choose from l1, l2, cosine"
    )
    parser.add_argument(
        "--output_dir",
        default="lift_imagenet_trains",
        help="dir to save checkpoints to",
    )
    ### DATASET ###
    parser.add_argument("--dataroot", default="/path/to/imagenet/train")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    ### OTHER ###
    parser.add_argument(
        "--my_extractor",
        action="store_true",
        help="Enable loading of a custom feature extractor instead of a built in one. Must be implemented in my_extractor.",
    )
    ###
    args = parser.parse_args()
    infer_settings(args)
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
