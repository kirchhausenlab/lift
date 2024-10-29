"""
Feature Extractor for a pre-trained ViT+LiFT combination. Built as an extension to https://github.com/ShirAmir/dino-vit-features/blob/main/extractor.py

Please ensure that the LiFT module loaded is correct for the selected model, layer, and facet. An incorrect backbone feature + lift combination will
yield poor features, as LiFT is designed to operate on the particular feature distribution it was trained on.

Code by: Saksham Suri and Matthew Walmer

Sample Usage:
python lift_extractor.py --image_path assets/sample.jpg --output_path sample.pth --model_type dino_vits16 --lift_path pretrained/lift_dino_vits16.pth
"""
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from extractor import ViTExtractor
from lift import LiFT
from lift_utils import infer_settings


class ViTLiFTExtractor(nn.Module):
    def __init__(self, model_type: str = 'dino_vits8', lift_path: str = None, channel: int = 768, patch: int = 8, stride: int = 8,
            layer: int = 11, facet: str = 'key', model: nn.Module = None, device: str = 'cuda', silent=False):
        super(ViTLiFTExtractor, self).__init__()
        self.model_type = model_type
        self.model = model
        self.lift_path = lift_path
        self.channel = channel
        self.patch = patch
        self.stride = stride
        self.layer = layer
        self.facet = facet
        self.device = device
        # prep extractor
        self.extractor = ViTExtractor(model_type, stride, model, device)
        if not silent: print('Loaded Backbone: ' + model_type)
        # prep lift
        if lift_path is None:
            self.lift = None
            if not silent: print('No LiFT path provided, running backbone only')
        else:
            self.lift = LiFT(self.channel, self.patch)
            state_dict = torch.load(lift_path)
            # if "module." in state_dict, remove it
            for k in list(state_dict.keys()):
                if k.startswith('module.'):
                    state_dict[k[7:]] = state_dict[k]
                    del state_dict[k]
            self.lift.load_state_dict(state_dict)
            self.lift.to(device)
            if not silent: print('Loaded LiFT module from: ' + lift_path)


    def preprocess(self, image_path, load_size):
        return self.extractor.preprocess(image_path, load_size)


    def extract_descriptors(self, batch):
        fs = self.extractor.extract_descriptors(batch, self.layer, self.facet)[:,0,:,:]
        if self.lift is not None:
            fs = self.lift(batch, fs)
        return fs


    def extract_descriptors_iterative_lift(self, batch, lift_iter=1, return_inter=False):
        ret = {}
        fs = self.extractor.extract_descriptors(batch, self.layer, self.facet)[:,0,:,:]
        ret['back'] = fs
        if self.lift is not None:
            for i in range(lift_iter):
                fs = self.lift(batch, fs)
                ret['lift_%i'%(i+1)] = fs
                if i+1 < lift_iter:
                    batch = F.interpolate(batch, size=(batch.shape[-2]*2, batch.shape[-1]*2), mode='bilinear', align_corners=False)
        if return_inter: return ret
        return fs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Facilitate ViT+LIFT Descriptor extraction.')
    ### BACKBONE ###
    parser.add_argument('--model_type', default='dino_vits8', type=str,
                        help="""type of model to extract. 
                        Choose from [dino_vits8 | dino_vits16 | dino_vitb8 | dino_vitb16 | vit_small_patch8_224 | 
                        vit_small_patch16_224 | vit_base_patch8_224 | vit_base_patch16_224 ]""")
    parser.add_argument('--facet', default='key', type=str, help="""facet to create descriptors from. 
                        options: ['key' | 'query' | 'value' | 'token']""")
    parser.add_argument('--channel', default=None, type=int, help='backbone output channels (default: inferred from --model_type)')
    parser.add_argument('--patch', default=None, type=int, help='backbone patch size (default: inferred from --model_type)')
    parser.add_argument('--stride', default=None, type=int, help='stride of first convolution layer. small stride -> higher resolution. (default: equal to --patch)')
    parser.add_argument('--layer', default=None, type=int, help='layer to create descriptors from. (default: last layer)')
    ### LIFT ###
    parser.add_argument('--lift_path', default=None, type=str, help='path of pretrained LiFT model to use. If not given, lift LiFT is not used')
    parser.add_argument('--lift_iter', default=1, type=int, help='set to >1 to apply LiFT iteratively')
    ### INPUTS / OUTPUTS ###
    parser.add_argument('--image_path', type=str, required=True, help='path of the extracted image.')
    parser.add_argument('--output_path', type=str, required=True, help='path to file containing extracted descriptors.')
    parser.add_argument('--load_size', default=224, type=int, help='load size of the input image.')
    args = parser.parse_args()
    infer_settings(args)

    with torch.no_grad():
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        extractor = ViTLiFTExtractor(args.model_type, args.lift_path, args.channel, args.patch, args.stride, args.layer, args.facet, device=device)
        extractor.eval()
        image_batch, image_pil = extractor.preprocess(args.image_path, args.load_size)
        print(f"Image {args.image_path} is preprocessed to tensor of size {image_batch.shape}.")
        if args.lift_iter > 1:
            descriptors = extractor.extract_descriptors_iterative_lift(image_batch.to(device), args.lift_iter)
        else:
            descriptors = extractor.extract_descriptors(image_batch.to(device))
        print(f"Descriptors are of size: {descriptors.shape}")
        torch.save(descriptors, args.output_path)
        print(f"Descriptors saved to: {args.output_path}")