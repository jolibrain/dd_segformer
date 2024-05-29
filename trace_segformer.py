#!/usr/bin/python3

import sys
import os
import argparse
import logging

import torch

from models.modules.segformer.segformer_generator import Segformer

def main():
    parser = argparse.ArgumentParser(description="Trace Segformer model for segmentation task")
    parser.add_argument("--config", required=True, type=str, help="Segformer configuration")
    parser.add_argument("--nclasses", type=int, default=2, help="Number of classes")
    parser.add_argument("--image_size", type=int, default=512, help="Size of the input image")
    parser.add_argument("--weights", type=str, help="Segformer pretrained weights")
    parser.add_argument("--trace", action="store_true", help="If True, trace model instead of scripting")
    parser.add_argument("-o", "--output_dir", type=str, help="Output directory")
    parser.add_argument('-v', "--verbose", action='store_true', help="Set logging level to INFO")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)
  
    net = Segformer(
        os.path.dirname(__file__),
        args.config,
        3, # number of channels
        img_size=args.image_size,
        num_classes=args.nclasses,
        final_conv=False,
    )
    if args.weights:
        load_segformer_weights(net, args.weights)

    net.to("cuda:0")
    if args.trace:
        net.eval()
        script_model = torch.jit.trace(net, torch.rand(1, 3, args.image_size, args.image_size, device="cuda:0"))
    else:
        script_model = torch.jit.script(net)

    for name, val in script_model.state_dict().items():
        print(name, val.shape)
    # print(script_model(torch.rand(1, 3, args.image_size, args.image_size, device="cuda:0")).shape)
    script_model.save("segformer_b0_cls%d.pt" % args.nclasses)

# ====

def load_segformer_weights(net, weights_path):
    weights = torch.load(weights_path)

    try:
        net.backbone.net.load_state_dict(weights, strict=False)
    except:
        print(
            "f_s pretrained segformer decode_head size may have the wrong number of classes, fixing"
        )
        pretrained_dict = {k: v for k, v in weights.items() if k in weights}
        decode_head_keys = []
        for k in pretrained_dict.keys():
            if "decode_head" in k:
                decode_head_keys.append(k)
        for k in decode_head_keys:
            del weights[k]

        net.backbone.net.load_state_dict(weights, strict=False)

if __name__ == "__main__":
    main()
