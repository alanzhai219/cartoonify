import time
import os

import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.autograd import Variable

import sys
sys.path.append("..")
from network.Transformer import Transformer
import argparse

def load_model(args):
    pth_path = os.path.join(args.pretrained, args.style+"_net_G_float.pth")
    model = Transformer()
    model.load_state_dict(torch.load(pth_path))
    return model

def export_onnx(args):
    save_model_name = os.path.join(args.out, args.format, args.style+".onnx")
    model = load_model(args)
    model = model.eval()
    x = torch.randn(1,3,224,224, requires_grad=False)
    torch.onnx.export(model,
                      x,
                      save_model_name,
                      export_params=True,
                      opset_version=10,
                      do_constant_folding=True,
                      input_names=["input"],
                      output_names=["output"],
                      dynamic_axes={"input":{0:"batch_size", 2:"width", 3:"height"}})

def export_torchscript(args):
    save_model_name = os.path.join(args.out, args.format, args.style+".pt")
    model = load_model(args)
    model = model.eval()
    x = torch.randn(1,3,224,224, requires_grad=False)
    m = torch.jit.script(model, x)
    torch.jit.save(m, save_model_name)


def export_torchtrace(args):
    save_model_name = os.path.join(args.out, args.format, args.style+".pt")
    model = load_model(args)
    model = model.eval()
    x = torch.randn(1,3,224,224, requires_grad=False)
    m = torch.jit.trace(model, x)
    torch.jit.save(m, save_model_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="export model ...")
    parser.add_argument("--format", type=str, help="pass the exported format. onnx or script or trace", default="onnx")
    parser.add_argument("--style", type=str, help="select a model from [Hosoda, Hayao, Shinkai, Paprika]", default="Hosoda")
    parser.add_argument("--pretrained", type=str, required=True, help="select a pretrained directory")
    parser.add_argument("--out", type=str, help="specify a directory for output image", default="out")
    args = parser.parse_args()

    # check output
    save_dir_name = os.path.join(args.out, args.format)
    if not os.path.exists(save_dir_name):
        os.makedirs(save_dir_name)

    if (args.format == "onnx"):
        export_onnx(args)
    elif (args.format == "script"):
        export_torchscript(args)
    elif (args.format == "trace"):
        export_torchtrace(args)
    else:
        assert(f"{args.format} is NOT supported!")