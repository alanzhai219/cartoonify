import time
import os

import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.autograd import Variable

from network.Transformer import Transformer

import argparse

def transform(model, input, load_size=450, gpu=-1):
    if gpu > -1:
        model.cuda()
    else:
        model.float()

    input_image = Image.open(input).convert("RGB")
    h, w = input_image.size

    ratio = h * 1.0 / w

    if ratio > 1:
        h = load_size
        w = int(h * 1.0 / ratio)
    else:
        w = load_size
        h = int(w * ratio)

    input_image = input_image.resize((h, w), Image.BICUBIC)
    input_image = np.asarray(input_image)

    input_image = input_image[:, :, [2, 1, 0]]
    input_image = transforms.ToTensor()(input_image).unsqueeze(0)

    input_image = -1 + 2 * input_image
    if gpu > -1:
        input_image = Variable(input_image).cuda()
    else:
        input_image = Variable(input_image).float()

    t0 = time.time()
    print("input shape", input_image.shape)
    with torch.no_grad():
        output_image = model(input_image)[0]
    print(f"inference time took {time.time() - t0} s")

    output_image = output_image[[2, 1, 0], :, :]
    output_image = output_image.data.cpu().float() * 0.5 + 0.5

    output_image = output_image.numpy()

    output_image = np.uint8(output_image.transpose(1, 2, 0) * 255)
    output_image = Image.fromarray(output_image)

    return output_image

def read_image(image):
    return image

def load_model(style):
    pth_path = os.path.join("pretrained_models", style+"_net_G_float.pth")
    model = Transformer()
    model.load_state_dict(torch.load(pth_path))
    return model

def dump_output(out, out_dir):
    out.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="need some args ...")
    parser.add_argument("image", type=str, help="pass the path of image")
    parser.add_argument("style", type=str, help="select a model from [Hosoda, Hayao, Shinkai, Paprika]", default="Hosoda")
    parser.add_argument("out", type=str, help="specify a directory for output image", default="out")
    args = parser.parse_args()
    # step1: read image
    image = read_image(args.image)

    # step2: load model
    model = load_model(args.style)

    # step3: cartoonify
    out = transform(model, image, 1200)

    # step4: dump output
    dump_output(out, args.out)
