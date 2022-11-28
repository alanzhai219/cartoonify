import time
import os

import numpy as np
from PIL import Image

import argparse

def run_ov_infer(model, input_image, gpu):
    import openvino.runtime as ov
    # 1. Create OpenVINO Runtime Core
    core = ov.Core()

    # 2. Compile Model
    net = core.compile_model(model, "CPU")

    # 3. Create the infer request
    infer_request = net.create_infer_request()

    # 4. copy the input
    input_tensor = ov.Tensor(input_image)
    infer_request.set_input_tensor(input_tensor)

    # 5. start inference
    t0 = time.time()
    print("input shape", input_image.shape)
    infer_request.start_async()
    infer_request.wait()
    print(f"inference time took {time.time() - t0} s")

    # 6. get output
    output = infer_request.get_output_tensor()
    output_tensor = output.data
    output_tensor = np.squeeze(output_tensor, axis=0)
    return output_tensor

def pre_process(args):
    input_image = Image.open(args.image).convert("RGB")
    h, w = input_image.size
    
    ratio = h * 1.0 / w
    
    if ratio > 1:
        h = args.load_size
        w = int(h * 1.0 / ratio)
    else:
        w = args.load_size
        h = int(w * ratio)
        
    input_image = input_image.resize((h, w), Image.BICUBIC)
    input_image = np.asarray(input_image)
    input_image = input_image[:, :, [2, 1, 0]]
    # implement the transform.ToTensor() operation
    input_image = input_image.transpose(2,0,1)
    input_image = input_image.astype(np.float32)
    input_image = input_image / 255
    input_image = -1 + 2 * input_image
    input_image = np.expand_dims(input_image, axis=0)
    return input_image

def post_process(output_image):
    output_image = output_image[[2, 1, 0], :, :]
    output_image = output_image * 0.5 + 0.5
    output_image = np.uint8(output_image.transpose(1, 2, 0) * 255)
    output_image = Image.fromarray(output_image)
    return output_image

def run_torch_infer(model, input_image, gpu):
    import torch
    import torchvision.transforms as transforms

    input_image = torch.tensor(input_image)

    if gpu > -1:
        input_image = input_image.cuda()
    else:
        input_image = input_image.float()

    t0 = time.time()
    print("input shape", input_image.shape)
    with torch.no_grad():
        output_image = model(input_image)[0]
    print(f"inference time took {time.time() - t0} s")
    output_image = output_image.numpy()
    return output_image

def transform(model, args, gpu=-1):
    input_image = pre_process(args)
    output_image = run_torch_infer(model, input_image, gpu)
    # output_image = run_ov_infer(model, input_image, gpu)
    output_image = post_process(output_image)

    return output_image

def read_image(image):
    return image

def load_model(args):
    import torch
    import sys
    sys.path.append("..")
    from network.Transformer import Transformer
    pth_path = os.path.join(args.pretrained, args.style+"_net_G_float.pth")
    model = Transformer()
    model.load_state_dict(torch.load(pth_path))
    return model

def dump_output(out, args):
    img_name = os.path.join(args.out, os.path.basename(args.image))
    out.save(img_name)
    out.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="need some args ...")
    parser.add_argument("--image", type=str, help="pass the path of image")
    parser.add_argument("--style", type=str, help="select a model from [Hosoda, Hayao, Shinkai, Paprika]", default="Hosoda")
    parser.add_argument("--load_size", type=int, help="the load size of input", default=1200)
    parser.add_argument("--pretrained", type=str, required=True, help="select a pretrained directory")
    parser.add_argument("--out", type=str, help="specify a directory for output image", default="out")
    args = parser.parse_args()
    # step1: read image
    image = read_image(args.image)

    # step2: load model
    model = load_model(args)
    # model = f"out/onnx/{args.style}.onnx"

    # step3: cartoonify
    out = transform(model, args)

    # step4: dump output
    dump_output(out, args)