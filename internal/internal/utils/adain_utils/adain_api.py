import os
import sys
sys.path.append(os.path.dirname(__file__))

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

import net
from function import adaptive_instance_normalization, coral

decoder_model = 'internal/utils/adain_utils/models/decoder.pth'
vgg_model = 'internal/utils/adain_utils/models/vgg_normalised.pth'
content_size = 1024
style_size = 512
style_weight = 1

def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def style_transfer(vgg, decoder, content, style, alpha=1.0,
                   interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = adaptive_instance_normalization(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        content_f = content_f[0:1]
    else:
        feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)

device = torch.device("cuda")

def generate_adain(content_path, style_path, save_path):
    decoder = net.decoder
    vgg = net.vgg

    decoder.eval()
    vgg.eval()

    decoder.load_state_dict(torch.load(decoder_model))
    vgg.load_state_dict(torch.load(vgg_model))
    vgg = nn.Sequential(*list(vgg.children())[:31])

    vgg.to(device)
    decoder.to(device)

    content_tf = test_transform(content_size, False)
    style_tf = test_transform(style_size, False)

    content = content_tf(Image.open(content_path))
    style = style_tf(Image.open(style_path))
    style = style.to(device).unsqueeze(0)
    content = content.to(device).unsqueeze(0)
    with torch.no_grad():
        output = style_transfer(vgg, decoder, content, style, style_weight)
    output = output.cpu()
    if save_path is not None:
        save_image(output, save_path)
        return Image.open(save_path)
    return output

if __name__ == "__main__":
    image = generate_adain(content_path="preprocess/playroom_test/cam_-1/pano_img.png",
                   style_path="style_data/indoor/room3.jpg",
                   save_path="temp.png")