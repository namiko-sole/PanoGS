import torch
from diffusers import ControlNetModel, StableDiffusionXLControlNetInpaintPipeline, StableDiffusionControlNetInpaintPipeline, \
    StableDiffusionXLControlNetPipeline, StableDiffusionXLControlNetImg2ImgPipeline, DiffusionPipeline, \
    AutoencoderKL
# from diffusion_utils import StableDiffusionXLControlNetLoopConsistPipeline
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusion_utils_inpaint import StableDiffusionXLControlNetInpaintLoopConsistentPipeline
from diffusion_utils import StableDiffusionXLControlNetLoopConsistPipeline
from diffusers_utils.sdxl_i2i_control_loop import StableDiffusionXLControlNetImg2ImgLoopConsistentPipeline

import cv2
from PIL import Image, ImageChops

from ip_adapter import IPAdapterXL, IPAdapterPlus, IPAdapter
import numpy as np
import time
import os
import math
from diffusers.utils import load_image
# from hidiffusion import apply_hidiffusion, remove_hidiffusion
import torchvision

from internal.utils.adain_utils.adain_api import generate_adain

device = "cuda"

model_pipe = None
model_ip_model = None
inpaint_pipe = None
inpaint_ip_model = None
refine_pipe = None
refine_ip_model = None

# All diffusion checkpoints are loaded from the local ``checkpoints/`` folder
# CHECKPOINTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
CHECKPOINTS_DIR = "/nas1/nas1/data/hyh22/HuggingFaceModels"

base_model_path = os.path.join(CHECKPOINTS_DIR, "stabilityai", "stable-diffusion-xl-base-1.0")
image_encoder_path = os.path.join(CHECKPOINTS_DIR, "h94", "IP-Adapter", "sdxl_models", "image_encoder")
ip_ckpt = os.path.join(CHECKPOINTS_DIR, "h94", "IP-Adapter", "sdxl_models", "ip-adapter_sdxl.bin")
controlnet_canny_path = os.path.join(CHECKPOINTS_DIR, "diffusers", "controlnet-canny-sdxl-1.0")


def init_inpaint_model():
    # free_model()
    # init_model()
    global inpaint_pipe
    global inpaint_ip_model

    inpaint_model_path = base_model_path
    inpaint_encoder_path = image_encoder_path
    inpaint_ip_ckpt = ip_ckpt
    inpaint_controlnet_canny_path = controlnet_canny_path

    if inpaint_ip_model is not None: return
    # vae = AutoencoderKL.from_pretrained("/raid0/hyh22/HuggingFaceModels/madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

    controlnet_canny = ControlNetModel.from_pretrained(inpaint_controlnet_canny_path, torch_dtype=torch.float16).to(device)
    controlnets = MultiControlNetModel([controlnet_canny])

    # load SDXL pipeline
    # inpaint_pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    inpaint_pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
    # inpaint_pipe = StableDiffusionXLControlNetInpaintLoopConsistentPipeline.from_pretrained(
        inpaint_model_path,
        controlnet=controlnets,
        # controlnet=controlnet_canny,
        # vae=vae,
        # variant="fp16",
        torch_dtype=torch.float16,
        # add_watermarker=False,
    )
    inpaint_pipe.enable_model_cpu_offload()
    # inpaint_pipe.enable_vae_tiling()

    # target_blocks=["block"] for original IP-Adapter
    # target_blocks=["up_blocks.0.attentions.1"] for style blocks only
    # target_blocks = ["up_blocks.0.attentions.1", "down_blocks.2.attentions.1"] # for style+layout blocks
    # inpaint_ip_model = IPAdapterXL(inpaint_pipe, inpaint_encoder_path, inpaint_ip_ckpt, device, target_blocks=["block"])

    global refine_pipe
    global refine_ip_model
    if refine_ip_model is not None: return
    controlnet_canny = ControlNetModel.from_pretrained(controlnet_canny_path, use_safetensors=True, torch_dtype=torch.float16).to(device)
    controlnets = MultiControlNetModel([controlnet_canny])

    # load SDXL pipeline
    # refine_pipe = StableDiffusionXLControlNetLoopConsistPipeline.from_pretrained(
    refine_pipe = StableDiffusionXLControlNetImg2ImgLoopConsistentPipeline.from_pretrained(
        base_model_path,
        controlnet=controlnets,
        torch_dtype=torch.float16,
        add_watermarker=False,
    )
    refine_pipe.enable_vae_tiling()
    # apply_hidiffusion(model_pipe)

    refine_ip_model = IPAdapterXL(refine_pipe, image_encoder_path, ip_ckpt, device, target_blocks=["up_blocks.0.attentions.1"])

def init_model():
    # free_inpaint_model()
    global model_pipe
    global model_ip_model

    if model_ip_model is not None: return
    controlnet_canny = ControlNetModel.from_pretrained(controlnet_canny_path, use_safetensors=True, torch_dtype=torch.float16).to(device)
    controlnets = MultiControlNetModel([controlnet_canny])

    # load SDXL pipeline
    # model_pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    # model_pipe = StableDiffusionXLControlNetLoopConsistPipeline.from_pretrained(
    model_pipe = StableDiffusionXLControlNetImg2ImgLoopConsistentPipeline.from_pretrained(
    # model_pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        base_model_path,
        controlnet=controlnets,
        torch_dtype=torch.float16,
        add_watermarker=False,
    )
    model_pipe.enable_vae_tiling()
    # apply_hidiffusion(model_pipe)

    model_ip_model = IPAdapterXL(model_pipe, image_encoder_path, ip_ckpt, device, target_blocks=["up_blocks.0.attentions.1"])

def free_model():
    global model_pipe
    global model_ip_model
    del model_ip_model
    del model_pipe
    model_ip_model = None
    model_pipe = None

def free_inpaint_model():
    global inpaint_pipe
    global inpaint_ip_model
    global refine_pipe
    global refine_ip_model
    del inpaint_ip_model
    del inpaint_pipe
    del refine_pipe
    del refine_ip_model
    inpaint_ip_model = None
    inpaint_pipe = None
    refine_pipe = None
    refine_ip_model = None

def normalize(image):
    image = image / 127.5 - 1
    image = torch.tensor(image).unsqueeze(0).permute(0, 3, 1, 2)
    return image

def generate_image(prompt, style_img_path, input_img_path, ref_img_path, mask_img_path=None, pano_res=None, debug=False, **kwargs):
    style_img = Image.open(style_img_path)
    # style_img.resize((512, 512))
    
    input_img = load_image(input_img_path)
    ref_img = load_image(ref_img_path)
    detected_map = cv2.Canny(np.array(ref_img), 50, 200)
    if pano_res is not None:
        target_size = (pano_res * 2, pano_res)
        if (input_img.width, input_img.height) != target_size:
            input_img = input_img.resize(target_size, Image.LANCZOS)
        if (ref_img.width, ref_img.height) != target_size:
            ref_img = ref_img.resize(target_size, Image.LANCZOS)
    # canny_map = Image.fromarray(cv2.cvtColor(detected_map, cv2.COLOR_BGR2RGB))
    canny_map = Image.fromarray(detected_map).resize((input_img.width, input_img.height))

    canny_map.save(os.path.join(os.path.dirname(ref_img_path), "pano_canny.png"))

    # --- debug: save all images fed into the model ---
    if debug:
        _debug_dir = os.path.join(os.path.dirname(__file__), "debug")
        os.makedirs(_debug_dir, exist_ok=True)
        style_img.save(os.path.join(_debug_dir, "debug_style_img.png"))
        input_img.save(os.path.join(_debug_dir, "debug_input_img.png"))
        ref_img.save(os.path.join(_debug_dir, "debug_ref_img.png"))
        canny_map.save(os.path.join(_debug_dir, "debug_canny_map.png"))
        if mask_img_path is not None:
            input_img_mask_dbg = load_image(mask_img_path).convert('L')
            input_img_mask_dbg.save(
                os.path.join(_debug_dir, "debug_mask_raw.png")
            )
        print(f"[DEBUG] saved model-input images to {_debug_dir}")
    # --------------------------------------------------

    c_prompt = prompt + ", masterpiece, best quality, high quality"
    negative_prompt = "lowres, low quality, worst quality, deformed, glitch, noisy, saturation, blurry"
    # input_img = input_img.convert('L')
    with torch.no_grad():
        if mask_img_path is None:
            init_model()
            pipe = model_ip_model

            # input_img_latent = model_pipe.vae.encode(normalize(np.array(input_img)).to(device, dtype=torch.float16)).latent_dist.mode()
            # input_img_latent = model_pipe.vae.config.scaling_factor * input_img_latent

            ### Base Model ###
            # images = pipe.generate(pil_image=style_img,
            #                         prompt=prompt,
            #                         negative_prompt=negative_prompt,
            #                         scale=1,
            #                         guidance_scale=5,
            #                         num_samples=1,
            #                         seed=42,
            #                         num_inference_steps=42,
            #                         image=[canny_map],
            #                         controlnet_conditioning_scale=[0.8],
            #                         **kwargs,
            #                     )

            ### Img2Img Model ###
            images = pipe.generate(pil_image=style_img,
                                    prompt=c_prompt,
                                    negative_prompt=negative_prompt,
                                    scale=1,
                                    guidance_scale=5,
                                    num_samples=1,
                                    seed=42,
                                    num_inference_steps=math.ceil(42/kwargs['strength']),
                                    # strength=1.0,
                                    image=input_img,
                                    # latents=input_img_latent,
                                    control_image=[canny_map],
                                    controlnet_conditioning_scale=[1.0],
                                    **kwargs,
                                )
            free_model()
        else:
            init_inpaint_model()
            input_img_mask = load_image(mask_img_path).convert('L')
            input_img_mask = Image.fromarray(~np.array(input_img_mask))
            if pano_res is not None:
                input_img_mask = input_img_mask.resize(
                    (input_img.width, input_img.height), Image.NEAREST
                )
            if debug:
                input_img_mask.save(os.path.join(_debug_dir, "debug_input_img_mask.png"))
            # pipe = inpaint_ip_model
            # images = pipe.generate(pil_image=style_img,
            #                         prompt=c_prompt,
            #                         negative_prompt=negative_prompt,
            #                         scale=1.0,
            #                         # guidance_scale=5,
            #                         num_samples=1,
            #                         seed=42,
            #                         num_inference_steps=43,
            #                         # strength=0.99,
            #                         image=input_img,
            #                         mask_image=input_img_mask,
            #                         control_image=[canny_map],
            #                         controlnet_conditioning_scale=[1.0],
            #                         # control_guidance_end=0.95,
            #                         **kwargs,
            #                     )
            image = inpaint_pipe(
                            prompt=c_prompt,
                            # negative_prompt=negative_prompt,
                            num_inference_steps=20,
                            # generator=generator,
                            eta=1.0,
                            seed=42,
                            # strength=1.0,
                            # guess_mode=True,
                            image=input_img,
                            mask_image=input_img_mask,
                            control_image=[canny_map],
                            controlnet_conditioning_scale=[1.0],
                            # output_type="latent",
                        ).images[0]

            if debug:
                image.save(os.path.join(_debug_dir, "debug_inpaint_mid.png"))

            pipe = refine_ip_model
            images = pipe.generate(pil_image=style_img,
                                    prompt=c_prompt,
                                    negative_prompt=negative_prompt,
                                    scale=1,
                                    guidance_scale=5,
                                    num_samples=1,
                                    seed=42,
                                    num_inference_steps=math.ceil(42/kwargs['strength']),
                                    # strength=1.0,
                                    image=image,
                                    control_image=[canny_map],
                                    controlnet_conditioning_scale=[1.0],
                                    **kwargs,
                                )
            free_inpaint_model()
            return images[0], image
            # return image
    return images[0]

if __name__ == "__main__":
    pass