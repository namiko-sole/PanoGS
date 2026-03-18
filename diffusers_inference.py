import torch
from diffusers import ControlNetModel, StableDiffusionXLControlNetInpaintPipeline, StableDiffusionControlNetInpaintPipeline, \
    StableDiffusionXLControlNetPipeline, StableDiffusionXLControlNetImg2ImgPipeline, DiffusionPipeline, \
    AutoencoderKL
# from diffusion_utils import StableDiffusionXLControlNetLoopConsistPipeline
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusion_utils_inpaint import StableDiffusionXLControlNetInpaintLoopConsistentPipeline
from diffusion_utils import StableDiffusionXLControlNetLoopConsistPipeline
from diffusers_utils.sdxl_i2i_control_loop import StableDiffusionXLControlNetImg2ImgLoopConsistentPipeline

from transformers import pipeline
from transformers import DPTFeatureExtractor, DPTForDepthEstimation

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
depth_pipe = None

base_model_path = "/nas1/hyh22/HuggingFaceModels/stabilityai/stable-diffusion-xl-base-1.0"
image_encoder_path = "/nas1/hyh22/HuggingFaceModels/h94/IP-Adapter/sdxl_models/image_encoder"
ip_ckpt = "/nas1/hyh22/HuggingFaceModels/h94/IP-Adapter/sdxl_models/ip-adapter_sdxl.bin"
controlnet_canny_path = "/nas1/hyh22/HuggingFaceModels/diffusers/controlnet-canny-sdxl-1.0"
controlnet_depth_path = "/nas1/hyh22/HuggingFaceModels/diffusers/controlnet-depth-sdxl-1.0"


def init_inpaint_model():
    # free_model()
    # init_model()
    global inpaint_pipe
    global inpaint_ip_model

    inpaint_model_path = "/nas1/hyh22/HuggingFaceModels/stabilityai/stable-diffusion-xl-base-1.0"
    inpaint_encoder_path = "/nas1/hyh22/HuggingFaceModels/h94/IP-Adapter/sdxl_models/image_encoder"
    inpaint_ip_ckpt = "/nas1/hyh22/HuggingFaceModels/h94/IP-Adapter/sdxl_models/ip-adapter_sdxl.bin"
    inpaint_controlnet_canny_path = "/nas1/hyh22/HuggingFaceModels/diffusers/controlnet-canny-sdxl-1.0"
    inpaint_controlnet_depth_path = "/nas1/hyh22/HuggingFaceModels/diffusers/controlnet-depth-sdxl-1.0"

    # inpaint_model_path = "/raid0/hyh22/HuggingFaceModels/runwayml/stable-diffusion-inpainting/"
    # inpaint_encoder_path = "/raid0/hyh22/HuggingFaceModels/h94/IP-Adapter/models/image_encoder"
    # inpaint_ip_ckpt = "/raid0/hyh22/HuggingFaceModels/h94/IP-Adapter/models/ip-adapter_sd15.bin"
    # inpaint_controlnet_canny_path = "/raid0/hyh22/HuggingFaceModels/lllyasviel/control_v11p_sd15_canny/"
    # inpaint_controlnet_depth_path = "/raid0/hyh22/HuggingFaceModels/lllyasviel/control_v11f1p_sd15_depth/"
    
    if inpaint_ip_model is not None: return
    # vae = AutoencoderKL.from_pretrained("/raid0/hyh22/HuggingFaceModels/madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

    controlnet_canny = ControlNetModel.from_pretrained(inpaint_controlnet_canny_path, torch_dtype=torch.float16).to(device)
    controlnet_depth = ControlNetModel.from_pretrained(inpaint_controlnet_depth_path, torch_dtype=torch.float16).to(device)
    controlnets = MultiControlNetModel([controlnet_depth, controlnet_canny])

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
    # inpaint_ip_model = IPAdapter(inpaint_pipe, image_encoder_path, ip_ckpt, device, target_blocks=["block"])

    # depth_pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")

    global refine_pipe
    global refine_ip_model
    if refine_ip_model is not None: return
    controlnet_canny = ControlNetModel.from_pretrained(controlnet_canny_path, use_safetensors=True, torch_dtype=torch.float16).to(device)
    controlnet_depth = ControlNetModel.from_pretrained(controlnet_depth_path, use_safetensors=True, torch_dtype=torch.float16).to(device)
    controlnets = MultiControlNetModel([controlnet_depth, controlnet_canny])

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
    controlnet_depth = ControlNetModel.from_pretrained(controlnet_depth_path, use_safetensors=True, torch_dtype=torch.float16).to(device)
    controlnets = MultiControlNetModel([controlnet_depth, controlnet_canny])

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

def get_depth_map(depth_image):
    depth_estimator = DPTForDepthEstimation.from_pretrained("/nas1/hyh22/HuggingFaceModels/Intel/dpt-hybrid-midas", local_files_only=True).to(device)
    feature_extractor = DPTFeatureExtractor.from_pretrained("/nas1/hyh22/HuggingFaceModels/Intel/dpt-hybrid-midas", local_files_only=True)

    image = feature_extractor(images=depth_image, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad(), torch.autocast("cuda"):
        depth_map = depth_estimator(image).predicted_depth

    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=(depth_image.height, depth_image.width),
        mode="bicubic",
        align_corners=False,
    )
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    image = torch.cat([depth_map] * 3, dim=1)

    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
    image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
    return image

def normalize(image):
    image = image / 127.5 - 1
    image = torch.tensor(image).unsqueeze(0).permute(0, 3, 1, 2)
    return image

def generate_image(prompt, style_img_path, input_img_path, ref_img_path, depth_img_path, mask_img_path=None, **kwargs):
    global depth_pipe
    style_img = Image.open(style_img_path)
    # style_img.resize((512, 512))
    
    input_img = load_image(input_img_path)
    ref_img = load_image(ref_img_path)
    detected_map = cv2.Canny(np.array(ref_img), 50, 200)
    # canny_map = Image.fromarray(cv2.cvtColor(detected_map, cv2.COLOR_BGR2RGB))
    canny_map = Image.fromarray(detected_map).resize((input_img.width, input_img.height))

    depth_predict_path = os.path.join(os.path.dirname(depth_img_path), "pano_depth_predicted.png")
    # if depth_pipe is None:
    #     depth_pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")
    if os.path.exists(depth_predict_path):
        print("Loading Depth...")
        depth_map = Image.open(depth_predict_path)
    else:
        print("Predicting Depth...")
        depth_img = Image.open(depth_img_path)
        depth_map = get_depth_map(depth_img)
        # depth_map = depth_pipe(depth_img)["depth"]
        depth_map.save(depth_predict_path)
    canny_map.save(os.path.join(os.path.dirname(depth_img_path), "pano_canny.png"))

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
            #                         image=[depth_map, canny_map],
            #                         controlnet_conditioning_scale=[0.0, 0.8],
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
                                    control_image=[depth_map, canny_map],
                                    controlnet_conditioning_scale=[0.0, 1.0],
                                    **kwargs,
                                )
            
            ### Without Style Guide ###
            # images = model_pipe(
            #                         prompt=c_prompt,
            #                         guidance_scale=5,
            #                         num_samples=1,
            #                         seed=42,
            #                         num_inference_steps=math.ceil(42/kwargs['strength']),
            #                         # strength=1.0,
            #                         image=input_img,
            #                         # latents=input_img_latent,
            #                         control_image=[depth_map, canny_map],
            #                         controlnet_conditioning_scale=[0.0, 1.0],
            #                         **kwargs,
            #                     ).images
            free_model()
        else:
            init_inpaint_model()
            input_img_mask = load_image(mask_img_path).convert('L')
            input_img_mask = Image.fromarray(~np.array(input_img_mask))
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
            #                         control_image=[depth_map, canny_map],
            #                         controlnet_conditioning_scale=[0.0, 1.0],
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
                            control_image=[depth_map, canny_map],
                            controlnet_conditioning_scale=[0.0, 0.8],
                            # output_type="latent",
                        ).images[0]

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
                                    control_image=[depth_map, canny_map],
                                    controlnet_conditioning_scale=[0.0, 1.0],
                                    **kwargs,
                                )
            free_inpaint_model()
            return images[0], image
            # return image
    return images[0]

if __name__ == "__main__":
    input = "preprocess/truck_cartoon_focus_colmap_cnn_refine_plus/cam_162/styled/pano_img.png"
    style = "style_data/outdoor/red_truck_cartoon.png"
    ref = "preprocess/truck_cartoon_focus_colmap_cnn_refine_plus/cam_162/pano_img.png"

    image = generate_adain(content_path=input,
                   style_path=style,
                   save_path="temp.png")
    
    image,mid = generate_image(prompt="red truck, cartoon style, red carriage, red trunk, red wood",
                            style_img_path=style,
                            input_img_path=input,
                            # input_img_path=input,
                            mask_img_path="preprocess/truck_cartoon_focus_colmap_cnn_refine_plus/cam_162/pano_mask.png",
                            ref_img_path=ref,
                            depth_img_path=ref,
                            strength=0.3,
                            )
    image.save("temp1.png")
    # mid.save("temp2.png")

    # image = generate_image(prompt="room",
    #                         style_img_path="style_data/indoor/room5.jpg",
    #                         input_img_path="test_render/cam_191/pano_img.png",
    #                         mask_img_path="preprocess/truck_cartoon_focus_colmap_cnn/cam_149/pano_mask.png",
    #                         ref_img_path="preprocess/playroom_test_train_after_proj/cam_-1/pano_img.png",
    #                         depth_img_path="preprocess/playroom_test_train_after_proj/cam_-1/pano_img.png",
    #                         strength=1.0,
    #                         )
    # image.save("temp1.png")

    # image = generate_image(prompt="red train, cartoon style, snowfield",
    #                         style_img_path="style_data/outdoor/cartoon_train2.png",
    #                         input_img_path="preprocess/train_cartoon/cam_191/pano_img.png",
    #                         # mask_img_path="preprocess/truck_cartoon_focus_colmap_cnn/cam_149/pano_mask.png",
    #                         ref_img_path="preprocess/train_cartoon/cam_191/pano_img.png",
    #                         depth_img_path="preprocess/train_cartoon/cam_191/pano_img.png",
    #                         strength=1.0,
    #                         )
    
    # image.save("temp2.png")