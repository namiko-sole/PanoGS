import os
import io
import cv2
import base64
import requests
import json
from PIL import Image, PngImagePlugin
from transformers import pipeline

"""
    To use this example make sure you've done the following steps before executing:
    1. Ensure automatic1111 is running in api mode with the controlnet extension. 
       Use the following command in your terminal to activate:
            ./webui.sh --no-half --api
    2. Validate python environment meet package dependencies.
       If running in a local repo you'll likely need to pip install cv2, requests and PIL 
"""

depth_pipe = None

def read_image(img_path: str) -> str:
    img = cv2.imread(img_path)
    _, bytes = cv2.imencode(".png", img)
    encoded_image = base64.b64encode(bytes).decode("utf-8")
    return encoded_image

def inpaint(input_path: str,
             mask_path: str,
             canny_path: str,
             depth_path: str):
    
    global depth_pipe
    if depth_pipe is None: depth_pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf", device="cuda:1")

    input_image = read_image(input_path)
    mask_image = read_image(mask_path)

    # depth_predict_path = depth_path
    depth_predict_path = os.path.join(os.path.dirname(depth_path), "pano_depth_predicted.png")
    if os.path.exists(depth_predict_path):
        print("Loading Depth...")
    else:
        print("Predicting Depth...")
        depth_img = Image.open(depth_path)
        depth_map = depth_pipe(depth_img)["depth"]
        depth_map.save(depth_predict_path)

    img2img_payload = {
        "batch_size": 1,
        "cfg_scale": 5,
        "height": 1024,
        "width": 2048,
        "n_iter": 1,
        "steps": 20,
        "sampler_name": "DPM++ 2M Karras",
        "prompt": "",
        "negative_prompt": "",
        "seed": 42,
        "seed_enable_extras": False,
        # "seed_resize_from_h": 0,
        # "seed_resize_from_w": 0,
        # "subseed": -1,
        # "subseed_strength": 0,
        # "override_settings": {},
        # "override_settings_restore_afterwards": False,
        # "do_not_save_grid": False,
        # "do_not_save_samples": False,
        # "s_churn": 0,
        # "s_min_uncond": 0,
        # "s_noise": 1,
        # "s_tmax": None,
        # "s_tmin": 0,
        # "script_args": [],
        # "script_name": None,
        # "styles": [],
        "alwayson_scripts": {
            "ControlNet": {
                "args": [
                    {
                        "control_mode": 2,
                        "enabled": True,
                        "guidance_end": 1,
                        "guidance_start": 0,
                        "low_vram": False,
                        "model": "control_v11p_sd15_canny [d14c016b]",
                        "module": "canny",
                        "pixel_perfect": True,
                        # "processor_res": 512,
                        # "resize_mode": "Crop and Resize",
                        "threshold_a": 50,
                        "threshold_b": 200,
                        "weight": 2.0,
                        "input_image": read_image(canny_path),
                    },{
                        "control_mode": 2,
                        "enabled": True,
                        "guidance_end": 1,
                        "guidance_start": 0,
                        "low_vram": False,
                        "model": "control_v11f1p_sd15_depth [cfd03158]",
                        # "module": "depth_midas",
                        "module": "none",
                        "pixel_perfect": True,
                        # "processor_res": 512,
                        # "resize_mode": "Crop and Resize",
                        # "threshold_a": 100,
                        # "threshold_b": 200,
                        "weight": 0.6,
                        "input_image": read_image(depth_predict_path),
                    }
                ]
            }
        },
        "denoising_strength": 1,
        "initial_noise_multiplier": 1,
        "inpaint_full_res": 0,
        "inpaint_full_res_padding": 32,
        "inpainting_fill": 1,
        "inpainting_mask_invert": 1,
        "mask_blur_x": 0,
        "mask_blur_y": 0,
        "mask_blur": 4,
        "resize_mode": 0,
        "init_images": [input_image],
        "mask": mask_image,
    }

    response = requests.post(url="http://127.0.0.1:19527/sdapi/v1/img2img", json=img2img_payload).json()
    if "images" not in response:
        print(response)
        return None
    else:
        return [Image.open(io.BytesIO(base64.b64decode(base64image))) for base64image in response["images"]]


def refine(input_path: str,
            #  mask_path: str,
             canny_path: str,
             depth_path: str):
    
    input_image = read_image(input_path)
    # mask_image = read_image(mask_path)

    img2img_payload = {
        "batch_size": 1,
        "cfg_scale": 7,
        "height": 1024,
        "width": 2048,
        "n_iter": 1,
        "steps": 20,
        "sampler_name": "DPM++ 2M Karras",
        "prompt": "",
        "negative_prompt": "",
        # "seed": 42,
        "seed_enable_extras": False,
        # "seed_resize_from_h": 0,
        # "seed_resize_from_w": 0,
        # "subseed": -1,
        # "subseed_strength": 0,
        # "override_settings": {},
        # "override_settings_restore_afterwards": False,
        # "do_not_save_grid": False,
        # "do_not_save_samples": False,
        # "s_churn": 0,
        # "s_min_uncond": 0,
        # "s_noise": 1,
        # "s_tmax": None,
        # "s_tmin": 0,
        # "script_args": [],
        # "script_name": None,
        # "styles": [],
        "alwayson_scripts": {
            "ControlNet": {
                "args": [
                    {
                        "control_mode": 2,
                        "enabled": True,
                        "guidance_end": 1,
                        "guidance_start": 0,
                        "low_vram": False,
                        "model": "control_v11p_sd15_canny [d14c016b]",
                        "module": "canny",
                        "pixel_perfect": True,
                        # "processor_res": 512,
                        # "resize_mode": "Crop and Resize",
                        "threshold_a": 50,
                        "threshold_b": 200,
                        "weight": 2.0,
                        "input_image": read_image(canny_path),
                    },{
                        "control_mode": 2,
                        "enabled": True,
                        "guidance_end": 1,
                        "guidance_start": 0,
                        "low_vram": False,
                        "model": "control_v11f1p_sd15_depth [cfd03158]",
                        # "module": "none",
                        "module": "depth_midas",
                        "pixel_perfect": True,
                        # "processor_res": 512,
                        # "resize_mode": "Crop and Resize",
                        # "threshold_a": 100,
                        # "threshold_b": 200,
                        "weight": 0.4,
                        "input_image": read_image(depth_path),
                    }
                ]
            }
        },
        "denoising_strength": 0.3,
        "initial_noise_multiplier": 1,
        "inpaint_full_res": 0,
        "inpaint_full_res_padding": 32,
        "inpainting_fill": 1,
        "inpainting_mask_invert": 0,
        "mask_blur_x": 0,
        "mask_blur_y": 0,
        "mask_blur": 0,
        "resize_mode": 0,
        "init_images": [input_image],
        # "mask": mask_image,
    }

    response = requests.post(url="http://127.0.0.1:19527/sdapi/v1/img2img", json=img2img_payload).json()
    if "images" not in response:
        print(response)
        return None
    else:
        return [Image.open(io.BytesIO(base64.b64decode(base64image))) for base64image in response["images"]]


if __name__ == "__main__":
    # url = "http://127.0.0.1:19527/sdapi/v1/"

    images = inpaint(#url=url + "img2img",
                      input_path="preprocess/playroom_test/cam_57/styled/pano_img.png",
                      mask_path="preprocess/playroom_test/cam_57/mask/pano_img.png",
                    #   mask_path="preprocess/playroom_test/cam_57/pano_mask.png",
                      canny_path="preprocess/playroom_test/cam_57/pano_img.png",
                      depth_path="preprocess/playroom_test/cam_57/pano_img.png",
                    #   depth_path="temp3.png",
                      )
    images[0].save("temp.png")
    images[1].save("temp1.png")
    images[2].save("temp2.png")

    # images = refine(input_path="/data/hyh/github/2D-GS-Viser-Viewer/preprocess/playroom_room3/cam_20/styled/pano_styled.png",
    #                 #   mask_path="/data/hyh/github/2D-GS-Viser-Viewer/preprocess/playroom/cam_101/proj_mask_mid.png",
    #                   canny_path="/data/hyh/github/2D-GS-Viser-Viewer/preprocess/playroom_room3/cam_20/pano_img.png",
    #                   depth_path="/data/hyh/github/2D-GS-Viser-Viewer/preprocess/playroom_room3/cam_20/pano_depth.png",)
    # images[0].save("temp.png")
    i = 4




    # generate(url + "txt2img", txt2img_payload)

    # url = "http://127.0.0.1:19527"

    # payload = {
    #     "prompt": "puppy dog",
    #     "steps": 5
    # }

    # response = requests.post(url=f'http://127.0.0.1:19527/sdapi/v1/txt2img', json=payload)

    # r = response.json()

    # for i in r['images']:
    #     image = Image.open(io.BytesIO(base64.b64decode(i.split(",",1)[0])))

    #     png_payload = {
    #         "image": "data:image/png;base64," + i
    #     }
    #     response2 = requests.post(url=f'{url}/sdapi/v1/png-info', json=png_payload)

    #     pnginfo = PngImagePlugin.PngInfo()
    #     pnginfo.add_text("parameters", response2.json().get("info"))
    #     image.save('output.png', pnginfo=pnginfo)