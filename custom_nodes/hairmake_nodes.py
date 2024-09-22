from PIL import Image
import numpy as np
import torch
import os
import boto3

from configparser import RawConfigParser

if platform.system() == "Windows":
    conf_path = "config_win.ini"
else:
    conf_path = "config.ini"

CONF = RawConfigParser()
CONF.read(conf_path)



S3_CLIENT = boto3.client('s3',
                         aws_access_key_id=CONF.get('aws', 'aws_access_key_id'),
                         aws_secret_access_key=CONF.get('aws', 'aws_secret_access_key'),
                         region_name=CONF.get('aws', 'region'))


def upload(s3_client, bucket_name, local_file, s3_path, is_public=True):
    args = None
    if is_public:
        args = {'ACL': 'public-read'}
    s3_client.upload_file(local_file, bucket_name, s3_path, ExtraArgs=args)

class S3ImageUploader:
    def __init__(self):
        self.type = "output"

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"images": ("IMAGE", ),
                     "request_hash": ("STRING", {"default": "ComfyUI"}),
                     },
                }

    RETURN_TYPES = ()
    FUNCTION = "upload_image"

    OUTPUT_NODE = True

    CATEGORY = "image"

    def upload_image(self, images, request_hash):
        image = images[0]

        i = 255. * image.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        os.makedirs("tmp", exist_ok=True)
        local_path = f"tmp/{request_hash}_0.jpg"
        img.save(local_path, quality=100)

        s3_image_path = f"gen_hair_images/{request_hash}_0.jpg"
        try:
            upload(S3_CLIENT, 'hovits-bucket', local_path, s3_image_path)
        except Exception as e:
            print(e)
            # todo: set error status
        finally:
            os.remove(local_path)
            return {"ui": {"images": []}}

class S3ImageUploaderAndCache:
    def __init__(self):
        self.type = "output"

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"images": ("IMAGE", ),
                     "request_hash": ("STRING", {"default": "ComfyUI"}),
                     "cache_dir": ("STRING", {"default": "cache_images"})
                     },
                }

    RETURN_TYPES = ()
    FUNCTION = "upload_image"

    OUTPUT_NODE = True

    CATEGORY = "image"

    def upload_image(self, images, request_hash, cache_dir):
        image = images[0]

        i = 255. * image.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        os.makedirs(cache_dir, exist_ok=True)
        local_path = os.path.join(cache_dir, f"{request_hash}.jpg")
        img.save(local_path, quality=100)

        s3_image_path = f"gen_pre_masks/{request_hash}.jpg"
        try:
            upload(S3_CLIENT, 'hovits-bucket', local_path, s3_image_path)
        except Exception as e:
            print(e)
            # todo: set error status

        return {"ui": {"images": []}}


CLOTH_SPLIT_TAGS = [
    "shirt", "dress", "sweater", "vest", "sleeves", "suit", "jacket", "hoodie",
    "coat", "blazer", "cloth", "military", "parka", "cardigan", "cape", "clothes", "poncho", "kimono",
    "hanbok", "hanfu", "swimsuit", "bikini", "blouse", "downblouse", "top", "print"
]

CLOTH_TAGS = [
    "t-shirt", "collarbone", "sleeveless"
]

CLOTH_END_TAGS = [
    "tank top", "crop top", "bare shoulders", "tube top"
]


class CLIPTextEncodeFromWd14WithShirtAndBg:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True}),
                             "clip": ("CLIP",),
                             "wd14_tags": ("STRING", {"forceInput": True}),
                             "target_type": (["shirt_and_bg", "bg"],),
                             "append_to": (["first", "last"],),
                             "cloth_weight": ("STRING", {"default": "1.4"}),
                             "bg_weight": ("STRING", {"default": "1.3"}),
                             }}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"

    CATEGORY = "conditioning"

    def encode(self, clip, text, wd14_tags, target_type, append_to, cloth_weight, bg_weight):
        tags = wd14_tags.split(", ")
        clothes = []
        bgs = []
        for tag in tags:
            split_tags = tag.split(" ")
            if target_type == "shirt_and_bg":
                if any([t in split_tags for t in CLOTH_SPLIT_TAGS]):
                    clothes.append(f"({tag}:{cloth_weight})")
                if tag in CLOTH_TAGS:
                    clothes.append(f"({tag}:{cloth_weight})")
                if any([tag.endswith(t) for t in CLOTH_END_TAGS]):
                    clothes.append(f"({tag}:{cloth_weight})")
            if "background" in split_tags:
                bgs.append(f"({tag}:{bg_weight})")
        if target_type == "shirt_and_bg":
            if append_to == "first":
                prompt = f"{', '.join(clothes)}, {', '.join(bgs)}, {text}"
            else:
                prompt = f"{text}, {', '.join(clothes)}, {', '.join(bgs)}"
        else:
            if append_to == "first":
                prompt = f"{', '.join(bgs)}, {text}"
            else:
                prompt = f"{text}, {', '.join(bgs)}"

        tokens = clip.tokenize(prompt)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return ([[cond, {"pooled_output": pooled}]],)


class FillOrEmptyMaskRegionOverOrUnderMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "target_mask": ("MASK",),
            "by_mask": ("MASK",),
            "fill_or_empty": (["fill", "empty"],),
            "masking_direction": (["top", "bottom", "right", "left"],),
            "masking_range": ("STRING", {"default": "all"}),
            "by_direction": (["top", "bottom", "right", "left"],),
            "start_margin": ("INT", {"default": 0, "step": 1}),
            "use_other_axis_range": ("BOOLEAN", {"default": False}),
        }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "doit"

    CATEGORY = "mask"

    def doit(self, target_mask, by_mask, fill_or_empty, masking_direction, masking_range, by_direction, start_margin,
             use_other_axis_range):
        target_mask = target_mask.cpu()

        by_mask = by_mask.cpu()
        cv2_mask1 = np.array(target_mask) * 255
        cv2_mask2 = np.array(by_mask) * 255
        if cv2_mask1.shape == cv2_mask2.shape:
            # convert cv2_mask2 to binary mask
            width = cv2_mask2.shape[2]
            height = cv2_mask2.shape[1]
            cv2_mask2_binary = cv2_mask2 > 0
            cv2_mask2_binary = cv2_mask2_binary[0]
            cv2_mask2_y_axis_binary = np.sum(cv2_mask2_binary, axis=1)
            cv2_mask2_x_axis_binary = np.sum(cv2_mask2_binary, axis=0)
            print("cv2_mask2_y_axis_binary", cv2_mask2_y_axis_binary.shape)
            print("cv2_mask2_x_axis_binary", cv2_mask2_x_axis_binary.shape)

            cv2_mask2_y_axis_binary_index = np.where(cv2_mask2_y_axis_binary > 0)
            cv2_mask2_x_axis_binary_index = np.where(cv2_mask2_x_axis_binary > 0)
            if len(cv2_mask2_y_axis_binary_index[0]) < 1:
                return (target_mask,)
            # get lowest index of true value in cv2_mask2_y_axis_binary_index
            if by_direction == "top":
                idx = cv2_mask2_y_axis_binary_index[0][0]
                idx = max(idx - 1, 0)
                idx = min(idx, height)
                if use_other_axis_range:
                    other_start_idx = cv2_mask2_x_axis_binary_index[0][0]
                    other_end_idx = cv2_mask2_x_axis_binary_index[0][-1]
            elif by_direction == "bottom":
                idx = cv2_mask2_y_axis_binary_index[0][-1]
                idx = max(idx + start_margin, 0)
                idx = min(idx, height)
                if use_other_axis_range:
                    other_start_idx = cv2_mask2_x_axis_binary_index[0][0]
                    other_end_idx = cv2_mask2_x_axis_binary_index[0][-1]
            elif by_direction == "right":
                idx = cv2_mask2_x_axis_binary_index[0][-1]
                idx = max(idx + start_margin, 0)
                idx = min(idx, width)
                if use_other_axis_range:
                    other_start_idx = cv2_mask2_y_axis_binary_index[0][0]
                    other_end_idx = cv2_mask2_y_axis_binary_index[0][-1]
            elif by_direction == "left":
                idx = cv2_mask2_x_axis_binary_index[0][0]
                idx = max(idx + start_margin, 0)
                idx = min(idx, width)
                if use_other_axis_range:
                    other_start_idx = cv2_mask2_y_axis_binary_index[0][0]
                    other_end_idx = cv2_mask2_y_axis_binary_index[0][-1]
            if use_other_axis_range:
                if masking_range == "all" or not masking_range.isdigit():
                    if masking_direction == "top":
                        cv2_mask1[:, :idx, other_start_idx:other_end_idx] = 0 if fill_or_empty == "empty" else 255
                    elif masking_direction == "bottom":
                        cv2_mask1[:, idx:, other_start_idx:other_end_idx] = 0 if fill_or_empty == "empty" else 255
                    elif masking_direction == "right":
                        cv2_mask1[:, other_start_idx:other_end_idx, idx:] = 0 if fill_or_empty == "empty" else 255
                    elif masking_direction == "left":
                        cv2_mask1[:, other_start_idx:other_end_idx, :idx] = 0 if fill_or_empty == "empty" else 255
                elif masking_range.isdigit():
                    if masking_direction == "top":
                        cv2_mask1[:, idx - int(masking_range):idx,
                        other_start_idx:other_end_idx] = 0 if fill_or_empty == "empty" else 255
                    elif masking_direction == "bottom":
                        cv2_mask1[:, idx:idx + int(masking_range),
                        other_start_idx:other_end_idx] = 0 if fill_or_empty == "empty" else 255
                    elif masking_direction == "right":
                        cv2_mask1[:, other_start_idx:other_end_idx,
                        idx:idx + int(masking_range)] = 0 if fill_or_empty == "empty" else 255
                    elif masking_direction == "left":
                        cv2_mask1[:, other_start_idx:other_end_idx,
                        idx - int(masking_range):idx] = 0 if fill_or_empty == "empty" else 255
            else:
                if masking_range == "all" or not masking_range.isdigit():
                    if masking_direction == "top":
                        cv2_mask1[:, :idx, :] = 0 if fill_or_empty == "empty" else 255
                    elif masking_direction == "bottom":
                        cv2_mask1[:, idx:, :] = 0 if fill_or_empty == "empty" else 255
                    elif masking_direction == "right":
                        cv2_mask1[:, :, idx:] = 0 if fill_or_empty == "empty" else 255
                    elif masking_direction == "left":
                        cv2_mask1[:, :, :idx] = 0 if fill_or_empty == "empty" else 255
                elif masking_range.isdigit():
                    if masking_direction == "top":
                        cv2_mask1[:, idx - int(masking_range):idx, :] = 0 if fill_or_empty == "empty" else 255
                    elif masking_direction == "bottom":
                        cv2_mask1[:, idx:idx + int(masking_range), :] = 0 if fill_or_empty == "empty" else 255
                    elif masking_direction == "right":
                        cv2_mask1[:, :, idx:idx + int(masking_range)] = 0 if fill_or_empty == "empty" else 255
                    elif masking_direction == "left":
                        cv2_mask1[:, :, idx - int(masking_range):idx] = 0 if fill_or_empty == "empty" else 255

            return torch.clamp(torch.from_numpy(cv2_mask1) / 255.0, min=0, max=1)
        else:
            # do nothing - incompatible mask shape: mostly empty mask
            return (target_mask,)

class MakeBangsMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "face_and_hair_mask": ("MASK", ),
                        "eyebrows_mask": ("MASK", ),
                      }
                }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "doit"

    CATEGORY = "mask"

    def doit(self, face_and_hair_mask, eyebrows_mask):
        face_and_hair_mask = face_and_hair_mask.cpu()
        eyebrows_mask = eyebrows_mask.cpu()

        cv2_mask1 = np.array(face_and_hair_mask) * 255
        cv2_mask2 = np.array(eyebrows_mask) * 255
        # merge hair and face mask
        print("1", cv2_mask1.shape)
        print("2", cv2_mask2.shape)
        if cv2_mask1.shape == cv2_mask2.shape:
            #convert cv2_mask2 to binary mask
            cv2_mask2_binary = cv2_mask2 > 0
            cv2_mask2_binary = cv2_mask2_binary[0]
            cv2_mask2_binary = np.sum(cv2_mask2_binary, axis=1)

            cv2_mask2_binary_index = np.where(cv2_mask2_binary > 0)
            if len(cv2_mask2_binary_index[0]) < 1:
                return (face_and_hair_mask,)
            # get lowest index of true value in cv2_mask2_binary_index
            idx = cv2_mask2_binary_index[0][0]
            idx = max(idx-1, 0)
            cv2_mask1[:, idx:, :] = 0

            return torch.clamp(torch.from_numpy(cv2_mask1) / 255.0, min=0, max=1)
        else:
            # do nothing - incompatible mask shape: mostly empty mask
            return (face_and_hair_mask,)

NODE_CLASS_MAPPINGS = {
    "CLIPTextEncodeFromWd14WithShirtAndBg": CLIPTextEncodeFromWd14WithShirtAndBg,
    "FillOrEmptyMaskRegionOverOrUnderMask": FillOrEmptyMaskRegionOverOrUnderMask,
    "S3ImageUploaderAndCache": S3ImageUploaderAndCache,
    "S3ImageUploader": S3ImageUploader,
    "MakeBangsMask": MakeBangsMask
}
