# import torch
# import torch.nn as nn
# from diffusers import StableDiffusionPipeline
# import matplotlib.pyplot as plt
# from transformers import SamModel, SamProcessor
# from PIL import Image
# import numpy as np
# import json
# from collections import defaultdict
# import os
# import argparse
#
#
# def get_args_parser():
#     parser = argparse.ArgumentParser('RWD - Background Replace', add_help=False)
#     parser.add_argument('--dataset', default='Aquatic', type=str)
#     return parser
#
# def convert_bounding_boxes(img_shape, annotations):
#     H, W = img_shape[:2]
#     boxes = []
#     for row in annotations:
#         x1 = float(row[0] * W)
#         y1 = float(row[1] * H)
#         x2 = float(row[2] * W)
#         y2 = float(row[3] * H)
#         boxes.append([x1, y1, x2, y2])
#     return boxes
#
# def convert_json(json_path):
#     with open(json_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#
#     img_dict = defaultdict(list)
#
#     for cls in data:
#         for img_name, box in data[cls]:
#             img_dict[img_name].append(box)
#
#     result = []
#     for img_name, boxes in img_dict.items():
#         result.append({
#             "img_name": img_name,
#             "boxes": boxes
#         })
#
#     return result
#
# class CausalInterventionTransformation(nn.Module):
#     def __init__(self, bg_prompt, sam_id, sd_id, device):
#         super().__init__()
#         self.bg_prompt = bg_prompt
#         self.sam_model = SamModel.from_pretrained(sam_id)
#         self.sam_processor = SamProcessor.from_pretrained(sam_id)
#         self.sd_model = StableDiffusionPipeline.from_pretrained(sd_id, torch_dtype=torch.float16).to(device)
#         self.sd_model.enable_attention_slicing()
#
#     def replace_bg(self, img, boxes, save_path, save_name):
#         inputs = self.sam_processor(img, input_boxes=[boxes], return_tensors="pt")
#         outputs = self.sam_model(**inputs)
#         masks = self.sam_processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(),
#                                                              inputs["reshaped_input_sizes"].cpu())
#         scores = outputs.iou_scores
#         best_masks = []
#         gen_img = self.sd_model(prompt=self.bg_prompt).images[0]
#         gen_np = gen_img.resize((img.shape[1], img.shape[0]))
#         bg_img = np.array(gen_np)
#
#         for i in range(len(masks)):
#             for j in range(len(masks[i])):
#                 best_idx = scores[i, j].argmax().item()
#
#                 best_mask = masks[i][j][best_idx].cpu().numpy()
#
#                 best_masks.append(best_mask)
#
#         final_mask = np.zeros_like(best_masks[0], dtype=np.uint8)
#
#         for m in best_masks:
#             final_mask = np.logical_or(final_mask, m)
#
#         final_mask = final_mask.astype(np.uint8) * 255
#
#         img_pil = Image.fromarray(img)
#         bg_pil = Image.fromarray(bg_img)
#         mask_pil = Image.fromarray(final_mask,mode='L')
#
#         result = Image.composite(img_pil, bg_pil, mask_pil)
#         name, ext = os.path.splitext(save_name)
#         new_name = name + "_bg" + ext
#
#         result.save(os.path.join(save_path, new_name))
#         # result.save(os.path.join(save_path, save_name.replace('.jpg', '_bg.jpg')))
#
#         return
#
#
# def main(dataset):
#     bg_prompt = f"wide open landscape background, flat ground, distant horizon, minimal scene, clean background, natural lighting, photorealistic, high detail"
#     sam_id = "/root/autodl-tmp/inv-fomo/models/sam-vit-base"
#     sd_id = "/root/autodl-tmp/inv-fomo/models/stable-diffusion-v1-4"
#     device = "cuda"
#     RBSMSD = CausalInterventionTransformation(bg_prompt, sam_id, sd_id, device)
#     json_dir = f"/root/autodl-tmp/inv-fomo/data/RWD/ImageSets/{dataset}/few_shot_data.json"
#     img_dir = f"/root/autodl-tmp/inv-fomo/data/RWD/JPEGImages/{dataset}/"
#     save_dir = f"/root/autodl-tmp/inv-fomo/data/RWD/JPEGImages/{dataset}_bg/"
#
#     result = convert_json(json_dir)
#     for img_boxes in result:
#         img_name = img_boxes['img_name']
#         boxes = img_boxes['boxes']
#         img = Image.open(os.path.join(img_dir, img_name)).convert('RGB')
#         img = np.array(img)
#         ch_boxes =  convert_bounding_boxes(img.shape, boxes)
#         RBSMSD.replace_bg(img, ch_boxes, save_dir, img_name)
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser('RWD - Background Replace', parents=[get_args_parser()])
#     args = parser.parse_args()
#     main(args.dataset)
#
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from transformers import SamModel, SamProcessor
from PIL import Image
import numpy as np
import json
from collections import defaultdict
import os
import cv2
import random
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('RWD - Background Replace', add_help=False)
    parser.add_argument('--dataset', default='Aquatic', type=str)
    return parser


def convert_bounding_boxes(img_shape, annotations):
    H, W = img_shape[:2]
    boxes = []
    for row in annotations:
        x1 = float(row[0] * W)
        y1 = float(row[1] * H)
        x2 = float(row[2] * W)
        y2 = float(row[3] * H)
        boxes.append([x1, y1, x2, y2])
    return boxes


def load_json_grouped(json_path):

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    img_dict = defaultdict(list)
    for cls in data:
        for img_name, box in data[cls]:
            img_dict[img_name].append({
                "class": cls,
                "box": box
            })

    return img_dict

import torch

def calculate_stability_score(mask_logits, threshold=0.0, offset=1.0):

    high_threshold = threshold + offset
    low_threshold = threshold - offset

    mask_high = mask_logits > high_threshold
    mask_low = mask_logits > low_threshold

    area_high = mask_high.flatten(-2).sum(-1)
    area_low = mask_low.flatten(-2).sum(-1)

    stability_score = area_high / (area_low + 1e-6)

    return stability_score

class CausalInterventionTransformation(nn.Module):
    def __init__(self, sam_id,device):
        super().__init__()
        self.device = device
        self.sam_model = SamModel.from_pretrained(sam_id).to(device)
        self.sam_processor = SamProcessor.from_pretrained(sam_id)


#     def image_augment(self, x, radius_ratio=0.3):

#         if x.ndim == 3 and x.shape[0] in [1, 3]:  # (C,H,W)
#             x = np.transpose(x, (1, 2, 0))

#         H, W, C = x.shape

#         F = np.fft.fft2(x, axes=(0, 1))
#         F = np.fft.fftshift(F, axes=(0, 1))

#         y, x_coord = np.meshgrid(
#             np.arange(H),
#             np.arange(W),
#             indexing='ij'
#         )

#         center_y, center_x = H // 2, W // 2
#         dist = np.sqrt((y - center_y) ** 2 + (x_coord - center_x) ** 2)

#         radius = radius_ratio * min(H, W)
#         mask = (dist < radius).astype(np.float32)  # (H,W)
#         mask = mask[:, :, None]  # (H,W,1)


#         F_low = F * mask
#         F_high = F * (1 - mask)


#         noise = np.random.randn(*F_low.shape)
#         F_low_rand = F_low * (1.0 + noise)


#         F_new = F_low_rand + F_high
#         F_new = np.fft.ifftshift(F_new, axes=(0, 1))

#         x_new = np.fft.ifft2(F_new, axes=(0, 1)).real
#         x_new = np.clip(x_new, 0, 255)

#         return x_new.astype(np.uint8)

    def image_augment(self, x, radius_ratio=0.3, strength=0.1):
        if x.ndim == 3 and x.shape[0] in [1, 3]:
            x = np.transpose(x, (1, 2, 0))

        H, W, C = x.shape
        F = np.fft.fft2(x, axes=(0, 1))
        F = np.fft.fftshift(F, axes=(0, 1))

        y, x_coord = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        center_y, center_x = H // 2, W // 2
        dist = np.sqrt((y - center_y) ** 2 + (x_coord - center_x) ** 2)
        radius = radius_ratio * min(H, W)
        mask = (dist < radius).astype(np.float32)[:, :, None]

        F_low = F * mask
        F_high = F * (1 - mask)

        noise_per_channel = np.random.randn(1, 1, C) * strength 
        F_low_rand = F_low * (1.0 + noise_per_channel)

        F_new = F_low_rand + F_high
        F_new = np.fft.ifftshift(F_new, axes=(0, 1))

        x_new = np.fft.ifft2(F_new, axes=(0, 1)).real
        return np.clip(x_new, 0, 255).astype(np.uint8)

    def get_object_masks_use_sam(self, img, boxes):
        inputs = self.sam_processor(img, input_boxes=[boxes], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.sam_model(**inputs)

        masks = self.sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )
        scores = outputs.iou_scores

        best_masks = []
        valid_indices = []

        for i in range(len(masks)):
            for j in range(len(masks[i])):
                best_idx = scores[i, j].argmax().item()
                best_mask = masks[i][j][best_idx].cpu().numpy()
                best_masks.append(best_mask)
                valid_indices.append(j)

        return best_masks, valid_indices

    def augment_single_object_with_mask(self, img, box, mask, idx):
            h, w = img.shape[:2]
            x1, y1, x2, y2 = max(0, int(box[0])), max(0, int(box[1])), min(w, int(box[2])), min(h, int(box[3]))

            # 1. 提取 Patch 和对应的 Mask
            patch = img[y1:y2, x1:x2].copy()
            mask_patch = mask[y1:y2, x1:x2]
            mask_patch_3d = mask_patch[..., None]

#             # 设置保存路径
#             save_dir = "/root/autodl-tmp/inv-fomo/cit_plot"
#             os.makedirs(save_dir, exist_ok=True)

#             # --- 修正点：确保保存预览图时颜色转换正确 ---
#             # 假设输入的 img 是 RGB 格式（如果是 BGR，请将 RGB2BGR 改为直接 copy）
#             orig_bgr = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR) 
#             orig_bgra = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2BGRA)
#             orig_bgra[:, :, 3] = (mask_patch * 255).astype(np.uint8)
#             cv2.imwrite(os.path.join(save_dir, f"obj_{idx}_orig.png"), orig_bgra)

            # 2. 执行增强逻辑
            # 这里保留你的 aug_type 逻辑，但统一处理颜色空间
            aug_type = random.choice(["color", "gray", "blur", "erase"])
            aug_patch = patch.copy()

            if aug_type == "color":
            # if idx == 4:
                # 修正：如果输入是 RGB，应使用 RGB2HSV
                hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV).astype(np.float32)
                hue_shift = random.randint(-20, 20)
                hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (0.8 + random.random() * 0.4), 0, 255)
                hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (0.8 + random.random() * 0.4), 0, 255)
                aug_patch = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

            elif aug_type == "gray":
            # if idx == 0:
                gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
                aug_patch = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

            elif aug_type == "blur":
            # if idx == 2:
                k = random.choice([3, 5, 7])
                aug_patch = cv2.GaussianBlur(patch, (k, k), 0)

            elif aug_type == "erase":
            # if idx == 1:
                h_orig, w_orig = y2 - y1, x2 - x1
                eh = random.randint(int(0.2 * h_orig), int(0.3 * h_orig))
                ew = random.randint(int(0.2 * w_orig), int(0.3 * w_orig))
                ex = random.randint(0, max(0, w_orig - ew))
                ey = random.randint(0, max(0, h_orig - eh))
                mean_color = np.mean(patch, axis=(0, 1)).astype(np.uint8)
                aug_patch[ey:ey + eh, ex:ex + ew] = mean_color

            # # --- 步骤 B: 保存增强后的对象 ---
            # # 修正：保持 RGB 转 BGR 保存的逻辑一致
            # aug_bgr = cv2.cvtColor(aug_patch, cv2.COLOR_RGB2BGR)
            # aug_bgra = cv2.cvtColor(aug_bgr, cv2.COLOR_BGR2BGRA)
            # aug_bgra[:, :, 3] = (mask_patch * 255).astype(np.uint8)
            # cv2.imwrite(os.path.join(save_dir, f"obj_{idx}_aug_{aug_type}.png"), aug_bgra)

            # 3. 将增强效果应用回原图 (仅应用在 mask 区域)
            # 这一步通过 mask 确保只替换物体本身，不影响 box 内的背景
            final_patch = np.where(mask_patch_3d.astype(bool), aug_patch, patch)
            img[y1:y2, x1:x2] = final_patch

            return img, box

    def object_augment(self, img, boxes):

        img = img.copy()
        new_boxes = []
        obj_list = []

        # 1. 首先通过 SAM 提取所有 Box 的掩码
        # object_augment 返回的 best_masks 是一个列表，包含了有效目标的掩码
        best_masks, valid_indices = self.get_object_masks_use_sam(img, boxes)

        # 将 boxes 转为列表方便索引（如果原先是 tensor）
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy().tolist()

        # 2. 遍历每一个检测到的目标，结合 Mask 进行增强
        for idx in range(len(boxes)):
            box = boxes[idx]

            # 如果该 box 成功生成了 mask，则使用掩码增强
            if idx in valid_indices:
                mask_idx = valid_indices.index(idx)
                mask = best_masks[mask_idx]

                # 传入 img, box 以及对应的 mask
                # 这里 img 会被原地修改（In-place update）
                img, new_box = self.augment_single_object_with_mask(img, box, mask, idx)
                new_boxes.append(new_box)
                # 注意：此时的 img 已经是替换后的图
                obj_list.append(img.copy()) 
            else:
                # 如果 SAM 没有生成有效掩码，可以选择不增强，或退化为原有的 Box 增强
                new_boxes.append(box)
        
        # import ipdb;ipdb.set_trace()
        return img, valid_indices



    def cit(self, img, boxes, save_path, save_name):
        alpha = np.random.uniform(0, 1)
        image_aug = self.image_augment(img)
        # image_aug_ = Image.fromarray(image_aug)
        # image_aug_.save("/root/autodl-tmp/inv-fomo/cit_plot/glb.jpg")
        # print(f"img_name:{save_name}")
        
        object_aug, valid_indices = self.object_augment(img, boxes)
        # object_aug_ = Image.fromarray(object_aug)
        # object_aug_.save("/root/autodl-tmp/inv-fomo/cit_plot/obj.jpg")
        gl_img = alpha * image_aug + (1-alpha) * object_aug
        gl_img = gl_img.round().astype(np.uint8)

        name, ext = os.path.splitext(save_name)
        new_name = name + "_cit" + ext

        os.makedirs(save_path, exist_ok=True)

        save_img = Image.fromarray(gl_img)
        save_img.save(os.path.join(save_path, new_name))
        # save_img.save("/root/autodl-tmp/inv-fomo/cit_plot/cit.jpg")
        return gl_img, valid_indices




def main(dataset):
    # sam_id = "/root/autodl-tmp/inv-fomo/models/sam-vit-base"  #
    sam_id = "/root/autodl-tmp/inv-fomo/models/sam-vit-huge"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    RBSMSD = CausalInterventionTransformation( sam_id,  device)


    base_dir = "/root/autodl-tmp/inv-fomo/data/RWD"
    json_path = f"{base_dir}/ImageSets/{dataset}/few_shot_data.json"
    new_json_path = f"{base_dir}/ImageSets/{dataset}/few_shot_data_cit.json"

    img_dir = f"{base_dir}/JPEGImages/{dataset}/"
    save_dir = f"{base_dir}/JPEGImages/{dataset}_cit/"
    os.makedirs(save_dir, exist_ok=True)

    img_dict = load_json_grouped(json_path)
    new_json_data = defaultdict(list)

    for img_name, items in img_dict.items():
        boxes = [item['box'] for item in items]
        classes = [item['class'] for item in items]

        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path):
            print(f"Warning: Image not found {img_path}")
            continue

        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        ch_boxes = convert_bounding_boxes(img.shape, boxes)

        img, valid_indices = RBSMSD.cit(img, ch_boxes, save_dir, img_name)

        for idx in valid_indices:

            cls = classes[idx]
            original_box = boxes[idx]
            name, ext = os.path.splitext(img_name)
            new_name = name + "_cit" + ext

            new_json_data[cls].append([new_name, original_box])

    with open(new_json_path, 'w', encoding='utf-8') as f:
        json.dump(new_json_data, f, ensure_ascii=False, indent=4)

    print(f"Finished！New json saved to {new_json_path}")

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser('RWD - CIT', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args.dataset)
#     sam_id = "/root/autodl-tmp/inv-fomo/models/sam-vit-huge"
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     RBSMSD = CausalInterventionTransformation( sam_id,  device)
#     img_name = "/root/autodl-tmp/inv-fomo/data/RWD/JPEGImages/Aquatic/IMG_8517_MOV-0_jpg.rf.68c2e085e4da2648884e7fd22c2671fb.jpg"
#     img = Image.open(img_name).convert('RGB')
#     img = np.array(img)
#     boxes = [
#     [1107,329,1439,526],
#     [135, 294, 531, 595],
#     [297, 105, 475, 301]

# ]
#     save_dir = f"./"
#     img, valid_indices = RBSMSD.cit(img, boxes, save_dir, img_name)