from einops import rearrange, repeat
import torch
from torchvision import transforms
from PIL import Image, ImageFile
import random
from torchvision.ops.boxes import box_area

from torchvision.transforms.transforms import InterpolationMode
from torchvision.transforms import functional as F
import numpy as np
from icecream import ic
import re

ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None

from .constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

def box_iou(boxes1, area1, boxes2, eps=1e-5):
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / (union+eps)
    return iou, union

def anchor_rank(anchors, anchors_areas, input_image_size, eps=1e-5):
    # anchors x1 y1 x2 y2

    # image_size: (h, w)
    # xyxy
    input_image_bbox = torch.tensor([0, 0, input_image_size[1], input_image_size[0]]).unsqueeze(0)

    boxes1 = anchors
    boxes2 = input_image_bbox
    boxes3 = anchors.clone()
    # y2
    boxes3[:,3] = input_image_size[0]/input_image_size[1]*anchors[:,2] # 用于算分辨率无关的iou
    
    area1 = anchors_areas
    
    iou, _ = box_iou(boxes1, area1, boxes2)
    iou = iou.squeeze(1)
    shape_iou, _ = box_iou(boxes1, area1, boxes3)
    shape_iou = shape_iou.diag()
    # 优先匹配形状接近 再匹配分辨率接近
    index = torch.argmax(shape_iou*100+iou,dim=0)
    return index

class AnchorResize(torch.nn.Module):

    def __init__(self, image_size, anchors, interpolation=InterpolationMode.BILINEAR, antialias=None):
        super().__init__()
        # xyxy
        self.anchors = torch.tensor(
            [[0, 0, _[1]*image_size[1], _[0]*image_size[0]] 
            for _ in anchors], requires_grad=False
        )
        
        self.anchor_areas = box_area(self.anchors)

        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, img, skip_resize=False):
        """
        Args:
            img (PIL Image or Tensor): Image to be scaled.

        Returns:
            PIL Image or Tensor: Rescaled image.
        """
        selected_anchor = anchor_rank(self.anchors, self.anchor_areas, (img.size[1], img.size[0]))
        target_size = self.anchors[selected_anchor][2:].tolist() # w,h
        if skip_resize:
            # for debug
            return selected_anchor
        return F.resize(img, [target_size[1],target_size[0]], self.interpolation, max_size=None, antialias=self.antialias), selected_anchor

    def __repr__(self) -> str:
        detail = f"(size={self.image_size}, anchor={self.anchors}, interpolation={self.interpolation.value}, antialias={self.antialias})"
        return f"{self.__class__.__name__}{detail}"


class DocProcessor():
    def __init__(self, tokenizer=None, image_size=504, anchors='grid_12'):
        self.media_token= "<|image|>"
        # h,w
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = image_size
        # h,w
        # anchors = grid_dict[anchors]
        max_crop = int(anchors.split('_')[1])
        anchors = [(j, int(i/j)) for i in range(1,max_crop+1) for j in range(1, i+1) if i%j==0]
        self.anchors = [tuple(_) for _ in anchors]
        self.anchor_max = max([max(_) for _ in self.anchors])
        # xywh -> xyxy
        self.resizer = AnchorResize(image_size=image_size, anchors=anchors, interpolation=InterpolationMode.BICUBIC)
        self.old_resizer = transforms.Resize(image_size,interpolation=InterpolationMode.BICUBIC)
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        self.tokenizer = tokenizer
    
    def _process_image(self, images):
        new_images = []
        new_patch_position = []
        num_image_mult = []
        for image in images:
            nocut_image = self.image_transform(self.old_resizer(image)).unsqueeze(0)
                
            image, selected_anchor = self.resizer(image)
            image_input = self.image_transform(image) # h,w,3 -> 3,h,w
            # rearrange(x,'B C (n1 h) (n2 w) -> (B n1 n2) C h w', n1=self.down_sample[0], n2=self.down_sample[1])
            image_input = rearrange(image_input, 'C (num_h h) (num_w w) -> (num_h num_w) C h w', h=self.image_size[0], w=self.image_size[1])

            image_input = torch.cat([nocut_image, image_input], dim=0)

            anchor = self.anchors[selected_anchor] # w,h
            patch_position = torch.cat([
                repeat(torch.arange(anchor[0]), 'num_h -> num_h num_w 1', num_w=anchor[1]),
                repeat(torch.arange(anchor[1]), 'num_w -> num_h num_w 1', num_h=anchor[0])],dim=2)
            patch_position = rearrange(patch_position, 'num_h num_w p-> (num_h num_w) p', p=2) # num_patch, (ph,pw)

            patch_position = torch.cat([torch.ones(1,2).long()*self.anchor_max, patch_position], dim=0)

            new_images.append(image_input)
            new_patch_position.append(patch_position)
            num_image_mult.append(patch_position.shape[0])

        new_images = torch.cat(new_images,dim=0)
        new_patch_position = torch.cat(new_patch_position, dim=0)
        return new_images, new_patch_position, num_image_mult

    def __call__(self, images=None, messages=None):
        assert images is not None
        # print(images)

        ## 1. process images
        if not isinstance(images, list):
            images = [images]
        image_pils = []
        for image in images:
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            else:

                image = image.convert('RGB')
            # ic(image.size)
            image_pils.append(image)

        image_data, patch_position, num_image_mult = self._process_image(image_pils)

        ## 2. process text
        # 2.1 add image ordinal token (e.g. <img 1>) before image placeholder <|image|>
        image_index = 1 # start from 1
        for m in messages:
            try:
                assert m['role'] in ['USER', 'ASSISTANT']
            except Exception as e:
                print("Unexpected role: "+m['role']+", only support 'USER' or 'ASSISTANT'")
                exit(0)

            if m['role'] == 'USER' and self.media_token in m.get('content', ''):
                pattern = '|'.join(map(re.escape, [self.media_token]))
                text_list = re.split(f'({pattern})', m['content'])
                text = ''
                for x in text_list:
                    if x == '<|image|>':
                        text += '<img '+str(image_index)+'><|image|>'
                        image_index += 1
                    else:
                        text += x
                m['content'] = text
        
        if messages[-1]['role'] == 'USER':
            messages.append({'role':'ASSISTANT'})
        else:
            try:
                assert messages[-1].get('content', '') == ''
            except Exception as e:
                print("Unexpected end message: "+str(messages[-1]), "only (role=='USER') or (role=='ASSISTANT' and content=='') are expected.")
                exit(0)

        # print('after adding img ordinal token: ', messages)
        # 2.2 text tokenize
        seps = [' ', '</s>']
        prompt = ""
        for i, m in enumerate(messages):
            if 'content' in m:
                prompt += m['role'] + ": " + m['content'] + seps[i % 2]
            else:
                prompt += m['role'] + ":"
        ic(prompt)
        assert self.media_token in prompt
        input_ids = self.tokenizer_token(prompt)

        return image_data, patch_position, input_ids
    

    def tokenizer_token(self, prompt):
        prompt_chunks = [self.tokenizer(chunk).input_ids if len(chunk) > 0 else [] for chunk in prompt.split(DEFAULT_IMAGE_TOKEN)]

        def insert_separator(X, sep):
            return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

        input_ids = []
        offset = 0
        if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == self.tokenizer.bos_token_id:
            offset = 1
            input_ids.append(prompt_chunks[0][0])

        for x in insert_separator(prompt_chunks, [IMAGE_TOKEN_INDEX] * (offset + 1)):
            input_ids.extend(x[offset:])

        return torch.tensor(input_ids, dtype=torch.long)
            