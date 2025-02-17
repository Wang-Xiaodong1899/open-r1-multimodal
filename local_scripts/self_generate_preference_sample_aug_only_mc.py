import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_anyres_image,tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

import json
import os
import math
from tqdm import tqdm
import torchvision.transforms as transforms
from decord import VideoReader, cpu

from transformers import AutoConfig

import cv2
import base64
import openai

from PIL import Image


import numpy as np

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument("--video_root", help="Path to the video files.", required=True)
    parser.add_argument("--output_dir", help="Directory to save the model results JSON.", required=True)
    parser.add_argument("--output_name", help="Name of the file for storing results JSON.", required=True)
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--mm_resampler_type", type=str, default="spatial_pool")
    parser.add_argument("--mm_spatial_pool_stride", type=int, default=4)
    parser.add_argument("--mm_spatial_pool_out_channels", type=int, default=1024)
    parser.add_argument("--mm_spatial_pool_mode", type=str, default="average")
    parser.add_argument("--image_aspect_ratio", type=str, default="anyres")
    parser.add_argument("--image_grid_pinpoints", type=str, default="[(224, 448), (224, 672), (224, 896), (448, 448), (448, 224), (672, 224), (896, 224)]")
    parser.add_argument("--mm_patch_merge_type", type=str, default="spatial_unpad")
    parser.add_argument("--overwrite", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--for_get_frames_num", type=int, default=4)
    parser.add_argument("--normal_frames", type=int, default=32)
    parser.add_argument("--load_8bit",  type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--prompt", type=str, default=None) 
    parser.add_argument("--api_key", type=str, help="OpenAI API key")
    parser.add_argument("--mm_newline_position", type=str, default="no_token")
    parser.add_argument("--force_sample", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--add-aug", type=bool, default=True)
    parser.add_argument("--add-hallu", type=bool, default=False) 
    parser.add_argument("--jsonl-file", type=str, default="/volsparse1/wxd/data/llava_hound/chatgpt_qa_900k.jsonl")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=2000)
    parser.add_argument("--image_resolution", type=int, default=224) 
    
    return parser.parse_args()

import random
from PIL import ImageFilter
class GaussianBlur(object):
    """Gaussian blur augmentation from SimCLR: https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def augmentation(frame, transform, state):
    torch.set_rng_state(state)
    return transform(frame)

def load_video(video_path, args):
    if os.path.isdir(video_path):
        frame_files = [os.path.join(video_path, f) for f in os.listdir(video_path) if os.path.isfile(os.path.join(video_path, f))]
        frame_files.sort()  # Ensure the frames are sorted if they are named sequentially
        num_frames_to_sample = args.normal_frames # previous author hard code sampling 10 frames

        total_frames = len(frame_files)

        sampled_indices = np.linspace(0, total_frames - 1, num_frames_to_sample, dtype=int)

        # Read and store the sampled frames
        video = []
        for idx in sampled_indices:
            frame_path = frame_files[idx]
            try:
                with Image.open(frame_path) as img:
                    frame = img.convert("RGB")
                    video.append(frame)
            except IOError:
                print(f"Failed to read frame at path: {frame_path}")
        
        # add augmentation
        # NOTE fix image_resolution 224
        aug_tranform = transforms.Compose([
            transforms.RandomResizedCrop(args.image_resolution, scale=(0.08, 0.3)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0),
            transforms.RandomHorizontalFlip()
        ])
        
        # # save original video frame
        # for (idx, v) in enumerate(video):
        #     v.save(f'{os.path.basename(video_path)}_00{idx}.jpg')
        ori_video = video
        if args.add_aug:
            state = torch.get_rng_state()
            aug_video = [augmentation(v, aug_tranform, state) for v in video]
        else:
            aug_video = video
        
        # save aug video frame
        # for (idx, v) in enumerate(video):
        #     v.save(f'{os.path.basename(video_path)}_00{idx}_aug.jpg')
        
        # NOTE fix bug
        # for_get_frames_num not work for frames dir
        total_frame_num = len(ori_video)
        sample_frame = args.for_get_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_frame, dtype=int)
        aug_video = [aug_video[idx] for idx in uniform_sampled_frames]
        # import pdb; pdb.set_trace()
        return ori_video, aug_video
    else:
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        fps = round(vr.get_avg_fps())
        frame_idx = [i for i in range(0, len(vr), fps)]
        # sample_fps = args.for_get_frames_num if total_frame_num > args.for_get_frames_num else total_frame_num
        if len(frame_idx) > args.for_get_frames_num or args.force_sample:
            sample_fps = args.for_get_frames_num
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        
        aug_tranform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(args.image_resolution, scale=(0.08, 0.3)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0),
            transforms.RandomHorizontalFlip()
        ])

        video = spare_frames
        if args.add_aug:
            state = torch.get_rng_state()
            aug_video = [augmentation(v, aug_tranform, state) for v in video]
        else:
            aug_video = video

    return aug_video


def load_video_base64(path):
    video = cv2.VideoCapture(path)

    base64Frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

    video.release()
    # print(len(base64Frames), "frames read.")
    return base64Frames


def run_inference(args):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    print(f"********************************")
    print(f"add-aug: {args.add_aug}")
    print(f"********************************")

    # Initialize the model
    if "gpt4v" != args.model_path:
        model_name = get_model_name_from_path(args.model_path)
        print(f"model_name: {model_name}")
        # TODO
        if "next" in model_name.lower():
            model_name = "llava_llama"
            print(f"model_name: {model_name}")
        if "onevision" in args.model_path or "ov" in args.model_path:
            model_name = "llava_qwen"
            print(f'***********************************')
            print(f'model_name: {model_name} !!!!!!!!!')
            print(f'***********************************')
        # Set model configuration parameters if they exist
        if args.overwrite == True:
            overwrite_config = {}
            overwrite_config["mm_spatial_pool_mode"] = args.mm_spatial_pool_mode
            overwrite_config["mm_spatial_pool_stride"] = args.mm_spatial_pool_stride
            overwrite_config["mm_newline_position"] = args.mm_newline_position

            cfg_pretrained = AutoConfig.from_pretrained(args.model_path)

            # import pdb;pdb.set_trace()
            if "qwen" not in args.model_path.lower():
                if "224" in cfg_pretrained.mm_vision_tower:
                    # suppose the length of text tokens is around 1000, from bo's report
                    least_token_number = args.for_get_frames_num*(16//args.mm_spatial_pool_stride)**2 + 1000
                else:
                    least_token_number = args.for_get_frames_num*(24//args.mm_spatial_pool_stride)**2 + 1000

                scaling_factor = math.ceil(least_token_number/4096)
                if scaling_factor >= 2:
                    if "vicuna" in cfg_pretrained._name_or_path.lower():
                        print(float(scaling_factor))
                        overwrite_config["rope_scaling"] = {"factor": float(scaling_factor), "type": "linear"}
                    overwrite_config["max_sequence_length"] = 4096 * scaling_factor
                    overwrite_config["tokenizer_model_max_length"] = 4096 * scaling_factor

            tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, load_8bit=args.load_8bit, overwrite_config=overwrite_config) #, multimodal=True)
        else:
            tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)
    else:
        pass

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_name = args.output_name
    answers_file = os.path.join(args.output_dir, f"{output_name}.jsonl")
    ans_file = open(answers_file, "w")

    video_root = args.video_root

    with open(args.jsonl_file, 'r') as file:
        jsonl_data = json.load(file)

    # import pdb;pdb.set_trace()
    for item in tqdm(jsonl_data[args.start:args.end]):

        sample_set = {}
        video_ = item["video"]
        sample_set['id'] = item["id"]
        
        
        # first QA
        question = next(convo['value'] for convo in item['conversations'] if convo['from'] == 'human')
        answer = next(convo['value'] for convo in item['conversations'] if convo['from'] == 'gpt')
        
        question = question.replace("<image>", "").replace("<video>", "")

        # delete letter prompt
        question = question.split("\nPlease")[0]

        sample_set["prompt"] = question
        sample_set["answer"] = answer
        
        sample_set["video"] = video_
        sample_set["data_source"] = item["data_source"]
        
        
        video_path = os.path.join(video_root, video_)
        
        video = None

        # Check if the video exists
        if os.path.exists(video_path):
            try:
                if "gpt4v" != args.model_path:
                    aug_video = load_video(video_path, args)
                    aug_video = image_processor.preprocess(aug_video, return_tensors="pt")["pixel_values"].half().cuda()
                    aug_video = [aug_video]
                    
                else:
                    aug_video = load_video_base64(video_path)
                    interval = int(len(video) / args.for_get_frames_num)
            except:
                print(f"{video_} video read Failed!")
                continue
        else:
            print(f"{video_} video does not exist!")
            continue
        
        K = 2
        # import pdb; pdb.set_trace()
        outputs_list = []
        for idx in range(K):
            if idx == 0:
                continue
            # chosen answer
            if "gpt4v" != args.model_path:
                qs = question
                # TODO inject GT info
                if idx==0:
                    prefix = "Here are some hints: " + answer + "\n" + "Please respond based on the given hints and video content." + "\n"
                else:
                    prefix = ""
                    if args.add_aug:
                        video = aug_video
                        qs = f"""
{qs}
First, please describe the entire video in detail based on the video content.
Then, give the reasoning process.
Finally, output the final answer.
Answer step by step.
"""
                if model.config.mm_use_im_start_end:
                    qs = prefix + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
                else:
                    qs = prefix + DEFAULT_IMAGE_TOKEN + "\n" + qs

                conv = conv_templates[args.conv_mode].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()

                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
                if tokenizer.pad_token_id is None:
                    if "qwen" in tokenizer.name_or_path.lower():
                        print("Setting pad token to bos token for qwen model.")
                        tokenizer.pad_token_id = 151643
                        
                attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                keywords = [stop_str]
                stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

                cur_prompt = question
            else:
                prompt = question

            system_error = ""

            if "gpt4v" != args.model_path:
                # print(f'video length: {len(video)}')
                with torch.inference_mode():
                    # model.update_prompt([[cur_prompt]])
                    # import pdb;pdb.set_trace()
                    # output_ids = model.generate(inputs=input_ids, images=video, attention_mask=attention_masks, modalities="video", do_sample=True, temperature=0.2, max_new_tokens=1024, use_cache=True, stopping_criteria=[stopping_criteria])
                    if "mistral" not in cfg_pretrained._name_or_path.lower():
                        # import pdb; pdb.set_trace()
                        output_ids = model.generate(inputs=input_ids, images=video, attention_mask=attention_masks, modalities="video", do_sample=True, temperature=1.0, max_new_tokens=1024, top_p=1.0, use_cache=True, stopping_criteria=[stopping_criteria])
                        # output_ids = model.generate(inputs=input_ids, images=video, attention_mask=attention_masks, modalities="video", do_sample=True, temperature=0.7, max_new_tokens=1024, top_p=0.9, use_cache=True, stopping_criteria=[stopping_criteria])
                        # output_ids = model.generate(inputs=input_ids, images=video, attention_mask=attention_masks, modalities="video", do_sample=False, temperature=0.0, max_new_tokens=1024, top_p=0.1,num_beams=1,use_cache=True, stopping_criteria=[stopping_criteria])
                        # output_ids = model.generate(inputs=input_ids, images=video, attention_mask=attention_masks, modalities="video", do_sample=True, temperature=0.2, max_new_tokens=1024, use_cache=True, stopping_criteria=[stopping_criteria])
                    else:
                        output_ids = model.generate(inputs=input_ids, images=video, attention_mask=attention_masks, modalities="video", do_sample=True, temperature=1.0, max_new_tokens=1024, top_p=1.0, use_cache=True)
                        # output_ids = model.generate(inputs=input_ids, images=video, attention_mask=attention_masks, modalities="video", do_sample=True, temperature=0.7, max_new_tokens=1024, top_p=0.9, use_cache=True)
                        # output_ids = model.generate(inputs=input_ids, images=video, attention_mask=attention_masks, modalities="video", do_sample=False, temperature=0.0, max_new_tokens=1024, top_p=0.1, num_beams=1, use_cache=True)
                        # output_ids = model.generate(inputs=input_ids, images=video, attention_mask=attention_masks, modalities="video", do_sample=True, temperature=0.2, max_new_tokens=1024, use_cache=True)

            if "gpt4v" != args.model_path:
                outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            
            # print(f"Question: {prompt}\n")
            # print(f"Response: {outputs}\n")

            if "gpt4v" == args.model_path:
                if system_error == 'content_policy_violation':
                    continue
                elif system_error == "":
                    continue
                else:
                    import pdb;pdb.set_trace()

            # import pdb;pdb.set_trace()
            if "mistral" not in cfg_pretrained._name_or_path.lower():
                if outputs.endswith(stop_str):
                    outputs = outputs[: -len(stop_str)]

            outputs = outputs.strip()
            outputs_list.append(outputs)
        
        # sample_set["chosen"] = outputs_list[0]
        sample_set["rejected"] = outputs_list[0]
        
        ans_file.write(json.dumps(sample_set, ensure_ascii=False) + "\n")
        ans_file.flush()

    ans_file.close()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
