import cv2
import base64
from openai import OpenAI
import os

import os
from openai import OpenAI
client = OpenAI(
    base_url="https://api.ai-gaochao.cn/v1/",
    api_key=os.getenv("OPENAI_API_KEY"),
)

# ref: https://cookbook.openai.com/examples/gpt4o/introduction_to_gpt4o
def process_video(video_path, seconds_per_frame=1, max_frames=31):
    base64Frames = []
    base_video_path, _ = os.path.splitext(video_path)
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    total_time = total_frames / fps
    # frames_to_skip = int(fps * seconds_per_frame)
    frames_to_skip = int(total_frames / max_frames)
    curr_frame = 0

    while curr_frame < total_frames - 1:
        video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        curr_frame += frames_to_skip
    video.release()

    return base64Frames

def summarize_video(file_path):
    base64Frames = process_video(file_path, seconds_per_frame=0.5)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """You are an expert to analyze the video and provide useful information for users.""",
            },
            {
                "role": "user",
                "content": [
                    "These are frames taken from the video",
                    *map(
                        lambda x: {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpg;base64,{x}",
                                "detail": "low",
                            },
                        },
                        base64Frames,
                    ),
                    {
                        "type": "text",
                        "text": "Please describe this video in detail",
                    },
                ],
            },
        ],
        temperature=0,
    )
    
    print(response.choices[0].message.content)
    return response.choices[0].message.content

if __name__ == "__main__":
    summarize_video("/data/wangxd/LLaVA-Video-178K/2_3_m_youtube_v0_1/liwei_youtube_videos/videos/youtube_video_2024/ytb_zF1Xc3Zx_2k.mp4")