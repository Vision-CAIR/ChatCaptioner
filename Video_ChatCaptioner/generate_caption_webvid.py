import cv2
import numpy as np
import json
import sys
import os
import yaml
import torch
from PIL import Image
import csv


from chatcaptioner.video_chat import set_openai_key,caption_for_video
from chatcaptioner.blip2 import Blip2
from chatcaptioner.utils import RandomSampledDataset, plot_img, print_info
from chatcaptioner.video_reader import read_video_with_timestamp, read_video_sampling,read_video_with_timestamp_key_frame



VIDEO_FOLDER=sys.argv[1]
CAPTION_FILE = sys.argv[2]
OUTPUT_FOLDER=sys.argv[3]
VIDEO_LIMIT=int(sys.argv[4])



blip2s = {
    'FlanT5 XXL': Blip2('FlanT5 XXL', device_id=0, bit8=True)
}


data_file = {}
with open(CAPTION_FILE, 'r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  # Skip header row
    for row in csv_reader:
        video_id = row[0]
        duration = row[2]
        folder = row[3]
        caption = row[-1]
        
        data_file[video_id] = {"duration":duration, "folder": folder, "caption": caption}



# find all the video paths
video_files = []
for root, dirs, files in os.walk(VIDEO_FOLDER):
    for filename in files:
        full_path = os.path.join(root, filename)
        video_files.append(full_path)


# extract the video frames with uniform sampling
video_list = []
for video_path in video_files[:VIDEO_LIMIT]:
    video_id = video_path.split("/")[-1].replace(".mp4","")
    if video_id in data_file.keys():
        new_json_file = {}
        new_json_file["video_id"] = video_id
        new_json_file["video_path"] = video_path
        new_json_file["annotation"] = data_file[video_id]
        
        try:
            sampled_frames = read_video_sampling(video_path, num_frames=8)
            new_json_file["features"]=sampled_frames
            video_list.append(new_json_file)
        except:
            pass



for sample in video_list:
    video_id = sample["video_id"]
    features = sample["features"]
    output = sample["annotation"]["folder"]

    if not os.path.exists(OUTPUT_FOLDER+output):
        os.makedirs(OUTPUT_FOLDER+output)
    with open(OUTPUT_FOLDER+output+"/"+video_id+".txt","w") as f:
        sub_summaries = caption_for_video(blip2s['FlanT5 XXL'], features, print_mode="chat",n_rounds=30, model='gpt-3.5-turbo')
        caption =  sample["annotation"]["caption"]
        f.write("ground truth: "+ caption+"\n\n\n")
        f.write("chatCaptioner: " +sub_summaries["ChatCaptioner"]["caption"]+"\n\n\n")
        f.write("chat log:\n")
        for element in sub_summaries["ChatCaptioner"]["chat"]:
            f.write(element["content"]+"\n")
