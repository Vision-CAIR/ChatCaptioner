#!/bin/bash



VIDEO_FOLDER="webvid_data/videos"
CAPTION_FILE="webvid_data/caption.csv"
OUTPUT_FOLDER="output/"
VIDEO_LIMIT=6

python generate_caption_webvid.py ${VIDEO_FOLDER} ${CAPTION_FILE} ${OUTPUT_FOLDER} ${VIDEO_LIMIT}