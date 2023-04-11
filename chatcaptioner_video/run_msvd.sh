#!/bin/bash



VIDEO_FOLDER="msvd_data/videos"
CAPTION_FILE="msvd_data/caption.txt"
OUTPUT_FOLDER="output/"
VIDEO_LIMIT=7

python generate_caption_msvd.py ${VIDEO_FOLDER} ${CAPTION_FILE} ${OUTPUT_FOLDER} ${VIDEO_LIMIT}