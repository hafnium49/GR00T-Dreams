#!/bin/bash
# Simple test script for Cosmos Predict2 inference

set -e

echo "=== Simple Cosmos Predict2 Test ==="

# Extract first frame
echo "1. Extracting first frame from dataset..."
conda run -n cosmos python -c "
import cv2
from pathlib import Path

video_files = list(Path('paper_return_filtered_dataset/videos').glob('**/*.mp4'))
if video_files:
    video = str(video_files[0])
    print(f'Using video: {video}')
    cap = cv2.VideoCapture(video)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite('test_frame.jpg', frame)
        print('Saved first frame to test_frame.jpg')
    cap.release()
"

# Run simple inference test
echo -e "\n2. Running Cosmos Predict2 inference..."
cd /home/hafnium/cosmos-predict2

conda run -n cosmos python -c "
import sys
sys.path.insert(0, '.')

print('Loading Cosmos Predict2...')

from imaginaire.constants import get_cosmos_predict2_video2world_checkpoint
from imaginaire.utils.io import save_image_or_video
from cosmos_predict2.configs.base.config_video2world import get_cosmos_predict2_video2world_pipeline
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
import torch

# Create pipeline
print('Creating Video2World pipeline...')
pipe = Video2WorldPipeline.from_config(
    config=get_cosmos_predict2_video2world_pipeline(model_size='2B'),
    dit_path=get_cosmos_predict2_video2world_checkpoint(model_size='2B'),
)

# Generate video
print('Generating video...')
prompt = 'A robotic arm moving white paper into red square target area'
outputs = pipe.generate(
    prompt=prompt,
    image='/home/hafnium/GR00T-Dreams/test_frame.jpg',
    height=256,
    width=256,
    num_frames=8,
    num_inference_steps=25,
    guidance_scale=7.5,
)

# Save output
save_image_or_video(outputs.videos[0], '/home/hafnium/GR00T-Dreams/test_output.mp4')
print('✅ Video saved to test_output.mp4')
"

echo -e "\n✅ Test completed successfully!"
echo "Generated video: test_output.mp4"