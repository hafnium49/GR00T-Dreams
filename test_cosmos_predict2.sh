#!/bin/bash
# Test script to run Cosmos Predict2 inference on paper_return_filtered_dataset

set -e

# Set paths
COSMOS_DIR="/home/hafnium/cosmos-predict2"
DATASET_DIR="$(pwd)/paper_return_filtered_dataset"
OUTPUT_DIR="$(pwd)/results/paper_dreams_test"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=== Testing Cosmos Predict2 Video2World Inference ==="
echo "Dataset: $DATASET_DIR"
echo "Output: $OUTPUT_DIR"

# Extract first frame from dataset for testing
echo -e "\n1. Extracting first frame from dataset..."
python3 -c "
import cv2
from pathlib import Path

video_files = list(Path('$DATASET_DIR/videos').glob('**/*.mp4'))
if video_files:
    video = str(video_files[0])
    print(f'Using video: {video}')
    cap = cv2.VideoCapture(video)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite('$OUTPUT_DIR/first_frame.jpg', frame)
        print(f'Saved first frame to $OUTPUT_DIR/first_frame.jpg')
    cap.release()
else:
    print('No videos found!')
"

# Check if first frame was extracted
if [ ! -f "$OUTPUT_DIR/first_frame.jpg" ]; then
    echo "Error: Could not extract first frame from dataset"
    exit 1
fi

# Create a simple prompt file
echo -e "\n2. Creating prompt file..."
cat > "$OUTPUT_DIR/prompt.txt" << EOF
A high-definition video of a robotic arm performing paper manipulation task. The robot carefully picks up a white paper and places it precisely into a red square target area on the table. The movements are smooth and controlled, showing the precision of the robotic system. The camera captures the entire workspace clearly.
EOF

echo "Prompt saved to $OUTPUT_DIR/prompt.txt"

# Navigate to cosmos directory and run inference
echo -e "\n3. Running Cosmos Predict2 inference..."
cd "$COSMOS_DIR"

# Make sure the cosmos-predict2 module is in path
export PYTHONPATH="${COSMOS_DIR}:${PYTHONPATH}"

# Run the example video2world script
echo "Running video2world generation..."
python examples/video2world.py \
    --model_size 2B \
    --prompt "$OUTPUT_DIR/prompt.txt" \
    --image "$OUTPUT_DIR/first_frame.jpg" \
    --save_path "$OUTPUT_DIR/generated_video.mp4" \
    --height 256 \
    --width 256 \
    --num_frames 16 \
    --num_inference_steps 50

echo -e "\n✅ Test completed successfully!"
echo "Generated video saved to: $OUTPUT_DIR/generated_video.mp4"

# Return to original directory
cd -

echo -e "\n=== Next Steps ==="
echo "1. Review the generated video at: $OUTPUT_DIR/generated_video.mp4"
echo "2. If successful, run batch generation with:"
echo "   python run_cosmos_predict2_inference.py --dataset-path $DATASET_DIR --output-dir results/paper_dreams"
echo "3. Extract actions from generated videos using IDM"
echo "4. Merge datasets for SO-101 training"