#!/bin/bash
#
#SBATCH --partition=gpu         # Run on a GPU node
#SBATCH --gpus=1                # Request 1 GPU for the CUDA part
#SBATCH --job-name=gs_video     # Job Name
#SBATCH --time=00:01:00         # WallTime (adjust based on steps/ffmpeg complexity)

#SBATCH --nodes=1               # Use 1 node
#SBATCH --ntasks-per-node=1     # Run a single task (the script itself)
#SBATCH --cpus-per-task=12      # Request CPUs for ffmpeg threading (adjust as needed)
#SBATCH --mem=8G                # Memory (adjust as needed)

#SBATCH --output=slurm_gs_video_.out  # Standard output log
#SBATCH --error=slurm_gs_video_.err   # Standard error log

echo "===== Job Start ====="
date
echo "Node: $(hostname)"
echo "CPUs requested: ${SLURM_CPUS_PER_TASK}"
echo "Memory requested: ${SLURM_MEM_PER_NODE} MB"
nvidia-smi || echo "nvidia-smi not found or GPU not available" # Check GPU allocation

declare -a PATTERNS=(
    "Default 0.060 0.062"
    "Flower 0.055 0.062"
    "Mazes 0.029 0.057"
    "Mitosis 0.028 0.062"
    "Solitons 0.030 0.060"
)
# --- Parameters ---
CUDA_SOURCE="gray_scott_cuda_video.cu"
EXECUTABLE="gray_scott_cuda_video"
N_SIZE=256
SIM_STEPS=15000        # Increase steps for longer videos/more developed patterns
FEED_RATE=0.030
KILL_RATE=0.060
PATTERN_NAME="Solitons_Video" # Unique name for this video run
FRAME_INTERVAL=50      # Save a frame every 50 steps
VIDEO_FRAMERATE=30     # Frames per second for the output video
# --- End Parameters ---

# Construct directory and output video names
FRAME_DIR=$(printf "frames_%s_N%d_F%.3f_k%.3f" "$PATTERN_NAME" "$N_SIZE" "$FEED_RATE" "$KILL_RATE")
VIDEO_FILENAME=$(printf "%s_N%d_F%.3f_k%.3f_steps%d_fps%d.mp4" "$PATTERN_NAME" "$N_SIZE" "$FEED_RATE" "$KILL_RATE" "$SIM_STEPS" "$VIDEO_FRAMERATE")

echo "--- Configuration ---"
echo "Source:         $CUDA_SOURCE"
echo "Executable:     $EXECUTABLE"
echo "Grid Size:      ${N_SIZE}x${N_SIZE}"
echo "Steps:          $SIM_STEPS"
echo "Feed (F):       $FEED_RATE"
echo "Kill (k):       $KILL_RATE"
echo "Pattern Name:   $PATTERN_NAME"
echo "Frame Interval: $FRAME_INTERVAL"
echo "Frame Dir:      $FRAME_DIR"
echo "Video Out:      $VIDEO_FILENAME"
echo "Video FPS:      $VIDEO_FRAMERATE"
echo "---------------------"

echo "--- Loading Modules ---"
module purge
module load CUDA # Or specific version, e.g., CUDA/11.4
module load FFmpeg/6.0-GCCcore-12.3.0 # Or specific version
module list # Verify loaded modules
echo "---------------------"

echo "--- Compiling CUDA Code ---"
# Use C++11 features
nvcc -lm -O3 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80 \
     "$CUDA_SOURCE" -o "$EXECUTABLE"
if [ $? -ne 0 ]; then
    echo "!!! CUDA Compilation Failed !!!"
    exit 1
fi
echo "Compilation Successful."
echo "---------------------"

echo "--- Running CUDA Simulation (Frame Generation) ---"
# Run the CUDA code: N steps F k pattern_name frame_interval
srun ./"$EXECUTABLE" "$N_SIZE" "$SIM_STEPS" "$FEED_RATE" "$KILL_RATE" "$PATTERN_NAME" "$FRAME_INTERVAL"
cuda_exit_code=$?

if [ $cuda_exit_code -ne 0 ]; then
    echo "!!! CUDA Simulation Failed (Exit Code: $cuda_exit_code) !!!"
    exit 1
fi
echo "CUDA Simulation Finished."
echo "---------------------"


echo "--- Checking Frames ---"
if [ ! -d "$FRAME_DIR" ]; then
    echo "!!! Frame directory '$FRAME_DIR' not found! Cannot create video. !!!"
    exit 1
fi
frame_count=$(find "$FRAME_DIR" -name 'frame_*.ppm' | wc -l)
if [ "$frame_count" -eq 0 ]; then
     echo "!!! No PPM frames found in '$FRAME_DIR'! Cannot create video. !!!"
     exit 1
fi
echo "Found $frame_count frames in '$FRAME_DIR'."
echo "---------------------"

echo "--- Running FFmpeg (Video Creation) ---"
echo "Command: ffmpeg -y -framerate $VIDEO_FRAMERATE -i ${FRAME_DIR}/frame_%05d.ppm -c:v libx264 -pix_fmt yuv420p -threads ${SLURM_CPUS_PER_TASK} $VIDEO_FILENAME"

# Run ffmpeg using the requested CPUs
ffmpeg -y -framerate "$VIDEO_FRAMERATE" -i "${FRAME_DIR}/frame_%05d.ppm" \
       -c:v libx264 -pix_fmt yuv420p -threads "${SLURM_CPUS_PER_TASK}" \
       "$VIDEO_FILENAME"

ffmpeg_exit_code=$?

if [ $ffmpeg_exit_code -ne 0 ]; then
    echo "!!! FFmpeg Failed (Exit Code: $ffmpeg_exit_code) !!!"
    # Keep frames for debugging if ffmpeg fails
else
    echo "FFmpeg Finished Successfully."
    echo "Video saved as: $VIDEO_FILENAME"
    # Optional: Clean up frames after successful video creation
    # echo "Removing frame directory: $FRAME_DIR"
    # rm -rf "$FRAME_DIR"
fi
echo "---------------------"


echo "===== Job End ====="
date

exit $ffmpeg_exit_code # Exit with ffmpeg's status (or 0 if CUDA failed earlier)