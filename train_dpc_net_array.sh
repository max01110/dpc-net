#!/bin/bash
#SBATCH --time=11:59:00
#SBATCH --account=rrg-swasland
#SBATCH --mail-user=max.michet@mail.utoronto.ca
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --mail-type=ALL
#SBATCH --array=0-3%4
#SBATCH --job-name=march_09_DPC_net_train
#SBATCH --output=./logs/%x/%A_%a.out

# ---------------------------------------------------------------------------
# DPC-Net SLURM array job — baseline comparison with SWformer
#
# Uses the same train/val splits as SWformer-VO-Fork.
# Each array task trains one split configuration.
#
# Array tasks:
#   0 → train 04 07 08 09 10, val 06
#   1 → train 04 07 08 09 10, val 05
#   2 → train 04 05 06 07 08, val 10
#   3 → train 04 05 06 07 08, val 09
# ---------------------------------------------------------------------------

case "$SLURM_ARRAY_TASK_ID" in
  0) TRAIN_SEQS="04 07 08 09 10" ; VAL_SEQS="06" ;;
  1) TRAIN_SEQS="04 07 08 09 10" ; VAL_SEQS="05" ;;
  2) TRAIN_SEQS="04 05 06 07 08" ; VAL_SEQS="10" ;;
  3) TRAIN_SEQS="04 05 06 07 08" ; VAL_SEQS="09" ;;
  *)
    echo "Unsupported SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
    exit 1
    ;;
esac

RUN_NAME="DPC_net_val_${VAL_SEQS}"
echo "=== Starting ${RUN_NAME} (array task ${SLURM_ARRAY_TASK_ID}) ==="
echo "Train: ${TRAIN_SEQS}  |  Val: ${VAL_SEQS}"

# ---- Environment ----------------------------------------------------------
module load StdEnv/2023
module load python/3.10
module load cuda/12.2

WORKDIR=/home/$USER/projects/rrg-swasland/$USER/dpc-net
cd "$WORKDIR"

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch torchvision numpy scipy Pillow tqdm

# ---- Unpack KITTI data to fast local scratch ------------------------------
KITTI_STORE=/home/$USER/projects/rrg-swasland/datasets/KITTI_odometry

echo "Unpacking KITTI grayscale images and poses to \$SLURM_TMPDIR ..."
unzip -qq "$KITTI_STORE/data_odometry_gray.zip" -d "$SLURM_TMPDIR"
unzip -qq "$KITTI_STORE/data_odometry_poses.zip" -d "$SLURM_TMPDIR"
echo "Unpack complete."

# ---- Convert PNG to JPG (same as SWformer pipeline) ----------------------
SWFORMER_DIR=/home/$USER/projects/rrg-swasland/$USER/SWformer-VO-Fork

echo "Converting PNG to JPG ..."
python "$SWFORMER_DIR/png_to_jpg.py" \
    --dataset_dir "$SLURM_TMPDIR/dataset/sequences" \
    --output_dir "$SLURM_TMPDIR/sequences_jpg"
echo "Image conversion complete."

# ---- Paths ----------------------------------------------------------------
# Images: $SLURM_TMPDIR/sequences_jpg/<seq>/image_0/*.jpg
# GT poses: $SLURM_TMPDIR/dataset/poses/<seq>.txt
# VO priors: libviso2 stereo trajectories in poses/<seq>.txt format
VO_PRIOR_ROOT=/home/$USER/projects/rrg-swasland/$USER/SWformer-VO-Fork/data/vo_priors_libviso2_stereo
CHECKPOINT_DIR="${WORKDIR}/checkpoints/${RUN_NAME}"

mkdir -p "$CHECKPOINT_DIR" "logs/DPC_net_train"

# ---- Run training ----------------------------------------------------------
# Hyperparameters from DPC-Net paper (arXiv:1709.03128, IEEE RA-L 2018):
#   - Image size: [400, 120] pixels (Sec IV.A)
#   - 30 epochs, Adam, best epoch by validation loss (Sec IV.A, Table I footnote)
#   - Δp ∈ [3, 4, 5] for training, Δp=4 for validation (Sec IV.A)
#   - batch_size 64, lr 1e-4, step_size 5, gamma 1.0 (utiasSTARS/dpc-net repo)
echo "Starting DPC-Net training: train=[${TRAIN_SEQS}] val=[${VAL_SEQS}] ..."
python train_dpc_net.py \
    --data_root "$SLURM_TMPDIR/sequences_jpg" \
    --poses_root "$SLURM_TMPDIR/dataset/poses" \
    --vo_prior_root "$VO_PRIOR_ROOT" \
    --use_priors \
    --train_sequences $TRAIN_SEQS \
    --val_sequences $VAL_SEQS \
    --kitti_test_seq "$VAL_SEQS" \
    --trained_models_dir "$CHECKPOINT_DIR" \
    --batch_size 64 \
    --num_epochs 30 \
    --lr 1e-4 \
    --step_size 5 \
    --gamma 1.0 \
    --img_width 400 \
    --img_height 120 \
    --pose_deltas 3 4 5 \
    --val_pose_delta 4

echo "=== Finished ${RUN_NAME} ==="
