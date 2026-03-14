#!/bin/bash
#SBATCH --time=11:59:00
#SBATCH --account=rrg-swasland
#SBATCH --mail-user=max.michet@mail.utoronto.ca
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100G
#SBATCH --mail-type=ALL
#SBATCH --array=0-3%4
#SBATCH --job-name=mar13_DPC_net_mono_only_split1
#SBATCH --output=./logs/%x/%A_%a.out

# ---------------------------------------------------------------------------
# DPC-Net MONO SLURM array job (image_0 only)
# Uses explicit train/val sequence lists (no window-level split).
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

RUN_NAME="mar13_DPC_net_mono_only_split1_${VAL_SEQS}"
echo "=== Starting ${RUN_NAME} (array task ${SLURM_ARRAY_TASK_ID}) ==="
echo "Train: ${TRAIN_SEQS}  |  Val: ${VAL_SEQS}"

module load StdEnv/2023
module load python/3.10
module load cuda/12.2

WORKDIR=/home/$USER/projects/rrg-swasland/$USER/dpc-net
cd "$WORKDIR"

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch torchvision numpy scipy Pillow tqdm

KITTI_STORE=/home/$USER/projects/rrg-swasland/datasets/KITTI_odometry

echo "Unpacking KITTI grayscale images and poses to $SLURM_TMPDIR ..."
unzip -qq "$KITTI_STORE/data_odometry_gray.zip" -d "$SLURM_TMPDIR"
unzip -qq "$KITTI_STORE/data_odometry_poses.zip" -d "$SLURM_TMPDIR"
echo "Unpack complete."

# Paths:
#   Images: $SLURM_TMPDIR/dataset/sequences/<seq>/image_0/*.png
#   GT poses: $SLURM_TMPDIR/dataset/poses/<seq>.txt
VO_PRIOR_ROOT=/home/$USER/projects/rrg-swasland/$USER/SWformer-VO-Fork/data/vo_priors_libviso2_mono
CHECKPOINT_DIR="${WORKDIR}/checkpoints/${RUN_NAME}"

mkdir -p "$CHECKPOINT_DIR" "logs/DPC_net_train"

echo "Starting DPC-Net MONO training: train=[${TRAIN_SEQS}] val=[${VAL_SEQS}] ..."
# Mono mode: omit --stereo so train_dpc_net.py duplicates image_0 into both branches.
# Explicit validation mode: use provided val sequences.
python train_dpc_net.py \
    --data_root "$SLURM_TMPDIR/dataset/sequences" \
    --poses_root "$SLURM_TMPDIR/dataset/poses" \
    --vo_prior_root "$VO_PRIOR_ROOT" \
    --use_priors \
    --train_sequences $TRAIN_SEQS \
    --val_sequences $VAL_SEQS \
    --val_split 0.0 \
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
