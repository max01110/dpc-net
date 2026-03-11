import argparse
import math
import os
import time
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from lie_algebra import se3_inv, se3_log, so3_log, so3_to_rpy
from losses import (
    SE3GeodesicLoss,
    SO3GeodesicLoss,
    compute_loss,
    compute_loss_rot,
    compute_loss_yaw,
)
from net import (
    DeepPoseCorrectorMonoRotation,
    DeepPoseCorrectorMonoYaw,
    DeepPoseCorrectorStereoFullPose,
)
from utils import AverageMeter, save_checkpoint


def _load_kitti_image_paths(data_root: str, sequence: str) -> List[str]:
    seq_dir = os.path.join(data_root, sequence, "image_0")
    if not os.path.isdir(seq_dir):
        raise FileNotFoundError(f"Expected KITTI images under {seq_dir}")
    # KITTI uses zero-padded frame ids
    frames = sorted(
        [
            os.path.join(seq_dir, f)
            for f in os.listdir(seq_dir)
            if f.endswith(".png") or f.endswith(".jpg")
        ]
    )
    if len(frames) < 2:
        raise ValueError(f"Found fewer than 2 frames in {seq_dir}")
    return frames


def _load_vo_trajectory(vo_prior_root: str, sequence: str) -> List[np.ndarray]:
    """Load the VO estimator's absolute trajectory from poses/XX.txt.

    These are libviso2 stereo output in KITTI 12-value format, exactly the
    same structure as KITTI ground-truth pose files.
    """
    poses_file = os.path.join(vo_prior_root, "poses", f"{sequence}.txt")
    if not os.path.exists(poses_file):
        raise FileNotFoundError(
            f"VO trajectory file not found: {poses_file}. "
            f"Expected libviso2 absolute poses in "
            f"<vo_prior_root>/poses/<seq>.txt"
        )
    return _load_kitti_poses(poses_file, _raw_path=True)


def _load_kitti_poses(poses_root: str, sequence: str = None, _raw_path: bool = False) -> List[np.ndarray]:
    if _raw_path:
        poses_txt = poses_root
    else:
        poses_txt = os.path.join(poses_root, f"{sequence}.txt")
    if not os.path.exists(poses_txt):
        raise FileNotFoundError(f"Could not find KITTI poses file: {poses_txt}")

    Twv_list: List[np.ndarray] = []
    with open(poses_txt, "r") as f:
        for line in f:
            vals = [float(x) for x in line.strip().split()]
            if len(vals) != 12:
                continue
            T = np.eye(4, dtype=np.float64)
            T[:3, :4] = np.array(vals, dtype=np.float64).reshape(3, 4)
            Twv_list.append(T)
    return Twv_list


class KITTIDPCFromSWF(Dataset):
    """
    KITTI dataset for DPC-Net, built from:
      - KITTI grayscale images (converted to jpg by SWformer pipeline),
      - KITTI ground-truth poses (12-value txt, one per frame),
      - libviso2 stereo VO trajectory in <vo_prior_root>/poses/<seq>.txt
        (same 12-value KITTI format, the VO estimator's absolute poses).

    For each consecutive frame pair i -> i+1:
        T_gt   = Twv_gt[i]^{-1}  * Twv_gt[i+1]      (GT relative pose)
        T_est  = Twv_est[i]^{-1} * Twv_est[i+1]      (VO relative pose)
        T_corr = T_gt * T_est^{-1}                    (correction target)

    This is exactly how the original DPC-Net create_kitti_training_data.py
    builds its training data.
    """

    def __init__(
        self,
        data_root: str,
        poses_root: str,
        sequences: List[str],
        img_dims: Tuple[int, int],
        img_mean: List[float],
        img_std: List[float],
        vo_prior_root: str = None,
        run_type: str = "train",
        use_priors: bool = True,
        pose_deltas: List[int] = None,
    ):
        super().__init__()
        assert run_type in ("train", "validate")

        self.data_root = data_root
        self.poses_root = poses_root
        self.sequences = sequences
        self.vo_prior_root = vo_prior_root
        self.run_type = run_type
        self.use_priors = use_priors and vo_prior_root is not None

        # Paper: "For non-distorted data, we use Δp ∈ [3, 4, 5] for training, and test with Δp = 4"
        if pose_deltas is None:
            pose_deltas = [1] if run_type == "train" else [1]
        self.pose_deltas = pose_deltas

        self.transform_img = transforms.Compose(
            [
                transforms.Resize(img_dims),
                transforms.ToTensor(),
                transforms.Normalize(mean=img_mean, std=img_std),
            ]
        )

        self.image_quads: List[Tuple[str, str, str, str]] = []
        self.T_corr: List[np.ndarray] = []
        self.T_gt: List[np.ndarray] = []
        self.T_est: List[np.ndarray] = []

        for seq in sequences:
            Twv_gt = _load_kitti_poses(poses_root, seq)
            img_paths = _load_kitti_image_paths(data_root, seq)

            if self.use_priors:
                Twv_est = _load_vo_trajectory(vo_prior_root, seq)
            else:
                Twv_est = None

            n = min(len(Twv_gt), len(img_paths))
            if Twv_est is not None:
                n = min(n, len(Twv_est))
            if n < 2:
                continue

            for delta in self.pose_deltas:
                if delta < 1 or n - delta < 1:
                    continue
                for i in range(0, n - delta):
                    img0 = img_paths[i]
                    img1 = img_paths[i + delta]

                    self.image_quads.append((img0, img0, img1, img1))

                    T_12_gt = np.linalg.inv(Twv_gt[i]) @ Twv_gt[i + delta]

                    if Twv_est is not None:
                        T_12_est = np.linalg.inv(Twv_est[i]) @ Twv_est[i + delta]
                    else:
                        T_12_est = np.eye(4, dtype=np.float64)

                    T_corr = T_12_gt @ np.linalg.inv(T_12_est)

                    self.T_gt.append(T_12_gt)
                    self.T_est.append(T_12_est)
                    self.T_corr.append(T_corr)

        if len(self.T_corr) == 0:
            raise RuntimeError(
                f"No training samples constructed from sequences={sequences} "
                f"with data_root={data_root} and poses_root={poses_root}"
            )

        # Empirical precision matrix over training corrections (pose mode only)
        self.train_se3_precision = self._compute_se3_precision()
        # For compatibility with original code
        self.sequence = "+".join(sequences)
        self.test_pose_delta = self.pose_deltas[0] if len(self.pose_deltas) == 1 else 4

    def _compute_se3_precision(self) -> torch.Tensor:
        num = len(self.T_corr)
        # Stack corrections into a [N, 4, 4] tensor and use se3_log
        T_corr_np = np.stack(self.T_corr, axis=0).astype(np.float32)
        T_corr_t = torch.from_numpy(T_corr_np)
        xi = se3_log(T_corr_t)  # [N, 6]
        xi_np = xi.cpu().numpy().T  # [6, N]
        cov = np.cov(xi_np)
        # Tikhonov regularization to avoid singular/ill-conditioned covariance
        trace = np.trace(cov)
        eps = 1e-6 * (trace / 6.0 if trace > 0 else 1.0)
        cov_reg = cov + eps * np.eye(6, dtype=cov.dtype)
        precision = np.linalg.inv(cov_reg)
        return torch.from_numpy(precision.astype(np.float32))

    def __len__(self) -> int:
        return len(self.image_quads)

    def _read_image(self, path: str):
        img = Image.open(path).convert("RGB")
        return self.transform_img(img)

    def __getitem__(self, idx: int):
        paths = self.image_quads[idx]
        imgs = [self._read_image(p) for p in paths]

        T_corr = self.T_corr[idx]
        T_gt = self.T_gt[idx]
        T_est = self.T_est[idx]

        target_se3 = torch.from_numpy(T_corr.astype(np.float32))
        target_rot = torch.from_numpy(T_corr[:3, :3].astype(np.float32))

        R_gt = torch.from_numpy(T_gt[:3, :3].astype(np.float32)).unsqueeze(0)
        R_est = torch.from_numpy(T_est[:3, :3].astype(np.float32)).unsqueeze(0)
        rpy_gt = so3_to_rpy(R_gt)   # [1, 3] = [roll, pitch, yaw]
        rpy_est = so3_to_rpy(R_est)
        # Vehicle yaw ~ camera pitch (index 1)
        target_yaw = torch.tensor(
            [rpy_gt[0, 1].item() - rpy_est[0, 1].item()], dtype=torch.float32
        )

        return imgs, target_rot, target_yaw, target_se3


def train(
    epoch: int,
    model,
    train_loader,
    optimizer,
    loss_fn,
    precision,
    config,
    correction_type: str,
):
    print('Training...')

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    end = time.time()

    for batch_idx, (image_quad, target_rot, target_yaw, target_se3) in enumerate(
        train_loader
    ):
        data_time.update(time.time() - end)
        batch_size = image_quad[0].size(0)

        if correction_type == "pose":
            loss, output = compute_loss(
                image_quad,
                target_se3,
                model,
                loss_fn,
                precision,
                config,
                mode="train",
            )
        elif correction_type == "rotation":
            loss, output = compute_loss_rot(
                image_quad,
                target_rot,
                model,
                loss_fn,
                precision,
                config,
                mode="train",
            )
        elif correction_type == "yaw":
            loss, output = compute_loss_yaw(
                image_quad,
                target_yaw,
                model,
                loss_fn,
                precision,
                config,
                mode="train",
            )
        else:
            raise ValueError("correction_type must be 'pose', 'rotation', or 'yaw'")

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  [WARN] Skipping batch {batch_idx}: loss is {loss.item()}")
            continue

        losses.update(loss.item(), batch_size)

        optimizer.zero_grad()
        loss.backward()

        # Check for NaN gradients before stepping
        has_nan_grad = False
        for p in model.parameters():
            if p.grad is not None and torch.isnan(p.grad).any():
                has_nan_grad = True
                break
        if has_nan_grad:
            print(f"  [WARN] Skipping batch {batch_idx}: NaN gradients detected")
            optimizer.zero_grad()
            continue

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if (
            batch_idx % config["train_output_interval"] == 0
            or batch_idx == len(train_loader) - 1
        ):
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Batch {batch_time.val:.3f} (avg: {batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} (avg: {data_time.avg:.3f})\t"
                "Loss {loss.val:.2E} (avg: {loss.avg:.2E})\t".format(
                    epoch,
                    batch_idx,
                    len(train_loader) - 1,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                )
            )


def validate(valid_loader, model, loss_fn, precision, config, correction_type: str):
    val_seq = valid_loader.dataset.sequence
    print(f"Validating with sequences {val_seq}...")

    batch_time = AverageMeter()
    losses = AverageMeter()
    num_quads = len(valid_loader.dataset)

    if correction_type == "rotation":
        D = 3
    elif correction_type == "pose":
        D = 6
    else:
        D = 1

    predictions = torch.zeros(num_quads, D)
    targets = torch.zeros(num_quads, D)

    model.eval()

    end = time.time()
    start_idx = 0
    with torch.no_grad():
        for batch_idx, (image_quad, target_rot, target_yaw, target_se3) in enumerate(
            valid_loader
        ):
            batch_size = image_quad[0].size(0)

            if correction_type == "pose":
                loss, output = compute_loss(
                    image_quad,
                    target_se3,
                    model,
                    loss_fn,
                    precision,
                    config,
                    mode="eval",
                )
                targets[start_idx : start_idx + batch_size, :] = se3_log(target_se3)
            elif correction_type == "rotation":
                loss, output = compute_loss_rot(
                    image_quad,
                    target_rot,
                    model,
                    loss_fn,
                    precision,
                    config,
                    mode="eval",
                )
                targets[start_idx : start_idx + batch_size, :] = so3_log(target_rot)
            elif correction_type == "yaw":
                loss, output = compute_loss_yaw(
                    image_quad,
                    target_yaw,
                    model,
                    loss_fn,
                    precision,
                    config,
                    mode="eval",
                )
                targets[start_idx : start_idx + batch_size, :] = target_yaw
            else:
                raise ValueError(
                    "correction_type must be 'pose', 'rotation', or 'yaw'"
                )

            predictions[start_idx : start_idx + batch_size, :] = output.data
            start_idx += batch_size
            losses.update(loss.item(), batch_size)

            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % config["validate_output_interval"] == 0:
                print(
                    "Test: [{0}/{1}]\t"
                    "Time {batch_time.val:.3f} (avg: {batch_time.avg:.3f})\t"
                    "Loss {loss.val:.2E} (avg: {loss.avg:.2E})\t".format(
                        batch_idx,
                        len(valid_loader),
                        batch_time=batch_time,
                        loss=losses,
                    )
                )

    return losses.avg


def main():
    parser = argparse.ArgumentParser(
        description="Train DPC-Net (pose correction) on KITTI using SWformer-style data layout."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=os.environ.get("DPC_KITTI_DATA_ROOT", ""),
        help="Root to KITTI image sequences (e.g. $SLURM_TMPDIR/sequences_jpg)",
    )
    parser.add_argument(
        "--poses_root",
        type=str,
        default=os.environ.get("DPC_KITTI_POSES_ROOT", ""),
        help="Root to KITTI pose text files (e.g. $SLURM_TMPDIR/dataset/poses)",
    )
    parser.add_argument(
        "--vo_prior_root",
        type=str,
        default=os.environ.get("DPC_VO_PRIOR_ROOT", ""),
        help="Root to VO prior directory containing poses/<seq>.txt (libviso2 stereo trajectories in KITTI 12-value format).",
    )
    parser.add_argument(
        "--train_sequences",
        type=str,
        nargs="+",
        default=["04", "07", "08", "09", "10"],
        help="KITTI sequence IDs used for training.",
    )
    parser.add_argument(
        "--val_sequences",
        type=str,
        nargs="+",
        default=["06"],
        help="KITTI sequence IDs used for validation.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training."
    )
    parser.add_argument(
        "--num_epochs", type=int, default=30, help="Number of training epochs (paper: 30)."
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument(
        "--step_size",
        type=int,
        default=5,
        help="StepLR step size (every N epochs).",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="StepLR gamma (multiplicative LR decay).",
    )
    parser.add_argument(
        "--img_width", type=int, default=400, help="Input image width (pixels)."
    )
    parser.add_argument(
        "--img_height", type=int, default=120, help="Input image height (pixels)."
    )
    parser.add_argument(
        "--trained_models_dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints.",
    )
    parser.add_argument(
        "--kitti_test_seq",
        type=str,
        default="06",
        help="Identifier used in checkpoint filenames (e.g. val/test sequence).",
    )
    parser.add_argument(
        "--use_priors",
        action="store_true",
        help="Use VO priors as baseline (requires --vo_prior_root).",
    )
    parser.add_argument(
        "--pose_deltas",
        type=int,
        nargs="+",
        default=[3, 4, 5],
        help="Frame step(s) for training pairs (Δp). Paper uses [3,4,5]. For validation, first value or --val_pose_delta is used.",
    )
    parser.add_argument(
        "--val_pose_delta",
        type=int,
        default=4,
        help="Frame step for validation pairs. Paper uses Δp=4 for test/validation.",
    )
    args = parser.parse_args()

    if not args.data_root:
        raise ValueError(
            "data_root is empty. Set --data_root or DPC_KITTI_DATA_ROOT environment variable."
        )
    if not args.poses_root:
        raise ValueError(
            "poses_root is empty. Set --poses_root or DPC_KITTI_POSES_ROOT environment variable."
        )

    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    system_config = {
        "use_cuda": torch.cuda.is_available(),
        "validate_output_interval": 20,
        "train_output_interval": 50,
        "num_loader_workers": num_workers,
    }

    img_dims = (args.img_height, args.img_width)
    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]

    print(f"Train sequences: {args.train_sequences}")
    print(f"Val sequences:   {args.val_sequences}")
    print(f"Image dims:      {img_dims}")
    print(f"Data root:       {args.data_root}")
    print(f"Poses root:      {args.poses_root}")
    if args.use_priors:
        print(f"VO prior root:   {args.vo_prior_root}")
    else:
        print("VO priors:       disabled (identity baseline).")

    train_dataset = KITTIDPCFromSWF(
        data_root=args.data_root,
        poses_root=args.poses_root,
        sequences=args.train_sequences,
        img_dims=img_dims,
        img_mean=img_mean,
        img_std=img_std,
        vo_prior_root=args.vo_prior_root if args.use_priors else None,
        run_type="train",
        use_priors=args.use_priors,
        pose_deltas=args.pose_deltas,
    )
    val_dataset = KITTIDPCFromSWF(
        data_root=args.data_root,
        poses_root=args.poses_root,
        sequences=args.val_sequences,
        img_dims=img_dims,
        img_mean=img_mean,
        img_std=img_std,
        vo_prior_root=args.vo_prior_root if args.use_priors else None,
        run_type="validate",
        use_priors=args.use_priors,
        pose_deltas=[args.val_pose_delta],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=system_config["num_loader_workers"],
    )
    valid_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=system_config["num_loader_workers"],
    )

    correction_type = "pose"
    if correction_type == "pose":
        pose_corrector_net = DeepPoseCorrectorStereoFullPose()
        loss_fn = SE3GeodesicLoss()
        precision = train_loader.dataset.train_se3_precision
    elif correction_type == "rotation":
        pose_corrector_net = DeepPoseCorrectorMonoRotation()
        loss_fn = SO3GeodesicLoss()
        precision = train_loader.dataset.train_se3_precision[3:6, 3:6].contiguous()
    else:
        pose_corrector_net = DeepPoseCorrectorMonoYaw()
        loss_fn = nn.MSELoss()
        precision = torch.eye(1).float()

    if system_config["use_cuda"]:
        pose_corrector_net.cuda()
        loss_fn = loss_fn.cuda()

    cudnn.benchmark = True

    optimizer = optim.Adam(pose_corrector_net.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.gamma
    )

    os.makedirs(args.trained_models_dir, exist_ok=True)

    end = time.time()
    best_valid_loss = float("inf")
    train_output_interval = system_config["train_output_interval"]

    train_cfg = {
        "kitti_test_seq": args.kitti_test_seq,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "optimizer": "Adam",
        "img_type": "rgb",
        "lr": args.lr,
        "step_size": args.step_size,
        "correction_type": correction_type,
        "save_every_nth_epoch": 5,
        "gamma": args.gamma,
        "resize_factor": 1.0,
        "img_dims": [args.img_width, args.img_height],
        "trained_models_dir": args.trained_models_dir,
        "img_transforms": {"mean": img_mean, "std": img_std},
    }

    for epoch in range(1, args.num_epochs + 1):
        if epoch == 1:
            avg_valid_loss = validate(
                valid_loader,
                pose_corrector_net,
                loss_fn,
                precision,
                {"validate_output_interval": system_config["validate_output_interval"]},
                correction_type,
            )
            print(f"Initial validation loss: {avg_valid_loss:.2E}")

        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Training for seq tag {train_cfg['kitti_test_seq']}. "
            f"Starting epoch {epoch} / {train_cfg['num_epochs']}. "
            f"Learning rate: {current_lr}."
        )
        train(
            epoch,
            pose_corrector_net,
            train_loader,
            optimizer,
            loss_fn,
            precision,
            {
                "train_output_interval": train_output_interval,
            },
            correction_type,
        )

        avg_valid_loss = validate(
            valid_loader,
            pose_corrector_net,
            loss_fn,
            precision,
            {"validate_output_interval": system_config["validate_output_interval"]},
            correction_type,
        )
        print(
            "Validation completed in: {:.2f}. Current avg. validation loss: {:.2E}".format(
                time.time() - end, avg_valid_loss
            )
        )

        is_best = avg_valid_loss < best_valid_loss
        if is_best:
            best_valid_loss = avg_valid_loss
            print("New best validation loss!")

        save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": pose_corrector_net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "precision": precision,
                "train_config": train_cfg,
                "system_config": system_config,
                "kitti_config": {
                    "tm_path": "",
                    "kitti_data_file": "",
                },
                "avg_valid_loss": avg_valid_loss,
            },
            is_best=is_best,
            save_path=train_cfg["trained_models_dir"],
            epoch=epoch,
            seq=train_cfg["kitti_test_seq"],
            save_every_N=train_cfg["save_every_nth_epoch"],
        )

        scheduler.step()

        print(f"Epoch complete. Total epoch time: {time.time() - end:.2f}")
        end = time.time()


if __name__ == "__main__":
    main()
