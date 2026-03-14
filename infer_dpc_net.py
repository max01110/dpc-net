#!/usr/bin/env python3
"""
Run DPC-Net inference on KITTI-style sequences and export KITTI trajectory files.

This script is designed for the SWformer-style data layout:
  - images: <data_root>/<seq>/image_0/<frame>.jpg|png
  - VO priors (absolute poses): <vo_prior_root>/poses/<seq>.txt
  - output trajectories: <output_dir>/pred_poses/<seq>.txt
"""

import argparse
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from PIL import Image

from lie_algebra import rpy_to_so3, se3_exp, so3_exp, so3_to_rpy
from net import (
    DeepPoseCorrectorMonoRotation,
    DeepPoseCorrectorMonoYaw,
    DeepPoseCorrectorStereoFullPose,
)


def _load_kitti_poses(poses_file: str) -> List[np.ndarray]:
    poses: List[np.ndarray] = []
    with open(poses_file, "r") as f:
        for line in f:
            vals = [float(x) for x in line.strip().split()]
            if len(vals) != 12:
                continue
            T = np.eye(4, dtype=np.float64)
            T[:3, :4] = np.array(vals, dtype=np.float64).reshape(3, 4)
            poses.append(T)
    return poses


def _save_kitti_poses(poses: List[np.ndarray], output_file: str) -> None:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        for T in poses:
            vals = T[:3, :].reshape(-1)
            f.write(" ".join(f"{v:.6e}" for v in vals) + "\n")


def _resolve_vo_pose_file(vo_prior_root: str, seq: str) -> str:
    candidates = [
        os.path.join(vo_prior_root, "poses", f"{seq}.txt"),
        os.path.join(vo_prior_root, f"{seq}.txt"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c

    npy_candidate = os.path.join(vo_prior_root, f"{seq}.npy")
    if os.path.isfile(npy_candidate):
        arr = np.load(npy_candidate)
        if arr.ndim == 3 and arr.shape[1:] == (4, 4):
            tmp_txt = os.path.join(vo_prior_root, f".tmp_abs_{seq}.txt")
            _save_kitti_poses([arr[i].astype(np.float64) for i in range(arr.shape[0])], tmp_txt)
            return tmp_txt

    raise FileNotFoundError(
        f"Could not find absolute VO poses for sequence {seq}. "
        f"Expected one of: {candidates}"
    )


def _load_image_paths(seq_img_dir: str) -> List[str]:
    if not os.path.isdir(seq_img_dir):
        raise FileNotFoundError(f"Image directory not found: {seq_img_dir}")
    img_files = [
        os.path.join(seq_img_dir, p)
        for p in os.listdir(seq_img_dir)
        if p.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    img_files.sort()
    return img_files


class SequenceInferenceDataset(Dataset):
    def __init__(
        self,
        image_paths: List[str],
        transform,
        pose_delta: int,
        max_frames: int,
    ) -> None:
        super().__init__()
        if pose_delta < 1:
            raise ValueError("pose_delta must be >= 1")

        n = len(image_paths)
        if max_frames is not None and max_frames > 0:
            n = min(n, max_frames)
        self.image_paths = image_paths[:n]
        self.pose_delta = pose_delta
        self.transform = transform

        # Match original DPC evaluation style: one correction every pose_delta.
        self.start_indices = list(range(0, len(self.image_paths) - pose_delta, pose_delta))

    def __len__(self) -> int:
        return len(self.start_indices)

    def _read_image(self, idx: int) -> torch.Tensor:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img)

    def __getitem__(self, index: int):
        i = self.start_indices[index]
        j = i + self.pose_delta
        img0 = self._read_image(i)
        img1 = self._read_image(j)
        return img0, img1, i


def _model_from_correction_type(corr_type: str):
    if corr_type == "pose":
        return DeepPoseCorrectorStereoFullPose()
    if corr_type == "rotation":
        return DeepPoseCorrectorMonoRotation()
    if corr_type == "yaw":
        return DeepPoseCorrectorMonoYaw()
    raise ValueError(f"Unsupported correction type: {corr_type}")


def _correct_relative_pose(T_rel_est: np.ndarray, pred: np.ndarray, corr_type: str) -> np.ndarray:
    T_rel_corr = T_rel_est.copy()

    if corr_type == "pose":
        xi = torch.from_numpy(pred.astype(np.float32)).unsqueeze(0)
        T_corr = se3_exp(xi)[0].cpu().numpy().astype(np.float64)
        return T_corr @ T_rel_est

    if corr_type == "rotation":
        phi = torch.from_numpy(pred.astype(np.float32)).unsqueeze(0)
        R_corr = so3_exp(phi)[0].cpu().numpy().astype(np.float64)
        T_rel_corr[:3, :3] = R_corr @ T_rel_est[:3, :3]
        return T_rel_corr

    if corr_type == "yaw":
        R_est = torch.from_numpy(T_rel_est[:3, :3].astype(np.float32)).unsqueeze(0)
        rpy = so3_to_rpy(R_est)
        rpy[0, 1] += float(pred[0])  # Vehicle yaw ~ camera pitch in this convention.
        R_corr = rpy_to_so3(rpy)[0].cpu().numpy().astype(np.float64)
        T_rel_corr[:3, :3] = R_corr
        return T_rel_corr

    raise ValueError(f"Unsupported correction type: {corr_type}")


def _build_corrected_trajectory(
    Twv_est: List[np.ndarray],
    predictions: Dict[int, np.ndarray],
    start_indices: List[int],
    pose_delta: int,
    corr_type: str,
) -> List[np.ndarray]:
    n = len(Twv_est)
    if n == 0:
        return []

    Twv_corr: List[np.ndarray] = [np.eye(4, dtype=np.float64)]

    for i in start_indices:
        while len(Twv_corr) - 1 < i:
            j = len(Twv_corr) - 1
            T_single = np.linalg.inv(Twv_est[j]) @ Twv_est[j + 1]
            Twv_corr.append(Twv_corr[j] @ T_single)

        T_rel_est = np.linalg.inv(Twv_est[i]) @ Twv_est[i + pose_delta]
        T_rel_corr = _correct_relative_pose(T_rel_est, predictions[i], corr_type)
        Twv_corr.append(Twv_corr[i] @ T_rel_corr)

    while len(Twv_corr) < n:
        j = len(Twv_corr) - 1
        T_single = np.linalg.inv(Twv_est[j]) @ Twv_est[j + 1]
        Twv_corr.append(Twv_corr[j] @ T_single)

    return Twv_corr


def _to_resize_hw(img_dims_from_ckpt: List[int], override_hw: Tuple[int, int]) -> Tuple[int, int]:
    if override_hw is not None:
        return override_hw
    if not img_dims_from_ckpt or len(img_dims_from_ckpt) != 2:
        return (120, 400)
    # Training saved [width, height] in train_config.
    return (int(img_dims_from_ckpt[1]), int(img_dims_from_ckpt[0]))


def main():
    parser = argparse.ArgumentParser(description="Run DPC-Net inference and export KITTI trajectories.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth.tar checkpoint file")
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/michetma/Desktop/TRAIL/SWformer-VO-Fork/data/sequences_jpg",
        help="KITTI sequence image root: <data_root>/<seq>/image_0",
    )
    parser.add_argument(
        "--vo_prior_root",
        type=str,
        default="/home/michetma/Desktop/TRAIL/SWformer-VO-Fork/data/vo_priors_libviso2_stereo",
        help="VO prior root containing poses/<seq>.txt absolute trajectories",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory (pred_poses/ will be created)")
    parser.add_argument("--sequences", nargs="+", default=["05"], help="Sequence IDs, e.g. 00 05 10")
    parser.add_argument("--pose_delta", type=int, default=1, help="Correction interval (default: 1)")
    parser.add_argument("--batch_size", type=int, default=64, help="Inference batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--max_frames", type=int, default=0, help="Limit frames per sequence (0 = all)")
    parser.add_argument("--corr", type=str, default="", help="Override correction type: pose|rotation|yaw")
    parser.add_argument("--img_height", type=int, default=0, help="Optional image height override")
    parser.add_argument("--img_width", type=int, default=0, help="Optional image width override")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    train_cfg = checkpoint.get("train_config", {})
    corr_type = args.corr if args.corr else train_cfg.get("correction_type", "pose")

    img_mean = train_cfg.get("img_transforms", {}).get("mean", [0.485, 0.456, 0.406])
    img_std = train_cfg.get("img_transforms", {}).get("std", [0.229, 0.224, 0.225])
    override_hw = None
    if args.img_height > 0 and args.img_width > 0:
        override_hw = (args.img_height, args.img_width)
    resize_hw = _to_resize_hw(train_cfg.get("img_dims", [400, 120]), override_hw)

    transform_img = transforms.Compose(
        [
            transforms.Resize(resize_hw),
            transforms.ToTensor(),
            transforms.Normalize(mean=img_mean, std=img_std),
        ]
    )

    use_cuda = torch.cuda.is_available() and (not args.cpu)
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Device: {device}")
    print(f"Correction type: {corr_type}")
    print(f"Resize (H,W): {resize_hw}")
    print(f"Pose delta: {args.pose_delta}")

    model = _model_from_correction_type(corr_type)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    model.eval()

    output_pred_dir = Path(args.output_dir) / "pred_poses"
    output_pred_dir.mkdir(parents=True, exist_ok=True)

    max_frames = args.max_frames if args.max_frames > 0 else None

    with torch.no_grad():
        for seq in args.sequences:
            seq = f"{int(seq):02d}" if seq.isdigit() else seq
            t0 = time.time()
            print(f"\n=== Sequence {seq} ===")

            seq_img_dir = os.path.join(args.data_root, seq, "image_0")
            image_paths = _load_image_paths(seq_img_dir)
            vo_pose_file = _resolve_vo_pose_file(args.vo_prior_root, seq)
            Twv_est = _load_kitti_poses(vo_pose_file)

            n = min(len(image_paths), len(Twv_est))
            if max_frames is not None:
                n = min(n, max_frames)
            if n <= args.pose_delta:
                print(f"Skipping {seq}: insufficient frames (n={n}, pose_delta={args.pose_delta})")
                continue

            dataset = SequenceInferenceDataset(
                image_paths=image_paths[:n],
                transform=transform_img,
                pose_delta=args.pose_delta,
                max_frames=None,
            )
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=use_cuda,
            )

            pred_by_start: Dict[int, np.ndarray] = {}
            for img0, img1, start_idx in loader:
                img0 = img0.to(device, non_blocking=True)
                img1 = img1.to(device, non_blocking=True)

                if corr_type == "pose":
                    # Original DPC pose network expects two 6-channel stereo tensors.
                    stereo_1 = torch.cat((img0, img0), dim=1)
                    stereo_2 = torch.cat((img1, img1), dim=1)
                    preds = model(stereo_1, stereo_2)
                else:
                    preds = model(img0, img1)

                preds_np = preds.detach().cpu().numpy()
                for i, p in zip(start_idx.tolist(), preds_np):
                    pred_by_start[int(i)] = p

            missing = [i for i in dataset.start_indices if i not in pred_by_start]
            if missing:
                raise RuntimeError(
                    f"Missing predictions for sequence {seq}. Missing starts: {missing[:10]}"
                )

            Twv_corr = _build_corrected_trajectory(
                Twv_est=Twv_est[:n],
                predictions=pred_by_start,
                start_indices=dataset.start_indices,
                pose_delta=args.pose_delta,
                corr_type=corr_type,
            )

            output_file = output_pred_dir / f"{seq}.txt"
            _save_kitti_poses(Twv_corr, str(output_file))
            dt = time.time() - t0
            print(
                f"Wrote {output_file} | frames={len(Twv_corr)} "
                f"| preds={len(dataset.start_indices)} | time={dt:.1f}s"
            )

    print(f"\nDone. Trajectories saved to: {output_pred_dir}")


if __name__ == "__main__":
    main()
