"""
MagVLA 训练主脚本 - True 4-GPU DDP 版本 (version11)

关键设计（与 v10 一致）：
- torchrun --nproc_per_node=4 启动真 DDP
- 每个 GPU 各自加载模型（不重复加载）→ 节省显存
- DistributedSampler 分配数据，每个 rank 看到不同的 batch
- 前 ACCUM_STEPS-1 个 micro-step 用 model.no_sync() 跳过 all-reduce
- 最后一个 micro-step 才 backward + clip + optimizer.step()
- 只在 rank0 写 wandb / tensorboard / checkpoint
- eval 时所有 rank 并行处理 val set，再 all_reduce 聚合

v13 关键修改（2-Phase）：
- PhaseHead 添加 state_normalized + state_history (motion) 输入
- 解决 approach(0) ↔ return(2) 混淆问题
"""

import os
import sys
import time
import socket
from pathlib import Path
from datetime import datetime

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import wandb

# ── 必须在导入 config 之前设置 PYTHONPATH ─────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

import config
from dataset import MagVLADataset
from policy import MagVLAPolicy, get_optimizer

__all__ = ["train"]


def collate_fn(batch):
    """DataLoader 整理函数：对齐 batch 中每个 tensor 的形状"""
    episode_indices = [b["episode_index"] for b in batch]
    frame_indices = [b["frame_index"] for b in batch]
    images = torch.stack([b["observation.images.main"] for b in batch])  # [B, T, H, W, 3]
    states = torch.stack([b["observation.state"] for b in batch])          # [B, 4]
    state_histories = torch.stack([b["state_history"] for b in batch])    # [B, 4, 4]
    phase_gts = torch.stack([b["phase_gt"] for b in batch])              # [B]
    action_delta_chunks = torch.stack([b["action_delta_chunk"] for b in batch])  # [B, K, 4]
    task_texts = [b["task_text"] for b in batch]
    return {
        "episode_indices": episode_indices,
        "frame_indices": frame_indices,
        "observation.images.main": images,
        "observation.state": states,
        "state_history": state_histories,
        "phase_gt": phase_gts,
        "action_delta_chunk": action_delta_chunks,
        "task_text": task_texts,
    }


@torch.no_grad()
def evaluate(model: DDP, val_loader: DataLoader, rank: int, world_size: int) -> dict:
    """
    分布式正确 eval：
    - 每个 rank 处理自己那部分 val set
    - 通过 all_reduce 聚合到所有 rank
    """
    model.eval()

    local_phase_loss = 0.0
    local_chunk_loss = 0.0
    local_phase_correct = 0
    local_samples = 0

    for batch in val_loader:
        images = batch["observation.images.main"].cuda()
        states = batch["observation.state"].cuda()
        state_history = batch["state_history"].cuda()
        phase_gt = batch["phase_gt"].cuda()
        action_delta_gt = batch["action_delta_chunk"].cuda()
        task_texts = batch["task_text"]

        out = model(images, task_texts, states, phase_gt,
                    action_delta_gt=action_delta_gt, state_history=state_history)

        local_phase_loss += out["phase_loss"].item() * images.shape[0]
        local_chunk_loss += out["chunk_loss"].item() * images.shape[0]
        preds = out["phase_logits"].argmax(dim=-1)
        local_phase_correct += (preds == phase_gt).sum().item()
        local_samples += images.shape[0]

    # ── all_reduce 聚合 ──────────────────────────────────────────────
    phase_loss_tensor = torch.tensor(local_phase_loss, dtype=torch.float64, device="cuda")
    chunk_loss_tensor = torch.tensor(local_chunk_loss, dtype=torch.float64, device="cuda")
    phase_correct_tensor = torch.tensor(local_phase_correct, dtype=torch.long, device="cuda")
    samples_tensor = torch.tensor(local_samples, dtype=torch.long, device="cuda")

    dist.all_reduce(phase_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(chunk_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(phase_correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(samples_tensor, op=dist.ReduceOp.SUM)

    total_samples = samples_tensor.item()
    n = max(1, total_samples)

    result = {
        "val_phase_loss": phase_loss_tensor.item() / n,
        "val_chunk_loss": chunk_loss_tensor.item() / n,
        "val_phase_acc": phase_correct_tensor.item() / n,
    }

    model.train()
    return result


def train(
    data_dir: str | Path = config.DATA_DIR,
    output_dir: str | Path = config.PROJECT,
    resume_from: str | Path | None = None,
    keep_checkpoints: int = 1,
):
    # ── 0. DDP 初始化（由 torchrun 自动注入环境变量）────────────────────────
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://", device_id=torch.device(f"cuda:{local_rank}"))

    is_rank0 = (rank == 0)
    device = f"cuda:{local_rank}"
    output_dir = Path(output_dir)
    ckpt_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"

    if is_rank0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

    dist.barrier()

    # ── 诊断信息 ───────────────────────────────────────────────────────
    if is_rank0:
        print("=" * 70)
        print(f"MagVLA Training - True 4-GPU DDP (version11)")
        print(f"  hostname:   {socket.gethostname()}")
        print(f"  rank:       {rank} / {world_size}")
        print(f"  local_rank: {local_rank}")
        print(f"  device:     {device}")
        print(f"  data_dir:   {data_dir}")
        print(f"  output_dir: {output_dir}")
        print(f"  version:    v13-2phase-no-return")
        print(f"  BATCH_SIZE: {config.BATCH_SIZE} (per-GPU)")
        print(f"  ACCUM_STEPS: {config.ACCUM_STEPS}")
        print(f"  global_batch: {config.BATCH_SIZE} * {world_size} * {config.ACCUM_STEPS} = {config.BATCH_SIZE * world_size * config.ACCUM_STEPS}")
        print(f"  lr:         VLM(LoRA)={config.LR}  phase={config.LR_PHASE_HEAD}  act={config.LR_ACT_HEAD}")
        print(f"  MAX_STEPS:  {config.MAX_STEPS}")
        print(f"  WARMUP:     {config.WARMUP_STEPS} | PLATEAU: {config.PLATEAU_STEPS} | MIN_LR_RATIO: {config.MIN_LR_RATIO}")
        print(f"  image_downscale: {config.IMAGE_DOWNSCALE}")
        print(f"  PhaseHead:   num_phases=2 (approach vs transport)")
        print(f"  Dataset:     filtering phase==2 frames (no return)")
        print(f"  resume_from: {resume_from}")
        print("=" * 70)
    else:
        print(f"[Rank {rank}] Initializing...")

    dist.barrier()

    # ── 1. Dataset & DataLoader ─────────────────────────────────────────────
    train_ds = MagVLADataset(
        data_dir=data_dir,
        split="train",
        image_history=config.IMAGE_HISTORY,
        chunk_size=config.CHUNK_SIZE,
        normalize=True,
        image_downscale=config.IMAGE_DOWNSCALE,
    )
    val_ds = MagVLADataset(
        data_dir=data_dir,
        split="val",
        image_history=config.IMAGE_HISTORY,
        chunk_size=config.CHUNK_SIZE,
        normalize=True,
        image_downscale=config.IMAGE_DOWNSCALE,
    )

    # drop_last=True 确保每批 batch size 一致，避免 DDP rank 间 batch 大小不同
    train_sampler = DistributedSampler(
        train_ds, shuffle=True, drop_last=True,
    )
    val_sampler = DistributedSampler(
        val_ds, shuffle=False, drop_last=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        sampler=train_sampler,
        num_workers=0,   # 多卡 DDP 下 num_workers>0 容易导致 DataLoader worker 进程间竞争/死锁
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        sampler=val_sampler,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    per_rank_samples = len(train_ds) // world_size
    if is_rank0:
        print(f"  train: {len(train_ds)} samples ({per_rank_samples}/rank) | val: {len(val_ds)} samples")
        print(f"  train loader batches per epoch: {len(train_loader)}")
        print(f"  sampler epochs will be set correctly in training loop")

    dist.barrier()

    # ── 2. Policy（每个 rank 独立加载到自己的 GPU）──────────────────────────
    policy = MagVLAPolicy(
        vlm_name=config.VLM_NAME,
        chunk_size=config.CHUNK_SIZE,
        image_history=config.IMAGE_HISTORY,
        image_downscale=config.IMAGE_DOWNSCALE,
        use_lora=config.USE_LORA,
        device=device,
    )

    model = DDP(
        policy,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )

    total_steps = config.MAX_STEPS
    optimizer, scheduler = get_optimizer(policy, total_steps)

    # ── 3. Resume ───────────────────────────────────────────────────────
    start_step = 0
    if resume_from is not None:
        resume_path = Path(resume_from)
        if resume_path.exists():
            if is_rank0:
                print(f"  Loading checkpoint: {resume_path}")
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            model.module.load_state_dict(ckpt["policy_state"], strict=False)
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scheduler.load_state_dict(ckpt["scheduler_state"])
            start_step = ckpt.get("step", 0)
            if is_rank0:
                print(f"  Resumed from step {start_step}")
            del ckpt
            torch.cuda.empty_cache()
        else:
            if is_rank0:
                print(f"  WARNING: checkpoint not found: {resume_path}")

    dist.barrier()

    # ── 4. Logging（只 rank0）────────────────────────────────────────────
    writer = None
    if is_rank0:
        writer = SummaryWriter(log_dir=log_dir / config.RUN_NAME)

        if config.WANDB_KEY:
            wandb.login(key=config.WANDB_KEY)
        wandb.init(
            project=config.WANDB_PROJECT,
            name=config.RUN_NAME,
            config={
                "vlm": config.VLM_NAME,
                "batch_size_per_gpu": config.BATCH_SIZE,
                "world_size": world_size,
                "accum_steps": config.ACCUM_STEPS,
                "global_batch": config.BATCH_SIZE * world_size * config.ACCUM_STEPS,
                "lr_lora": config.LR,
                "lr_phase": config.LR_PHASE_HEAD,
                "lr_act": config.LR_ACT_HEAD,
                "warmup_steps": config.WARMUP_STEPS,
                "plateau_steps": config.PLATEAU_STEPS,
                "min_lr_ratio": config.MIN_LR_RATIO,
                "max_steps": config.MAX_STEPS,
                "image_downscale": config.IMAGE_DOWNSCALE,
                "use_lora": config.USE_LORA,
                "lora_r": config.LORA_R,
                "lora_alpha": config.LORA_ALPHA,
                "lambda_phase": config.LAMBDA_PHASE,
                "train_samples": len(train_ds),
                "val_samples": len(val_ds),
                "loss_type": "noise_prediction_mse",
                "ddp": True,
                "world_size": world_size,
            },
        )
        wandb.define_metric("train/global_step")
        wandb.define_metric("val/global_step")

    global_step = start_step
    best_val_chunk_loss = float("inf")
    best_ckpt_path = None
    epoch = 0

    # ── 5. Training Loop ──────────────────────────────────────────────────
    model.train()
    iterator = iter(train_loader)

    if is_rank0:
        print(f"\n开始训练，共 {config.MAX_STEPS} steps...\n")
        pbar = tqdm(total=config.MAX_STEPS, desc="train")

    # local_acc: micro-step counter within current accumulation cycle
    # 0 means start of a new accumulation cycle
    local_acc = 0

    while global_step < config.MAX_STEPS:
        train_sampler.set_epoch(epoch)
        epoch += 1

        for batch in train_loader:
            # ── 新一轮 accumulation cycle ───────────────────────────────────
            if local_acc == 0:
                optimizer.zero_grad(set_to_none=True)
                accum_loss = 0.0
                accum_phase_loss = 0.0
                accum_chunk_loss = 0.0
                batch_action_chunk_list = []
                batch_action_gt_list = []
                batch_grad_norms = []

            # ── 加载数据 ─────────────────────────────────────────────────
            images = batch["observation.images.main"].to(device)
            states = batch["observation.state"].to(device)
            state_history = batch["state_history"].to(device)
            phase_gt = batch["phase_gt"].to(device)
            action_delta_gt = batch["action_delta_chunk"].to(device)
            task_texts = batch["task_text"]

            # ── Forward Pass ─────────────────────────────────────────────────
            # no_sync() 必须同时包裹 forward 和 backward，才不会触发梯度同步
            is_last_accum = (local_acc == config.ACCUM_STEPS - 1)  # 当前这步之后，local_acc 会不会达到 ACCUM_STEPS-1
            if is_last_accum:
                out = model(
                    images, task_texts, states, phase_gt,
                    return_loss=True, action_delta_gt=action_delta_gt,
                    state_history=state_history,
                )
            else:
                with model.no_sync():
                    out = model(
                        images, task_texts, states, phase_gt,
                        return_loss=True, action_delta_gt=action_delta_gt,
                        state_history=state_history,
                    )

            loss = out["loss"] / config.ACCUM_STEPS
            accum_phase_loss += out["phase_loss"].detach().item() / config.ACCUM_STEPS
            accum_chunk_loss += out["chunk_loss"].detach().item() / config.ACCUM_STEPS
            accum_loss += loss.item()
            batch_action_chunk_list.append(out["action_chunk"].detach())
            batch_action_gt_list.append(action_delta_gt.detach())

            # local_acc 更新后，才正式成为"最后一个 micro-step"
            local_acc += 1
            is_last_accum = (local_acc == config.ACCUM_STEPS)

            if is_last_accum:
                # 最后一步：正常 backward（触发 DDP all-reduce 同步梯度）
                loss.backward()

                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                batch_grad_norms.append(total_norm.item())

                optimizer.step()
                scheduler.step()
                global_step += 1
                optimizer.zero_grad(set_to_none=True)
                local_acc = 0

                if is_rank0:
                    pbar.update(1)
                    pbar.set_postfix(
                        loss=f"{accum_loss:.4f}",
                        phase=f"{accum_phase_loss:.4f}",
                        chunk=f"{accum_chunk_loss:.4f}",
                    )
            else:
                # 前 N-1 步：no_sync 包裹 forward + backward，跳过 all-reduce
                loss.backward()

            # ── Logging（rank0）─────────────────────────────────────────────
            if is_rank0 and global_step > 0 and global_step % config.LOG_INTERVAL == 0 and is_last_accum:
                writer.add_scalar("train/loss", accum_loss, global_step)
                writer.add_scalar("train/phase_loss", accum_phase_loss, global_step)
                writer.add_scalar("train/chunk_loss", accum_chunk_loss, global_step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

                all_chunk = torch.cat(batch_action_chunk_list, dim=0)
                all_gt = torch.cat(batch_action_gt_list, dim=0)
                chunk_max = all_chunk.abs().max().item()
                gt_max = all_gt.abs().max().item()

                phase_pred = out["phase_logits"].argmax(dim=-1)
                phase_acc = (phase_pred == phase_gt).float().mean().item()
                avg_grad_norm = sum(batch_grad_norms) / max(1, len(batch_grad_norms))

                print(
                    f"  [Step {global_step}] "
                    f"loss={accum_loss:.4f}  "
                    f"chunk={accum_chunk_loss:.4f}  "
                    f"phase={accum_phase_loss:.4f}  "
                    f"phase_acc={phase_acc:.3f}  "
                    f"| chunk_max={chunk_max:.2f}  "
                    f"gt_max={gt_max:.2f}  "
                    f"| grad_norm={avg_grad_norm:.4f}  "
                    f"| lr={scheduler.get_last_lr()[0]:.2e}"
                )

                writer.add_scalar("train/phase_acc", phase_acc, global_step)
                writer.add_scalar("train/chunk_max", chunk_max, global_step)
                writer.add_scalar("train/grad_norm", avg_grad_norm, global_step)

                wandb.log(
                    {
                        "train/loss": accum_loss,
                        "train/phase_loss": accum_phase_loss,
                        "train/chunk_loss": accum_chunk_loss,
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/global_step": global_step,
                        "train/phase_acc": phase_acc,
                        "train/chunk_max": chunk_max,
                        "train/grad_norm": avg_grad_norm,
                    },
                    step=global_step,
                )

            # ── Eval（所有 rank 都调用 evaluate() 进行 all_reduce，rank0 打印日志）─────────────────
            if is_last_accum and global_step > 0 and global_step % config.EVAL_INTERVAL == 0:
                metrics = evaluate(model, val_loader, rank, world_size)  # 所有 rank 都要调用
                if is_rank0:
                    tqdm.write(
                        f"  [Step {global_step}] val_phase_loss={metrics['val_phase_loss']:.4f}  "
                        f"val_chunk_loss={metrics['val_chunk_loss']:.4f}  "
                        f"val_phase_acc={metrics['val_phase_acc']:.3f}"
                    )
                    writer.add_scalar("val/phase_loss", metrics["val_phase_loss"], global_step)
                    writer.add_scalar("val/chunk_loss", metrics["val_chunk_loss"], global_step)
                    writer.add_scalar("val/phase_acc", metrics["val_phase_acc"], global_step)
                    wandb.log(
                        {
                            "val/phase_loss": metrics["val_phase_loss"],
                            "val/chunk_loss": metrics["val_chunk_loss"],
                            "val/phase_acc": metrics["val_phase_acc"],
                            "val/global_step": global_step,
                        },
                        step=global_step,
                    )

                    # 只在 val chunk loss 更低时保存 best 模型（只保留 best + final）
                    current_val_chunk_loss = metrics["val_chunk_loss"]
                    if current_val_chunk_loss < best_val_chunk_loss:
                        old_best = best_ckpt_path
                        best_val_chunk_loss = current_val_chunk_loss
                        best_ckpt_path = ckpt_dir / f"step_{global_step}_best.pt"
                        torch.save(
                            {
                                "step": global_step,
                                "policy_state": model.module.state_dict(),
                                "optimizer_state": optimizer.state_dict(),
                                "scheduler_state": scheduler.state_dict(),
                                "val_chunk_loss": best_val_chunk_loss,
                                "val_phase_loss": metrics["val_phase_loss"],
                                "val_phase_acc": metrics["val_phase_acc"],
                                "config": {
                                    "vlm_name": config.VLM_NAME,
                                    "chunk_size": config.CHUNK_SIZE,
                                    "image_history": config.IMAGE_HISTORY,
                                    "use_lora": config.USE_LORA,
                                },
                            },
                            best_ckpt_path,
                        )
                        tqdm.write(
                            f"  [Best model saved] step={global_step}  "
                            f"val_chunk_loss={best_val_chunk_loss:.6f}  "
                            f"(prev best: {old_best.name if old_best else 'N/A'})"
                        )
                        # 删除旧的 best（只保留最新的 best）
                        if old_best is not None and old_best.exists():
                            old_best.unlink()
                            tqdm.write(f"  [Old best deleted] {old_best.name}")

            if global_step >= config.MAX_STEPS:
                break

        dist.barrier()

    if is_rank0:
        pbar.close()

        # ── Final save ────────────────────────────────────────────────────
        final_ckpt = ckpt_dir / "final.pt"
        torch.save(
            {
                "step": global_step,
                "policy_state": model.module.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_chunk_loss": best_val_chunk_loss,
                "best_ckpt_step": global_step if best_ckpt_path is None else int(best_ckpt_path.stem.split("_")[1]),
            },
            final_ckpt,
        )
        print(f"\n训练完成！最终 checkpoint: {final_ckpt}")
        print(f"Best val_chunk_loss: {best_val_chunk_loss:.6f}  (best ckpt: {best_ckpt_path.name if best_ckpt_path else 'N/A'})")
        writer.close()
        wandb.finish()

    dist.destroy_process_group()
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--max_steps", type=int, default=None, help="Override MAX_STEPS (for smoke test)")
    parser.add_argument(
        "--keep_checkpoints",
        type=int,
        default=3,
        help="Number of recent checkpoints to keep. Default=3",
    )
    args = parser.parse_args()

    if args.max_steps is not None:
        import config as _cfg
        _cfg.MAX_STEPS = args.max_steps
        print(f"[Smoke test mode] MAX_STEPS overridden to {args.max_steps}")

    train(resume_from=args.resume, keep_checkpoints=args.keep_checkpoints)
