import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import optim
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import argparse

from data import get_data_loaders
from config import (
    DATA_ROOT,
    BATCH_SIZE,
    EPOCHS,
    LR,
    WEIGHT_DECAY,
    TEACHER_CHECKPOINT,
    FEATURE_LAYERS,
)
from loss import DistillationLoss

# Which layers to print grads for
#monitored = FEATURE_LAYERS + ["head"]
monitored = ["head"]


# Teacher factory
from ml_depth_pro.src.depth_pro.depth_pro import (
    create_model_and_transforms as create_teacher_model,
    DEFAULT_MONODEPTH_CONFIG_DICT as T_CFG,
    DepthProConfig as TConfig,
)
# Student factory
from student_depth_model.student_depth_pro import (
    create_model_and_transforms as create_student_model,
    DEFAULT_MONODEPTH_CONFIG_DICT as S_CFG,
    DepthProConfig as SConfig,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true",
                        help="run exactly one batch/epoch then exit")
    args = parser.parse_args()

    # 0) DDP init
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # 1) Data
    train_loader, val_loader, _ = get_data_loaders(DATA_ROOT, BATCH_SIZE)
    num_batches = len(train_loader)
    print(f"Number of training batches per epoch: {num_batches}")

    # 2) Teacher (frozen, fp16)
    t_cfg = TConfig(
        patch_encoder_preset=T_CFG.patch_encoder_preset,
        image_encoder_preset=T_CFG.image_encoder_preset,
        decoder_features=T_CFG.decoder_features,
        checkpoint_uri=TEACHER_CHECKPOINT,
        use_fov_head=T_CFG.use_fov_head,
        fov_encoder_preset=T_CFG.fov_encoder_preset,
    )
    teacher, _ = create_teacher_model(
        config=t_cfg,
        device=device,
        precision=torch.float16,
    )
    teacher.eval()

    # 3) Student (trainable, fp32)
    s_cfg = SConfig(
        patch_encoder_preset=S_CFG.patch_encoder_preset,
        image_encoder_preset=S_CFG.image_encoder_preset,
        decoder_features=S_CFG.decoder_features,
        checkpoint_uri=None,
        use_fov_head=False,
        fov_encoder_preset=None,
    )
    student, _ = create_student_model(
        config=s_cfg,
        device=device,
        precision=torch.float32,
    )
    student = DDP(
        student,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )

    for name, param in student.module.named_parameters():
       if "head" not in name:
           param.requires_grad = False

    student.train()

    # 4) Criterion + optimizer + scheduler
    student_dims = [s_cfg.decoder_features] + list(student.module.encoder.dims_encoder)
    teacher_dims = [t_cfg.decoder_features] + list(teacher.encoder.dims_encoder)
    criterion = DistillationLoss(student_dims, teacher_dims).to(device)

    for conv in criterion.projectors.modules():
        if isinstance(conv, nn.Conv2d) and conv.kernel_size == (1,1):
            nn.init.eye_(conv.weight.view(conv.out_channels, conv.in_channels))
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)
    
    #optimizer = optim.Adam(
    #    list(student.parameters()) + list(criterion.parameters()),
    #    lr=LR,
    #    weight_decay=WEIGHT_DECAY,
    #)
                
    head_params = filter(lambda p: p.requires_grad, student.parameters())
    optimizer = optim.SGD(
        head_params,
        lr=LR,               # you may need to up the LR (e.g. 1e-2)
        momentum=0.9,
        weight_decay=WEIGHT_DECAY,
    )            
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    warmup_epochs = 5

    # linear warm-up: start at 0.1% of LR, go to full LR in warmup_epochs
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,    # warm-up starts at LR * 0.001
        end_factor=1.0,       # warms up to the full LR
        total_iters=warmup_epochs
    )

    # cosine decay for the rest of training
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS - warmup_epochs
    )

    # glue them together
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )

    # 5) TensorBoard + AMP
    writer = SummaryWriter(log_dir="runs/depth_distill")
    scaler = torch.cuda.amp.GradScaler()

    # 6) Training
    for epoch in range(EPOCHS):
        train_loader.sampler.set_epoch(epoch)
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):

            if args.smoke_test:
                print("Smoke test: 1 train batch completed successfully.")
                break

            imgs_hr = batch['image_hr'].to(device)
            imgs_lr = batch['image_lr'].to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                # teacher fp16 forward
                with torch.no_grad():
                    hr16 = imgs_hr.half()   
                    teacher_feats = teacher.encoder(hr16)
                    depth_t, _ = teacher(hr16)
                # student forward
                student_feats = student.module.encoder(imgs_lr)
                depth_s, _ = student(imgs_lr)
                loss, logs = criterion(depth_s, depth_t, student_feats, teacher_feats)
            
            if batch_idx % 200 == 0:
                print("depth_s range:", depth_s.min().item(), depth_s.max().item())
                print("depth_t range:", depth_t.min().item(), depth_t.max().item())

            # backprop w/ scaling
            scaler.scale(loss).backward()

            #for name, p in student.named_parameters():
            #    if p.grad is not None and 'upsample_latent1' in name:
            #        print(f"{name} grad norm:", p.grad.norm().item())
            #    break
            # gradient clipping
            scaler.unscale_(optimizer)
            total_norm = clip_grad_norm_(
                student.parameters(), max_norm=1.0
            )
            writer.add_scalar("GradNorm/clipped", total_norm, epoch * len(train_loader) + batch_idx)

            # monitored grads
            print(f"–– monitored gradients (epoch {epoch}, batch {batch_idx}) ––")
            for name, p in student.module.named_parameters():
                if p.grad is None: continue
                if any(m in name for m in monitored):
                    print(f"{name:50s} | grad_mean = {p.grad.abs().mean().item():.2e}")
            print("---------------------------")

            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

            # TensorBoard scalars
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar("Loss/total", loss.item(), global_step)
            writer.add_scalar("Loss/depth", logs['depth_loss'], global_step)
            #writer.add_scalar("Loss/feat_distill", logs['feat_loss'], global_step)
            writer.add_scalar("LR/batch", optimizer.param_groups[0]['lr'], global_step)

            # Image logging every 500 batches
            if batch_idx % 500 == 0 and batch_idx > 0:
                from torchvision.utils import make_grid
                writer.add_image(
                    "Input/low_res",
                    make_grid(imgs_lr[:4], nrow=2, normalize=True, scale_each=True),
                    global_step,
                )
                writer.add_image(
                    "Depth/student",
                    make_grid(depth_s[:4].detach(), nrow=2, normalize=True),
                    global_step,
                )
                writer.add_image(
                    "Depth/teacher",
                    make_grid(depth_t[:4].detach(), nrow=2, normalize=True),
                    global_step,
                )

        # epoch-end logging + scheduler step
        if local_rank == 0:
            student.module.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    if args.smoke_test:
                        print("Smoke test: 1 val batch completed successfully.")
                        break
                    imgs_hr = batch['image_hr'].to(device)
                    imgs_lr = batch['image_lr'].to(device)

                    # teacher (fp16) forward
                    teacher_feats = teacher.encoder(imgs_hr)
                    depth_t, _ = teacher(imgs_hr)

                    # student (fp32) forward
                    student_feats = student.module.encoder(imgs_lr)
                    depth_s, _ = student(imgs_lr)

                    loss, _ = criterion(depth_s, depth_t, student_feats, teacher_feats)
                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            writer.add_scalar("Loss/val", avg_val_loss, epoch)
            print(f"Epoch {epoch:02d} — Val Loss: {avg_val_loss:.4f}")

            # back to train mode
            student.module.train()
            avg = epoch_loss / len(train_loader)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:02d}/{EPOCHS} — Loss: {avg:.4f} — LR: {current_lr:.2e}")
            writer.add_scalar("Epoch/loss", avg, epoch)
            writer.add_scalar("Epoch/lr", current_lr, epoch)
            if epoch % 5 == 0:
                ckpt = f"checkpoints/student_epoch{epoch}.pt"
                torch.save(student.module.state_dict(), ckpt)
                print(f"Saved checkpoint: {ckpt}")
        scheduler.step()

    # 7) Cleanup
    writer.close()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()