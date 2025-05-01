# overfit_head_only.py
import torch
import torch.nn.functional as F
from torch import optim
from data import DualResDataset
from torch.utils.data import DataLoader
from config import DATA_ROOT, BATCH_SIZE

# Teacher imports
from ml_depth_pro.src.depth_pro.depth_pro import (
    create_model_and_transforms as create_teacher_model,
    DEFAULT_MONODEPTH_CONFIG_DICT as T_CFG,
    DepthProConfig as TConfig,
)
# Student imports
from student_depth_model.student_depth_pro import (
    create_model_and_transforms as create_student_model,
    DEFAULT_MONODEPTH_CONFIG_DICT as S_CFG,
    DepthProConfig as SConfig,
)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) grab one batch
    ds = DualResDataset(DATA_ROOT)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    batch = next(iter(loader))
    imgs_hr = batch["image_hr"].to(device).float()  # [B,3,1536,1536]
    imgs_lr = batch["image_lr"].to(device).float()  # [B,3, 384, 384]

    # 2) build & freeze teacher
    t_cfg = TConfig(
        patch_encoder_preset=T_CFG.patch_encoder_preset,
        image_encoder_preset=T_CFG.image_encoder_preset,
        decoder_features=T_CFG.decoder_features,
        checkpoint_uri=T_CFG.checkpoint_uri,
        use_fov_head=T_CFG.use_fov_head,
        fov_encoder_preset=T_CFG.fov_encoder_preset,
    )
    teacher, _ = create_teacher_model(config=t_cfg, device=device, precision=torch.float32)
    teacher.eval()

    # 3) build student (fp32)
    s_cfg = SConfig(
        patch_encoder_preset=S_CFG.patch_encoder_preset,
        image_encoder_preset=S_CFG.image_encoder_preset,
        decoder_features=S_CFG.decoder_features,
        checkpoint_uri=None,
        use_fov_head=False,
        fov_encoder_preset=None,
    )
    student, _ = create_student_model(config=s_cfg, device=device, precision=torch.float32)
    student.train()

    # 4) freeze everything except the final “head”
    for p in student.parameters():
        p.requires_grad = False
    
    last_conv = student.decoder.convs[-1]
    last_fuse = student.decoder.fusions[-1]
    for p in last_conv.parameters(): p.requires_grad = True
    for p in last_fuse.parameters(): p.requires_grad = True
    for p in student.head.parameters(): p.requires_grad = True

    conv_b2 = student.decoder.convs[-2]
    fuse_b2 = student.decoder.fusions[-2]
    for p in conv_b2.parameters(): p.requires_grad = True
    for p in fuse_b2.parameters(): p.requires_grad = True
    


    # 5) optimizer on head only, big LR, no weight decay
    optimizer = optim.Adam([
        {"params": conv_b2.parameters(), "lr": 1e-5},
        {"params": fuse_b2.parameters(), "lr": 1e-5},
        {"params": last_conv.parameters(), "lr": 1e-4},
        {"params": last_fuse.parameters(), "lr": 1e-4},
        {"params": student.head.parameters(), "lr": 1e-3},
    ])
    print("🔍 Overfitting **head-only**, watch it plummet!")
    for step in range(100):
        optimizer.zero_grad()

        # teacher forward
        with torch.no_grad():
            depth_t, _ = teacher(imgs_hr)             # [B,1,1536,1536]

        # student forward
        depth_s, _ = student(imgs_lr)                # [B,1, 384, 384]
        depth_s = F.interpolate(
            depth_s,
            size=depth_t.shape[2:],                  # (1536,1536)
            mode="bilinear",
            align_corners=False,
        )

        # compute & backward
        loss = F.mse_loss(depth_s, depth_t)
        loss.backward()

        # diagnostics: head weight & bias grad norms
        wgrad = student.head[4].weight.grad.norm().item()
        bgrad = student.head[4].bias.grad.norm().item()
        print(f"step {step:03d}  loss={loss.item():.4f}  wgrad={wgrad:.2e}  bgrad={bgrad:.2e}")

        optimizer.step()

if __name__ == "__main__":
    main()
