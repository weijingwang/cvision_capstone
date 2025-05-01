import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def validate(teacher, student, val_loader, criterion, device):
    """
    Run validation: compute distillation loss on the validation set.
    Returns the average loss.
    """
    teacher.eval()
    student.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            #imgs = batch["image"].to(device)

            # Clear stored features
            student.module.student_feats.clear()
            teacher.module.module_feats.clear()  # if your teacher hooks use this dict

            # Teacher forward (frozen)
            imgs_hr = batch["image_hr"].to(device)
            imgs_lr = batch["image_lr"].to(device)
            depth_t = teacher(imgs_hr)
            depth_s = student(imgs_lr)

            # Compute distillation loss
            loss = criterion(depth_s, depth_t,
                             student.module.student_feats,
                             teacher.module.module_feats)
            total_loss += loss.item()

    return total_loss / len(val_loader)


def evaluate_depth_metrics(teacher, student, test_loader, device):
    """
    Compute RMSE, MAE, and delta<1.25 accuracy between student and teacher
    on a test set.
    """
    teacher.eval()
    student.eval()

    rmse_sum = mae_sum = delta1_sum = 0.0
    count = 0

    with torch.no_grad():
        for batch in test_loader:
            #imgs = batch["image"].to(device)
            imgs_hr = batch["image_hr"].to(device)
            imgs_lr = batch["image_lr"].to(device)
            depth_t = teacher(imgs_hr)
            depth_s = student(imgs_lr)

            # Match resolutions if needed
            if depth_s.shape != depth_t.shape:
                depth_s = F.interpolate(
                    depth_s,
                    size=depth_t.shape[2:],
                    mode="bilinear",
                    align_corners=False
                )

            mse = F.mse_loss(depth_s, depth_t, reduction="mean")
            rmse = torch.sqrt(mse)
            mae = F.l1_loss(depth_s, depth_t, reduction="mean")

            # fraction of pixels where prediction is within 25% of teacher
            ratio = torch.max(depth_s / depth_t, depth_t / depth_s)
            delta1 = (ratio < 1.25).float().mean()

            rmse_sum += rmse.item()
            mae_sum += mae.item()
            delta1_sum += delta1.item()
            count += 1

    return rmse_sum / count, mae_sum / count, delta1_sum / count


def visualize_depth_comparison(teacher, student, sample_images, device, save_dir="results"):
    """
    Visualize teacher vs student depth maps and error heatmaps for a few example images.
    sample_images: a batch of tensors with shape [B, C, H, W]
    """
    os.makedirs(save_dir, exist_ok=True)
    teacher.eval()
    student.eval()

    with torch.no_grad():
        for i, img in enumerate(sample_images):
            img_t = img.to(device).unsqueeze(0)  # [1, C, H, W]
            dt = teacher(img_t)
            ds = student(img_t)

            if ds.shape != dt.shape:
                ds = F.interpolate(ds, size=dt.shape[2:], mode="bilinear", align_corners=False)

            # to numpy
            dt_np = dt.squeeze().cpu().numpy()
            ds_np = ds.squeeze().cpu().numpy()
            im_np = img.permute(1, 2, 0).cpu().numpy()

            # side-by-side
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.imshow(im_np)
            plt.title("Input")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(dt_np, cmap="plasma")
            plt.title("Teacher")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(ds_np, cmap="plasma")
            plt.title("Student")
            plt.axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"compare_{i}.png"))
            plt.close()

            # difference heatmap
            diff = abs(dt_np - ds_np)
            plt.figure(figsize=(6, 4))
            plt.imshow(diff, cmap="hot")
            plt.title("|Teacher - Student|")
            plt.axis("off")
            plt.colorbar()
            plt.savefig(os.path.join(save_dir, f"diff_{i}.png"))
            plt.close()
