# utils.py
import os, sys, math, torch, torch.distributed as dist, torch.nn.functional as F

def setup_ddp():
    """Initialize NCCL process group & set CUDA device from LOCAL_RANK."""
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

# two separate buffers for student / teacher
_feature_store = {
    "student": [],
    "teacher": [],
}

def clear_feature_store():
    """Call before each forward to reset both student & teacher feature lists."""
    _feature_store["student"].clear()
    _feature_store["teacher"].clear()

def hook_feature(prefix):
    """
    Returns a hook fn that appends the module output to
    _feature_store[prefix], in the order the blocks are hooked.
    """
    def _hook(module, input, output):
        _feature_store[prefix].append(output)
    return _hook

def register_student_hooks(student_model):
    """
    Hooks student_model.encoder.patch_encoder.blocks at the indices
    in student_model.encoder.hook_block_ids, storing outputs in
    _feature_store['student'].
    """
    for idx in student_model.encoder.hook_block_ids:
        student_model.encoder.patch_encoder.blocks[idx]\
            .register_forward_hook(hook_feature("student"))

def register_teacher_hooks(teacher_model):
    """
    Same as above but for teacher_model.
    """
    for idx in teacher_model.encoder.hook_block_ids:
        teacher_model.encoder.patch_encoder.blocks[idx]\
            .register_forward_hook(hook_feature("teacher"))

def get_student_features():
    """Returns the list of student feature tensors, in hook order."""
    return _feature_store["student"]

def get_teacher_features():
    """Returns the list of teacher feature tensors, in hook order."""
    return _feature_store["teacher"]

def resize_feat(tensor, target_size):
    """
    Resize a feature map [B, C, H, W] (or list thereof) to [B, C, target_size, target_size].
    """
    return F.interpolate(tensor, size=(target_size, target_size),
                         mode="bilinear", align_corners=False)  