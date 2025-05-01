# config.py

# Training
EPOCHS       = 50
BATCH_SIZE   = 1
LR           = 1e-2
WEIGHT_DECAY = 1e-2

STUDENT_IMG_SIZE = 384               # student network resolution
TEACHER_IMG_SIZE = 1536 



# Paths
TEACHER_CHECKPOINT = "ml_depth_pro/checkpoints/depth_pro.pt"
DATA_ROOT          = "/mldata/coco/unlabeled2017/"

# going to need to change this 
# Distillation settings
DISTILL_WEIGHT       = 1.0           # relative weight for feature distillation
DEPTH_LOSS_WEIGHT    = 1.0           # weight for supervised depth loss
FEATURE_LAYERS       = [             # which student / teacher features to compare
    "latent0", "latent1", "x0", "x1", "x2"
]
FEATURE_SCALES       = {             # spatial scales of student feature maps
    "latent0": 384,
    "latent1": 192,
    "x0":      96,
    "x1":      48,
    "x2":      24,
}

