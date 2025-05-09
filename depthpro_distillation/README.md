##

First, create an env and install  

```bash
conda create -n depth_distillation -y python=3.9
conda activate depth_distillation
pip install -r requirements.txt 
```
Then, to get the teacher pretrained model go ito the ml_depth_pro dir and run 
```bash
source get_pretrained_models.sh   # Files will be downloaded to `checkpoints` directory.
```

To get the dataset I used (COCO unlabeled objects 2017), go to: 
http://images.cocodataset.org/zips/unlabeled2017.zip

Then place this in /mldata or wherever you have space and update the config file. 

To run the training script (using DDP with 2 GPUs), type in 
```bash
torchrun --nproc_per_node=2 train.py
```

Then, launch a tensorboard session in a new terminal with 

```bash 
pip install tensorboard
tensorboard --logdir runs/depth_distill --port 6007
```

Optionally, to do a smoke test, add the arg
```bash
torchrun --nproc_per_node=2 train.py --smoke-test
```

To use tmux to keep training going at all times, use
```bash
tmux new -s myrun
Ctrl-b then %
run train command
Ctrl-b then o
run logging command
Ctrl-b then d #to detach
tmux attach -t myrun #to reattach
```

WARNING: this needs revision! the student depth map only produces a black image. Maybe experiment with warming up the feature distillation weight, experiment with lower LR (for the projector parameters). Look at depth ranges to see if it's producing anything. I'd also recommend looking through the student_encoder, student_vit_factory, etc to get a feel for how the source code was modified (and if you see any bugs)
