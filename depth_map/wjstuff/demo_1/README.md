# First demo install tutorial

```bash
cd checkpoints

./download_ckpts.sh

cd ..
```

## Usage

### 가상환경 준비 

```bash 
conda create -n dam2 python=3.11 -y

conda activate dam2

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121

```


and pip some other stuff...


### run
```
python sound4.py

```

## Credits 
Thanks to khw11044 for tutorial on depth anything webcam
https://github.com/khw11044/Depth-Anything-V2-streaming

And marmik_ch19 for temporary command line fix for pytorch error on mac
https://www.reddit.com/r/pytorch/comments/1c3kwwg/how_do_i_fix_the_mps_notimplemented_error_for_m1/


