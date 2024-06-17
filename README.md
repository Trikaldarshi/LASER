# LASER

Interspeech 2024 - **[LASER: Learning by Aligning Self-supervised Representations of Speech for Improving Content-related Tasks](https://arxiv.org/abs/2406.09153)**.

This repo consist the implementation of LASER fine-tuning proposed in the above paper using s3prl toolkit.

## Install s3prl toolkit
```
conda create -n s3prl python=3.8 \
conda activate s3prl \
git clone https://github.com/s3prl/s3prl.git \
cd s3prl \
pip install -e ".[all]"
```

## Step 1: Add downstream task to s3prl toolkit
