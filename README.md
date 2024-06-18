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

Move ```LASER_HuBERT``` or ```LASER_WavLM``` to ```s3prl/s3prl/downstream/``` to add as a downstream task

Setup the path ```downstream_expert.datarc.path``` in ```config.yaml```


## Step 2: Modify runner.py
As s3prl does not provide any layerwise control for fine-tuning, we need to modify the ```s3prl/downstream/runner.py``` to freeze the layers that we don't want to train and some other details for configuration used in the downstream task.

Take the code snippet from ```runner_part_freeze_layers.py``` and put it to the ```runner.py``` inside ```_get_upstream_modules()``` function, after the model is loaded, i.e.  

```
model = Upstream(\
            ckpt = ckpt_path,
            model_config = self.args.upstream_model_config,
            refresh = upstream_refresh,
        ).to(self.args.device)
> PASTE THE SCRIPT HERE (copied from ```runner_part_freeze_layers.py)
```
## Step 3: LASER fine-tuning
### For HuBERT

Set the hyperparameters in config.yaml file or in bash file as follows:
```
SIGMA=0 ## this corresponds to window size = σ = 1
MARGIN=1.1
ALPHA=0.4
GAMMA=0.1

python3 run_downstream.py -m train -p /path_to_experiment -u hubert_base -d LASER_HuBERT -f -l -1 \
-o "config.downstream_expert.modelrc.sigma=$SIGMA,,config.downstream_expert.modelrc.gamma=$GAMMA,,config.downstream_expert.modelrc.margin=$MARGIN,,config.downstream_expert.modelrc.loss_type=$LOSS_TYPE,,config.downstream_expert.modelrc.alpha=$ALPHA"

```
