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

Move ```LASER_finetuning``` to ```s3prl/s3prl/downstream/``` to add as a downstream task

In ```LASER_finetuning/dataset.py``` keep ```speed_factors = [0.9, 1.0, 1.1]``` for HuBERT **OR** ```speed_factors =[[0.80, 0.85, 0.9, 1.0, 1.1, 1.15, 1.20]]``` for WavLM

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
> PASTE THE SCRIPT HERE (copied from ```runner_part_freeze.py)
```
## Step 3: LASER fine-tuning
### For HuBERT

Set the hyperparameters in config.yaml file or in bash file as follows:
```
SIGMA=0 ## this corresponds to window size = σ = 1
MARGIN=1.1
ALPHA=0.4
GAMMA=0.1

python3 run_downstream.py -m train -p /path_to_laser_experiment -u hubert_base -d LASER_HuBERT -f -l -1 \
-o "config.downstream_expert.modelrc.sigma=$SIGMA,,config.downstream_expert.modelrc.gamma=$GAMMA,,config.downstream_expert.modelrc.margin=$MARGIN,,config.downstream_expert.modelrc.loss_type=$LOSS_TYPE,,config.downstream_expert.modelrc.alpha=$ALPHA"

```
### For WavLM

Set the hyperparameters in config.yaml file or in bash file as follows:
```
SIGMA=0 ## this corresponds to window size = σ = 1
MARGIN=1
ALPHA=0.15
GAMMA=0.1

python3 run_downstream.py -m train -p /path_to_laser_experiment -u wavlm_base -d LASER_WavLM -f -l -1 \
-o "config.downstream_expert.modelrc.sigma=$SIGMA,,config.downstream_expert.modelrc.gamma=$GAMMA,,config.downstream_expert.modelrc.margin=$MARGIN,,config.downstream_expert.modelrc.loss_type=$LOSS_TYPE,,config.downstream_expert.modelrc.alpha=$ALPHA"

```
## Step 4: Evaluate the SCORE finetuned model on QbE, ASR, and PR for SUPERB benchmark
Download the needed data, set data paths etc for the respective tasks. More info and hyperparameter values are available at [S3PRL/SUPERB](https://github.com/s3prl/s3prl/blob/main/s3prl/downstream/docs/superb.md)

### For QbE on test set with last layer

For HuBERT
```
python3 run_downstream.py -m evaluate -t "test" -u hubert_base -l -1 -d quesst14_dtw -p /path_to_qbe_experiment \
-o "config.downstream_expert.datarc.test_base_path=/path_to_laser_experiment/states-3600.ckpt,,config.runner.freeze_layers=False,,config.runner.baseline=custom"
```
For WavLM
```
python3 run_downstream.py -m evaluate -t "test" -u wavlm_base -l -1 -d quesst14_dtw -p /path_to_qbe_experiment \
-o "config.downstream_expert.datarc.test_base_path=/path_to_laser_experiment/states-3600.ckpt,,config.runner.freeze_layers=False,,config.runner.baseline=custom"
```
Then move to scoring directory (PATH_SCORING = /yourpath/quesst14Database/scoring/) and run score script
```
cd $PATH_SCORING
bash ./score-TWV-Cnxe.sh /path_to_qbe_experiment groundtruth_quesst14_eval -10

```
Note: to reproduce for superb benchmark, keep ```config.runner.freeze_layers=False,,config.runner.baseline=superb ```

### For PR
Note: Make sure lr is 5.0e − 4 \
Training:
```
python3 run_downstream.py -p /path_to_pr_experiment -m train -u hubert_base -d ctc -c downstream/ctc/libriphone.yaml \
-o "config.downstream_expert.datarc.test_base_path=path_to_laser_experiment/states-3600.ckpt,,config.runner.freeze_layers=False,,config.runner.baseline=custom"
```
OR
```
python3 run_downstream.py -p /path_to_pr_experiment -m train -u wavlm_base -d ctc -c downstream/ctc/libriphone.yaml \
-o "config.downstream_expert.datarc.test_base_path=path_to_laser_experiment/states-3600.ckpt,,config.runner.freeze_layers=False,,config.runner.baseline=custom"
```
Testing:
```
python3 run_downstream.py -m evaluate -e /path_to_pr_experiment/dev-best.ckpt \
-o "config.downstream_expert.datarc.test_base_path=path_to_laser_experiment/states-3600.ckpt,,config.runner.freeze_layers=False,,config.runner.baseline=custom"
```

### For ASR
Training:
```
python3 run_downstream.py -p /path_to_asr_experiment -m train -u hubert_base -d asr \
-o "config.downstream_expert.datarc.test_base_path=path_to_laser_experiment/states-3600.ckpt,,config.runner.freeze_layers=False,,config.runner.baseline=custom"
```
OR
```
python3 run_downstream.py -p /path_to_asr_experiment -m train -u wavlm_base -d asr \
-o "config.downstream_expert.datarc.test_base_path=path_to_laser_experiment/states-3600.ckpt,,config.runner.freeze_layers=False,,config.runner.baseline=custom"
```
Testing:
```
python3 run_downstream.py -m evaluate -t "test-clean" -e /path_to_asr_experiment/dev-clean-best.ckpt \
-o "config.downstream_expert.datarc.test_base_path=path_to_laser_experiment/states-3600.ckpt,,config.runner.freeze_layers=False,,config.runner.baseline=custom"
```

