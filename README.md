# IntentTalk

IntentTalk is a holistic co-speech motion generator with future-grounded
discrete planning. The released source contains the final model, its
future-motion tokenizer, data preprocessing utilities, training/evaluation
code, and configuration files.

This repository intentionally does **not** include datasets, pretrained model
weights, cached features, generated motions, or experimental outputs.

## Method at a glance

The system is trained in two stages:

1. `FuturePlanTokenizer` learns a discrete codebook from shifted future windows
   of coordinated upper-body, hand, and lower-body RVQ latents.
2. `IntentTalk` predicts those plan codes from speech semantics. Retrieved plan
   features control gesture realization through hierarchical temporal FiLM,
   while an event-aware gate controls when semantic motion is activated. The
   body branches are coordinated before residual-VQ token prediction; the
   facial stream is generated through the speech-conditioned context pathway.

At inference time, future motion is not required: plans are predicted from the
speech inputs and retrieved from the frozen codebook.

## Repository layout

```text
configs/
  future_plan_tokenizer.yaml  # future-plan tokenizer pretraining
  intenttalk.yaml             # final generator training/evaluation
dataloaders/                  # cached-data readers and preprocessing utilities
models/
  backbone.py                 # context and holistic motion encoders
  future_plan_tokenizer.py    # shifted-future tokenizer and codebook
  intenttalk.py               # final speech-to-plan-to-motion network
optimizers/                   # optimizer and scheduler utilities
utils/                        # metrics, rendering, configuration, and transforms
future_plan_tokenizer_trainer.py
generation_trainer.py
intenttalk_trainer.py
train.py
```

## Environment

The code was developed with Python 3.8 and PyTorch 2.1. Install a PyTorch build
matching your CUDA driver first, then install the remaining packages:

```bash
pip install -r requirements.txt
```

The OpenAI CLIP package is installed by the final line of
`requirements.txt`. Some visualization dependencies require system OpenGL and
FFmpeg packages.

## External assets

Obtain the assets under their respective licenses and place them outside Git.
The default configuration expects this layout:

```text
BEAT2/beat_english_v2.0.0/       # BEAT2 data, split metadata, and evaluator assets
  weights/
    AESKConv_240_100.bin
    vocab.pkl
datasets/                         # preprocessed LMDB and test pickle
facebook/hubert-large-ls960-ft/  # HuBERT feature extractor
Systran/faster-whisper-large-v3/ # speech transcription model
weights/
  context_motion_encoder.bin
  future_plan_tokenizer.bin
  pretrained_vq/
    rvq_face_600.bin
    rvq_upper_500.bin
    rvq_hands_500.bin
    rvq_lower_600.bin
    last_1700_foot.bin
  smplx_models/                   # official SMPL-X model files
```

Paths can be edited in the YAML files. `INTENTTALK_PRETRAINED_VQ_DIR`,
`INTENTTALK_CACHE_ROOT`, and `INTENTTALK_TMPDIR` can override the corresponding
defaults.

## Training

Pretrain the future-plan tokenizer:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config configs/future_plan_tokenizer.yaml \
  --notes _plan_tokenizer \
  --random_seed 43
```

Copy the selected tokenizer checkpoint to the path configured by
`plan_tokenizer_ckpt`, then train IntentTalk. An optional compatible generator
checkpoint can be provided through `generator_init_ckpt` for staged training:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config configs/intenttalk.yaml \
  --context_encoder_ckpt ./weights/context_motion_encoder.bin \
  --plan_tokenizer_ckpt ./weights/future_plan_tokenizer.bin \
  --notes _intenttalk \
  --random_seed 43
```

## Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config configs/intenttalk.yaml \
  --test_state \
  --load_ckpt /path/to/intenttalk_checkpoint.bin
```

Generated files are written under `outputs/custom/`. The output directory and
all model/data artifacts are ignored by Git.

## Data and model licenses

BEAT2, SMPL-X, HuBERT, Whisper, CLIP, and any third-party pretrained weights
remain subject to their original licenses and terms. Add an explicit repository
license only after confirming that redistribution of all retained source files
is permitted.
