# MagVLA | Magnetic microrobot policy training and CAD

MARSS contains five implementations of a vision–language–action policy for magnetic microrobot manipulation. Given recent camera frames, measured state and a language instruction, the baseline predicts the current manipulation phase and a short sequence of action increments. The accompanying CAD files describe the robot and maze used as physical design assets. Training consumes recorded demonstrations and does not load either CAD part.

## Choose an experiment

| Directory | What changes | What the comparison can tell you |
| --- | --- | --- |
| [MARSS](MARSS/codebase) | The baseline adapts Qwen2.5-VL-7B-Instruct with LoRA. A phase head classifies approach or transport. An ACT-style head regresses five action increments using smooth L1 loss. | Establishes a direct action-regression reference. |
| [MARSS-diffusion](MARSS-diffusion/codebase) | The action head learns to predict noise added to a demonstrated chunk. DDIM sampling generates a chunk at inference. | Tests iterative denoising as an alternative action generator. |
| [MARSS-flow](MARSS-flow/codebase) | The action head learns a conditional velocity field between Gaussian noise and demonstrated actions. Euler integration generates a chunk at inference. | Tests a continuous-time action generator. |
| [MARSS-label](MARSS-label/codebase) | Red and green circles mark the object and target on camera frames before resizing. Coordinates and radius come from per-episode annotations. | Tests the effect of supplying explicit visual location cues. |
| [MARSS-test](MARSS-test/codebase) | A DistilBERT encoding of the task instruction supplies an additional text token directly to the ACT head. | Tests a second route for language conditioning. This directory runs training. |

The regression, diffusion and flow heads optimise different targets. To compare variants, use the same held-out episodes and evaluate sampled action chunks in a common action space. Raw training losses have different meanings.

## Follow one sample through the code

1. **Load the demonstration.** The [dataset loader](MARSS/codebase/dataset.py) pairs an episode video with its Parquet table by episode number and maps the row’s task index to an instruction. It reads four frames ending at the current observation and aligns the four-dimensional state history to the same indices. At the start of an episode it repeats the first observation as needed.
2. **Build the target.** Starting at the current row, the loader takes five consecutive four-dimensional action increments, yielding a target of shape **5 × 4**. It skips incomplete future windows and windows containing the return phase. The retained phase labels are 0 (approach) and 1 (transport).
3. **Predict phase and actions.** The [policy](MARSS/codebase/policy.py) processes visual and language input with the pretrained backbone, incorporates measured state, and produces phase logits and an action chunk. The variant-specific policy file implements the corresponding action head.
4. **Optimise and evaluate.** The [training entry point](MARSS/codebase/train_ddp.py) uses one PyTorch distributed process per GPU. It batches samples, accumulates gradients, evaluates held-out data and writes logs and checkpoints from rank zero. The [configuration](MARSS/codebase/config.py) defines model identity, LoRA settings, learning rates, image scaling, loss weighting and training intervals. Each variant has its own copies of these modules.

## Prepare the demonstrations

Supply a dataset separately. The loaders expect the following paths relative to the dataset root. Episode filenames use six decimal digits.

| Path or pattern | Required contents |
| --- | --- |
| `meta/splits.json` | Episode index lists keyed by split. Training reads the train and validation splits. |
| `meta/tasks.jsonl` | One JSON object per line containing `task_index` and `task`. |
| `meta/info.json` | Dataset metadata. When present, the `normalization` object supplies `state_mean`, `state_std`, `action_delta_mean` and `action_delta_std`. |
| `data/chunk-000/episode_000000.parquet` | One row per frame, including `observation.state`, `action_delta`, `phase` and `task_index`. |
| `videos/chunk-000/observation.images.main/episode_000000.mp4` | Main-camera video aligned to the rows of the matching episode. |
| `annotations_point.csv` | For the marked-image experiment, supply `video_name`, `object_x`, `object_y`, `target_x`, `target_y` and `circle_radius`. Check annotation coverage because the loader can continue without marks. |

## Run the baseline

Use Python 3.10 or later and a CUDA-enabled PyTorch installation with NCCL and support for bfloat16. Install compatible PyTorch and torchvision builds for your GPU system first. The model weights must be accessible through Hugging Face or a local cache. The text-conditioning experiment also loads `distilbert-base-uncased`.

From the repository root, in an activated Python environment:

```bash
python -m pip install 'transformers>=4.49,<5' accelerate qwen-vl-utils peft wandb pandas pyarrow opencv-python-headless tqdm tensorboard pillow numpy
# Additional dependency for MARSS-diffusion
python -m pip install diffusers

export MARSS_DATA_DIR=/absolute/path/to/lerobot_datasetV2
export WANDB_MODE=offline
cd MARSS/codebase
torchrun --standalone --nproc_per_node=1 train_ddp.py --max_steps 1
```

The one-step run checks data loading, model initialisation, a training update and checkpoint writing on a GPU with sufficient memory. For a full run, omit `--max_steps 1` and choose a process count that matches the allocated GPUs. The baseline defaults to two samples per GPU and eight gradient accumulation steps, giving an effective batch size of 64 with four GPUs. Inspect each variant’s configuration before comparing runs.

For Slurm, activate the same environment and export the dataset path, then submit **from the chosen variant directory** (the directory containing `run_ddp.sh`). For example, run `sbatch --partition=YOUR_PARTITION run_ddp.sh`. The launcher requests four GPUs and starts four processes. If you change the allocation, set `MARSS_GPUS` to match. To use online W&B logging, set your own `WANDB_API_KEY` in the submitting environment and select online mode.

The default output directory is `MagVLA/` under the training working directory. Look in `logs/` for TensorBoard output and `checkpoints/` for saved states including `final.pt`. Pass `--resume /absolute/path/to/checkpoint.pt` to resume a compatible run. Use separate working directories for repeated runs to preserve their outputs.

## CAD files

| File | Role | How to inspect it |
| --- | --- | --- |
| [MARSS-robot.SLDPRT](cad/MARSS-robot.SLDPRT) | Part supplied as the microrobot design. | Open it in a compatible SOLIDWORKS installation and inspect its bodies, sketches and dimensions in the active configuration. |
| [maze.SLDPRT](cad/maze.SLDPRT) | Part supplied as the maze design. The uploaded filename was `maze SLDPRT`. | Open it separately and use the Measure tool to inspect passage widths. Compare with the robot envelope using the same document units. |

Both files preserve the supplied binary contents. The release includes no assembly mates, fabrication drawing, tolerance specification or material schedule. CAD units, dimensions and geometry have not been independently verified. The training archive contains neither demonstrations nor trained checkpoints, hardware drivers or a hardware deployment entry point.
