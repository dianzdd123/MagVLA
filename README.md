# Mag-VLA | Bimanual magnetic microrobot manipulation

Mag-VLA predicts coordinated planar movements of two magnet-mounted arms from microscope video, a language instruction and the arms’ measured positions. The external magnets steer the microrobot as it approaches cargo and transports it through a vascular phantom. Start with [the main ACT implementation](MARSS/codebase/policy.py).

**Paper.** Yongchen Wang and Kangyi Lu (equal contribution), Lan Wei, and Dandan Zhang. *Mag-VLA: Vision-Language-Action Model for Bimanual Magnetically Actuated Microrobot Manipulation* (2026). [IEEE publication](https://ieeexplore.ieee.org/document/11655691) · [Open full text](https://arxiv.org/html/2605.28486v1)

## From the paper to the code

| Paper reference | Implementation and scope |
| --- | --- |
| Figure 2 and Section III-B | `MagVLAPolicy._run_vlm` encodes four images and the instruction. `_inject_state` prepends a token for the two arms’ planar positions in the order left x, left y, right x, right y. |
| Section III-C.1 and Eq. (1) | `PhaseHead` combines pooled features with current arm positions and the displacement between the oldest and newest states. `ACTChunkHead` uses a learned phase embedding and five action queries to predict five successive four-axis increments. The code uses labelled phases during training and predicted phases in evaluation mode. |
| Section III-C.2 and Eq. (2) | The paper combines overlapping predicted chunks during deployment. The [offline reference implementation](MARSS/codebase/temporal_ensemble.py) accepts one predicted chunk per replan. It implements the equation independently of the hardware controller. |
| Section III-D and Eq. (4) | The ACT training objective combines smooth L1 action loss (beta 1) with phase cross-entropy. The released configuration sets the phase weight to 0.5. |
| Table I and Figure 4 | The paper compares four backbones with a common MLP action head. The release covers the Qwen2.5-VL-7B action-head experiments. It lacks the four-backbone MLP comparison. |
| Table II and Figure 5 | The three corresponding directories are `MARSS` (ACT), `MARSS-diffusion` and `MARSS-flow`. The paper reports test RMSE of 79.56, 153.52 and 140.50 actuator ticks respectively. Training logs report validation objectives in different units. |
| Section IV-E | Physical trials assess approach and cargo transport across tasks A–C. Robot rollouts require the external control system. The CAD files describe physical parts and are independent of policy training. |

The paper reports 75 demonstrations with 20,724 frames at 10 Hz and a 60/9/6 episode split. For comparisons with Table II, use held-out test episodes, sample predictions without action or phase targets, and undo action normalisation before computing RMSE. Endpoint error uses the four-axis action at the fifth step. The release does not include the demonstrations, exact split files, trained weights or the original test-scoring script. The two additional folders, `MARSS-label` and `MARSS-test`, are exploratory extensions outside the reported action-head comparison.

## Choose an experiment

| Directory | What changes | What the comparison can tell you |
| --- | --- | --- |
| [MARSS](MARSS/codebase) | The paper’s main model adapts Qwen2.5-VL-7B-Instruct with LoRA. A phase head classifies approach or transport. An ACT-style head regresses five action increments using smooth L1 loss. | Establishes a reference for Table II. |
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

## Run the main ACT model

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
