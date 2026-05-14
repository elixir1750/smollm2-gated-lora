# smollm2-mtp-phrase-proposer

Minimal feasibility repo for a gated-LoRA mask-token MTP phrase proposer on `HuggingFaceTB/SmolLM2-135M`.

The model input is:

```text
x_1 ... x_t <mtp_1> <mtp_2> <mtp_3> <mtp_4>
```

The frozen transformer backbone processes the whole sequence. Inside selected transformer Linear layers, a token-level LoRA gate is applied:

```text
y = W x + lora_gate * LoRA(x)
```

where:

- prefix token positions have `lora_gate = 0`
- `<mtp_k>` token positions have `lora_gate = 1`
- base model weights are frozen
- original `lm_head` is frozen
- LoRA A/B parameters are trainable
- MTP token input embeddings are trainable via `mtp_embedding_delta`

Predictions are taken from the MTP token positions:

```text
lm_head(hidden_at_<mtp_1>) -> x_{t+1}
lm_head(hidden_at_<mtp_2>) -> x_{t+2}
lm_head(hidden_at_<mtp_3>) -> x_{t+3}
lm_head(hidden_at_<mtp_4>) -> x_{t+4}
```

This repo does not implement speculative decoding or GFlowNet.

## Install

```bash
cd smollm2-mtp-phrase-proposer
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Smoke Train

```bash
python -m src.train \
  --model_name_or_path HuggingFaceTB/SmolLM2-135M \
  --dataset_name smollm2 \
  --dataset_config cosmopedia-v2 \
  --max_train_samples 2000 \
  --max_eval_samples 200 \
  --block_size 256 \
  --max_phrase_len 4 \
  --lora_rank 8 \
  --learning_rate 2e-4 \
  --num_train_steps 1000 \
  --per_device_train_batch_size 4 \
  --output_dir outputs/smoke
```

Training writes `train_log.jsonl` with `loss_mtp`, per-step losses, and learning rate.

## Colab CUDA Training

Mac/MPS can run smoke tests, but this gated-LoRA MTP setup still needs full transformer forward/backward through the frozen backbone, so longer runs are much happier on CUDA. A practical first CUDA run is:

```bash
bash scripts/colab_train_10m.sh
```

这个脚本会跑：

- `block_size=256`
- global batch size `8`
- `5000` optimizer steps
- 约 `10M` training tokens
- `top_k=4`, `beam_size=16` 的 before/after observe

在 Colab 里可以这样启动：

```bash
!git clone <your-repo-url>
%cd smollm2-mtp-phrase-proposer
!pip install -r requirements.txt
!bash scripts/colab_train_10m.sh
```

如果你是手动上传本地目录到 Colab 或 Google Drive，把 `%cd` 切到仓库目录后执行后两行即可。

训练完成后重点看：

- `outputs/colab_10m/before_after_observe.json`
- `after.mtp_step_{1..4}_acc@1/@4`
- `after.phrase_len_4_recall@16`
- `after.any_len_prefix_4_recall@beam`
- `delta_after_minus_before`

如果 Colab 显存不够，优先降低这些参数：

```bash
OUTPUT_DIR=outputs/colab_light python -m src.train \
  --dataset_name smollm2 \
  --dataset_config cosmopedia-v2 \
  --block_size 192 \
  --pad_to_length 192 \
  --max_train_samples 8000 \
  --num_train_steps 3000 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --lora_rank 4 \
  --dtype float16 \
  --device cuda \
  --output_dir outputs/colab_light
```

## Eval

```bash
python -m src.eval \
  --checkpoint_dir outputs/smoke \
  --max_eval_samples 500 \
  --top_k 4 \
  --beam_size 16
```

Evaluation writes `metrics.json`.

## Before/After Observation

Run training first so `outputs/smoke/config.json` and `outputs/smoke/trainable.pt` exist.

```bash
python -m src.observe \
  --checkpoint_dir outputs/smoke \
  --max_eval_samples 200 \
  --top_k 4 \
  --beam_size 16 \
  --max_phrase_len 4 \
  --num_examples 20
```

This writes `before_after_observe.json` with before/after token accuracy, phrase metrics, `any_len_prefix_{1..4}_recall@beam`, and example beam candidates.

## Beam Demo

```bash
python -m src.beam \
  --checkpoint_dir outputs/smoke \
  --prompt "The capital of France is" \
  --top_k 4 \
  --beam_size 16 \
  --max_phrase_len 4
```

## Dataset

The default training dataset alias is `smollm2`, which maps to `HuggingFaceTB/smollm-corpus` with the `cosmopedia-v2` subset. The default path uses Hugging Face streaming and stops once enough token blocks have been collected, avoiding full multi-shard downloads during smoke tests.

You can still pass any Hugging Face dataset through `--dataset_name` and `--dataset_config`.

## Method

For a token block `x_1 ... x_T`, training samples a prefix length `t` and constructs:

```text
x_1 ... x_t <mtp_1> <mtp_2> <mtp_3> <mtp_4>
```

Targets:

```text
<mtp_1> -> x_{t+1}
<mtp_2> -> x_{t+2}
<mtp_3> -> x_{t+3}
<mtp_4> -> x_{t+4}
```

The first version uses the standard causal mask. Prefix positions cannot see later MTP positions; each MTP token can see the prefix and previous MTP tokens.

By default, gated LoRA wraps Llama-like/SmolLM2 module names:

- attention: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- MLP: `gate_proj`, `up_proj`, `down_proj`

## Loss

```text
L_mtp = sum_k alpha_k * CE(logits_at_<mtp_k>, target_token_{t+k})
```

Default:

```text
alpha = [1.0, 0.7, 0.5, 0.3]
```

## Beam Search

Beam search uses logits at `<mtp_1>` ... `<mtp_4>`. For each depth it takes `top_k`, combines candidates with beam search, and scores each phrase with the sum of token log-probabilities.

Default practical constraint:

```bash
--top_k 4
--beam_size 16
```

## Metrics

Token-level:

- `mtp_step_1_acc@1/@4/@5/@50`
- `mtp_step_2_acc@1/@4/@5/@50`
- `mtp_step_3_acc@1/@4/@5/@50`
- `mtp_step_4_acc@1/@4/@5/@50`

Phrase-level:

- `phrase_len_1_recall@10/@16/@50/@100`
- `phrase_len_2_recall@10/@16/@50/@100`
- `phrase_len_3_recall@10/@16/@50/@100`
- `phrase_len_4_recall@10/@16/@50/@100`
- `any_len_prefix_1_recall@beam`
- `any_len_prefix_2_recall@beam`
- `any_len_prefix_3_recall@beam`
- `any_len_prefix_4_recall@beam`
- `mean_gold_rank`
- `median_gold_rank`
- `MRR`

`phrase_len_k_recall@N` checks exact length-`k` candidates at depth `k`. `any_len_prefix_k_recall@beam` checks all beam candidates of any length and counts a hit when the first `k` tokens match the gold future prefix.

## Checkpoints

`save_checkpoint` stores:

- trainable LoRA A/B parameters
- trainable `mtp_embedding_delta`
- tokenizer files
- JSON config

Loading reconstructs the frozen base model, adds MTP tokens, wraps gated LoRA layers, then loads the trainable state.
