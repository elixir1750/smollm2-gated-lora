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
