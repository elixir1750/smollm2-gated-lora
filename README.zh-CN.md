# smollm2-mtp-phrase-proposer

这是一个在 `HuggingFaceTB/SmolLM2-135M` 上验证 gated-LoRA mask-token MTP phrase proposer 的最小实验仓库。

模型输入是：

```text
x_1 ... x_t <mtp_1> <mtp_2> <mtp_3> <mtp_4>
```

整个序列经过同一个 frozen transformer backbone。在 transformer 内部被替换的 Linear 层使用 token-level gated LoRA：

```text
y = W x + lora_gate * LoRA(x)
```

其中：

- prefix token 的 `lora_gate = 0`
- `<mtp_k>` token 的 `lora_gate = 1`
- base model weights 冻结
- 原始 `lm_head` 冻结
- LoRA A/B 参数可训练
- MTP token 输入 embedding 通过 `mtp_embedding_delta` 训练

最后从 MTP token 位置取 logits：

```text
lm_head(hidden_at_<mtp_1>) -> x_{t+1}
lm_head(hidden_at_<mtp_2>) -> x_{t+2}
lm_head(hidden_at_<mtp_3>) -> x_{t+3}
lm_head(hidden_at_<mtp_4>) -> x_{t+4}
```

这个仓库不实现 speculative decoding，也不实现 GFlowNet。

## 安装

```bash
cd smollm2-mtp-phrase-proposer
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 快速训练

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

训练日志写入 `outputs/smoke/train_log.jsonl`。

## 评估

```bash
python -m src.eval \
  --checkpoint_dir outputs/smoke \
  --max_eval_samples 500 \
  --top_k 4 \
  --beam_size 16
```

输出：

```text
outputs/smoke/metrics.json
```

## 训练前/训练后观察

需要先完成一次训练，确保 `outputs/smoke/config.json` 和 `outputs/smoke/trainable.pt` 存在。

```bash
python -m src.observe \
  --checkpoint_dir outputs/smoke \
  --max_eval_samples 200 \
  --top_k 4 \
  --beam_size 16 \
  --max_phrase_len 4 \
  --num_examples 20
```

或者：

```bash
bash scripts/observe_before_after.sh
```

输出：

```text
outputs/smoke/before_after_observe.json
```

里面包含：

- 训练前/训练后的 `mtp_step_1/2/3/4_acc@1/@4/@5/@50`
- after minus before 的 delta
- phrase recall / gold rank / MRR
- `any_len_prefix_1/2/3/4_recall@beam`
- 样本级 example：gold phrase、训练前 rank、训练后 rank、prefix hit、top beam candidates

## Beam Demo

```bash
python -m src.beam \
  --checkpoint_dir outputs/smoke \
  --prompt "The capital of France is" \
  --top_k 4 \
  --beam_size 16 \
  --max_phrase_len 4
```

## 数据

默认训练数据：

```text
--dataset_name smollm2
--dataset_config cosmopedia-v2
```

`smollm2` 是本仓库 alias，会映射到：

```text
HuggingFaceTB/smollm-corpus
```

重要：`HuggingFaceTB/smollm-corpus` 很大。默认路径使用 Hugging Face streaming，边读边 tokenize，攒够 `--max_train_samples` 和 `--max_eval_samples` 指定的 token blocks 就停，避免 smoke test 下载完整 104 个 parquet shards。

## 方法

对 token block `x_1 ... x_T`，随机采样 prefix length `t`。输入构造为：

```text
x_1 ... x_t <mtp_1> <mtp_2> <mtp_3> <mtp_4>
```

targets：

```text
<mtp_1> -> x_{t+1}
<mtp_2> -> x_{t+2}
<mtp_3> -> x_{t+3}
<mtp_4> -> x_{t+4}
```

LoRA gate：

```text
[0, ..., 0, 1, 1, 1, 1]
```

prefix positions 严格走 frozen base path；MTP positions 走 frozen base path + LoRA residual。

默认替换 Llama-like/SmolLM2 常见 Linear 层：

- attention：`q_proj`、`k_proj`、`v_proj`、`o_proj`
- MLP：`gate_proj`、`up_proj`、`down_proj`

## Loss

```text
L_mtp = sum_k alpha_k * CE(logits_at_<mtp_k>, target_token_{t+k})
```

默认：

```text
alpha = [1.0, 0.7, 0.5, 0.3]
```

## Beam Search

beam search 使用 `<mtp_1>` ... `<mtp_4>` 四个位置的 logits。

默认真实约束：

```bash
--top_k 4
--beam_size 16
```

phrase score 是 token logprob 之和。

## 指标

Token-level：

- `mtp_step_1_acc@1/@4/@5/@50`
- `mtp_step_2_acc@1/@4/@5/@50`
- `mtp_step_3_acc@1/@4/@5/@50`
- `mtp_step_4_acc@1/@4/@5/@50`

Phrase-level：

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

`phrase_len_k_recall@N` 只看第 `k` 层生成出的长度为 `k` 的 candidates。

`any_len_prefix_k_recall@beam` 会把所有长度的 beam candidates 都放在一起看：只要某个候选的前 `k` 个 token 等于原文未来的前 `k` 个 token，就算长度 `k` 的 phrase 被召回。

## 代码结构

```text
smollm2-mtp-phrase-proposer/
  README.md
  README.zh-CN.md
  requirements.txt
  pyproject.toml
  scripts/
    train.sh
    eval.sh
    run_beam_demo.sh
    observe_before_after.sh
  src/
    __init__.py
    config.py
    data.py
    gated_lora.py
    model.py
    train.py
    eval.py
    observe.py
    beam.py
    metrics.py
    utils.py
  outputs/
    .gitkeep
```

文件职责：

- `src/config.py`：dataclass 参数和 CLI parser
- `src/data.py`：dataset loading、streaming tokenize、block grouping、prefix sampling、collator
- `src/gated_lora.py`：`GatedLoRALinear` 和 token-level gate 上下文
- `src/model.py`：添加 MTP tokens、构造 LoRA gate、forward、checkpoint
- `src/train.py`：训练入口
- `src/eval.py`：评估入口
- `src/observe.py`：训练前/训练后对比观察脚本
- `src/beam.py`：beam search phrase candidate generation 和 demo
- `src/metrics.py`：token acc、phrase recall/rank/MRR
- `src/utils.py`：seed、logging、device、JSON helper

## Checkpoint

保存内容：

- LoRA A/B 参数
- `mtp_embedding_delta`
- tokenizer
- config

加载时重新构造 frozen base model、添加 MTP tokens、替换 gated LoRA Linear，然后加载 `trainable.pt`。
