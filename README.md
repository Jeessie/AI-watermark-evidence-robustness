# A 组：数据准备 + KGW 水印生成

## 资源链接

| 资源 | 链接 | 用途 |
|---|---|---|
| MarkLLM（水印工具包） | https://github.com/THU-BPM/MarkLLM | KGW 的生成和检测代码；仓库自带 C4 样本 `dataset/c4/c4.json`（13,863 篇） |
| C4 realnewslike | https://huggingface.co/datasets/allenai/c4 （子集 `realnewslike`） | prompt 和人写文本。默认用 MarkLLM 自带的那份，不用下载 |
| Qwen2.5-1.5B（base） | https://huggingface.co/Qwen/Qwen2.5-1.5B | 生成模型，和 Tamim & Khan 原论文一致 |
| KGW 原论文 | https://arxiv.org/abs/2301.10226 | γ=0.25, δ=2.0 的设置来源 |
| Reliability of Watermarks（Kirchenbauer 2024） | https://arxiv.org/abs/2306.04634 | 改写后长文本仍可检测，写 related work 必须引用 |
| Online LLM watermark detection via e-processes（Su, Wang, Zhao 2026） | https://arxiv.org/abs/2602.14286 | 序贯检测（e-process）方法的来源 |

## 文件

```
configs/KGW_g025.json   KGW 参数：γ=0.25, δ=2.0, z 阈值 4（MarkLLM 默认 γ=0.5，这里改成了 0.25）
prepare_data.py         A1：从 C4 构造 prompt 和人写文本（不需要 GPU）
generate.py             A2：生成带水印文本（需要 GPU），记录速度，跑 sanity check
run_generate.slurm      在 CARC 上提交 generate.py 的 Slurm 脚本
data/                   A1 的输出（已提交），A2 的输出也写在这里
```

## 在 CARC 上的运行步骤

A1 的数据（`data/prompts.jsonl`、`data/human.jsonl`）**已经提交在仓库里**，clone 下来就能直接用，不用再跑 `prepare_data.py`。重跑也会得到完全相同的数据（固定 seed；MarkLLM 自带的 `c4.json` 和 HF 上 realnewslike validation 的前 13,863 篇一致）。

```bash
# 1. 登录 CARC，clone 仓库（建议放在 /project 下，home 配额小）
ssh <netid>@discovery.usc.edu
git clone https://github.com/Jeessie/AI-watermark-evidence-robustness.git
cd AI-watermark-evidence-robustness

# 2. 拿 MarkLLM 源码（只用 KGW，只需要 torch + transformers，不要 pip install 它的 requirements.txt）
git clone --depth 1 https://github.com/THU-BPM/MarkLLM.git

# 3. 建环境（第一次即可）
module load python          # 或 module load conda，看 `module avail` 里有什么
python -m venv .venv && source .venv/bin/activate
pip install torch transformers datasets

# 4. 把 run_generate.slurm 里的 --account=CHANGE_ME 改成自己的项目账号（用 `myaccount` 查）

# 5. A2 pilot：先生成 10 条，看速度和 sanity check（输出在 logs/slurm-<jobid>.out）
sbatch run_generate.slurm --limit 10

# 6. 没问题就跑完 200 条（作业被杀或超时后重新提交会从断点继续）
sbatch run_generate.slurm

# 7. （可选）不加水印的模型文本，作为额外对照
sbatch run_generate.slurm --no_watermark --limit 50
```

`squeue --me` 查看排队状态。交互式调试可以用 `salloc --partition=gpu --gres=gpu:1 --time=1:00:00 --account=<账号>`，拿到节点后直接 `python generate.py --limit 10`。

（A1 如需重跑，不需要 GPU：`python prepare_data.py`。）

## 输出

| 文件 | 条数 | 内容 |
|---|---|---|
| `data/prompts.jsonl` | 200 | `id, prompt_ids, prompt_text`（每篇文章前 50 token） |
| `data/human.jsonl` | 500 | `id, source="human", prompt_ids, token_ids, text`（同一篇文章接下来的 600 token） |
| `data/wm.jsonl` | 200 | `id, source="wm", prompt_ids, token_ids, text`（模型续写的 600 token） |
| `data/nowm.jsonl` | 可选 | 同上，`source="nowm"` |
| `logs/throughput.jsonl` | — | GPU 型号、tok/s、耗时 → 用来估算 GPU 小时和费用 |

prompt 和人写文本来自**不同的文章**，不会重叠。

## 给 B 组和 C 组的注意事项

- **检测时要用 `prompt_ids` 的最后一个 token 作为前缀**：KGW 用前一个 token 决定 green list，所以第 1 个 token 的 green list 由 prompt 最后一个 token 决定。写法参考 `generate.py` 末尾的 sanity check：
  `kgw.utils.score_sequence(torch.tensor(prompt_ids[-1:] + token_ids))` 返回 `(z, flags)`，`flags[0]` 是前缀，记为 -1，后面每一位是 1（green）或 0（red）。
- **`vocab_size` 必须用 `model.config.vocab_size`（151936）**，不能用 `len(tokenizer)`。检测时也要用同一个值，否则 green list 对不上。
- 改写后的文本只有 `text`，没有 token ids，需要用 `tok(text, add_special_tokens=False)` 重新切分。原始文本的 decode → encode 往返可能和 `token_ids` 有几个 token 的差异，所以干净文本的检测直接用 `token_ids`。
- 生成参数：temperature 0.7、repetition_penalty 1.1、强制恰好 600 个新 token。如果要改参数，在 `logs/` 里记下来，写报告时要用。

## 已经验证过的

本地 CPU 冒烟测试（Qwen2.5-0.5B，2 条，150 token）：

```
wm     z=  4.24  green=0.40
wm     z= 13.11  green=0.71
human  z=  0.57  green=0.26
human  z= -2.36  green=0.21
human  z= -2.64  green=0.20
```

水印文本的 z 远高于人写文本，人写文本的 green 比例约等于 γ=0.25，说明 pipeline 是对的。第一条水印文本只有 150 token，z 刚过 4，这正好说明长度对检测很重要。
