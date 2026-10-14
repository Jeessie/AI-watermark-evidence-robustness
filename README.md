# A 组：数据准备 + KGW 水印生成

## 资源链接

| 资源 | 链接 | 用途 |
|---|---|---|
| MarkLLM（水印工具包） | https://github.com/THU-BPM/MarkLLM | KGW 的生成和检测代码；仓库自带 C4 样本 `dataset/c4/c4.json`（13,863 篇） |
| C4 realnewslike | https://huggingface.co/datasets/allenai/c4 （子集 `realnewslike`） | prompt 和人写文本。默认用 MarkLLM 自带的那份，不用下载 |
| Qwen2.5-1.5B（base） | https://huggingface.co/Qwen/Qwen2.5-1.5B | 生成模型，和 Tamim & Khan 原论文一致 |
| KGW 原论文 | https://arxiv.org/abs/2301.10226 | γ=0.25, δ=2.0 的设置来源 |
| Reliability of Watermarks（Kirchenbauer 2024） | https://arxiv.org/abs/2306.04634 | 改写后长文本仍可检测，写 related work 必须引用 |

## 文件

```
configs/KGW_g025.json   KGW 参数：γ=0.25, δ=2.0, z 阈值 4（MarkLLM 默认 γ=0.5，这里改成了 0.25）
prepare_data.py         A1：从 C4 构造 prompt 和人写文本（不需要 GPU）
generate.py             A2：生成带水印文本（需要 GPU），记录速度，跑 sanity check
```

## Colab 上的运行步骤

Runtime → Change runtime type → **T4 GPU**。

```python
# 1. 把本文件夹上传到 Google Drive，然后挂载并进入
from google.colab import drive; drive.mount('/content/drive')
%cd /content/drive/MyDrive/watermark_project

# 2. 拿 MarkLLM 源码（只用 KGW，只需要 torch + transformers，不要 pip install 它的 requirements.txt）
!git clone --depth 1 https://github.com/THU-BPM/MarkLLM.git

# 3. A1：准备数据（约 10 秒）
!python prepare_data.py --c4_path MarkLLM/dataset/c4/c4.json

# 4. A2 pilot：先生成 10 条，看速度和 sanity check
!python generate.py --limit 10

# 5. 没问题就跑完 200 条（断线后重跑同一行会从断点继续）
!python generate.py

# 6. （可选）不加水印的模型文本，作为额外对照
!python generate.py --no_watermark --limit 50
```

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
