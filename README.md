# When Does AI Watermark Evidence Hold Up?

midterm 范围：只做 KGW，小规模跑通"生成 → 攻击 → 检测 → 分析"整条 pipeline。进度见 [TODO.md](TODO.md)。

## 分工

| 组 | 人数 | 负责 | 输入 | 输出 |
|---|---|---|---|---|
| **A：数据和生成** | 1 | C4 数据准备；在 CARC 上生成 KGW 水印文本；记录速度（用于 GPU 时长和费用估计） | MarkLLM 自带的 C4 | `data/prompts.jsonl`、`data/human.jsonl`、`data/wm.jsonl`、`logs/throughput.jsonl` |
| **B：攻击** | 2 | 用 Qwen2.5-1.5B-Instruct 逐句改写，攻击强度为改写 25% / 50% / 100% 的句子（三种强度嵌套）；质量过滤（语义相似度 ≥ 0.75）；人写文本做同样的攻击，作为误报对照 | `data/wm.jsonl`、`data/human.jsonl` | `data/attacked_wm.jsonl`、`data/attacked_human.jsonl`（含 `sim`、`token_edit_ratio`、`len_ratio`、`valid`） |
| **C：检测和统计** | 2 | 提取逐 token 的 green flags（同一对 token 只计一次）；固定长度 z-test、peeking z-test、e-process；人写文本上的 FPR 检查；相同 FPR 下比较两种检测；画图 | A、B 的全部输出 | peeking 演示、长度曲线、长度 × 攻击强度热图、停止时间分布 |
| **D：文献和 slides** | 1 | 读原论文（Tamim & Khan）、KGW、Kirchenbauer 2024、e-process 论文；写 related work；整理 GPU 时长和费用估计；做 midterm slides、组织排练 | 各组的结果图 | midterm slides |

周五以后 A 组去帮 C 组做分析和画图。

---

以下是 **A 组**的说明。

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
kgw_cpu.py              让 KGW 的 green list 在 CPU 上生成 → 任何机器 / 显卡上检测结果都一样（生成和检测都必须用）
c_eprocess/             C 组：flags 抽取、固定长度 z / peeking z / e-process 三条检测、FPR 与停止时间评估
docs/C_report.md        C 组完整报告：方法、数据、结果、讨论、局限、复现命令
verify_kgw.py           不加载模型，对整个文件重新检测，输出 z 统计和逐 token 结果的指纹 flags_md5（用来自查环境）
run_generate.slurm      在 CARC 上提交 generate.py 的 Slurm 脚本
setup_env.sh            在 CARC 上建 group4 共享环境（torch 2.5.1 + cu121，P100/V100/新卡都能跑）
data/                   A1 的输出（已提交），A2 的输出也写在这里
```

## 在 CARC 上的运行步骤

A1 的数据（`data/prompts.jsonl`、`data/human.jsonl`）**已经提交在仓库里**，clone 下来就能直接用，不用再跑 `prepare_data.py`。重跑也会得到完全相同的数据（固定 seed；MarkLLM 自带的 `c4.json` 和 HF 上 realnewslike validation 的前 13,863 篇一致）。

group4 已经准备好了**共享环境、共享模型缓存和仓库副本**，组员不用下载模型、不用配环境：

```bash
# 1. 登录 CARC（校外先连 USC VPN；域名解析不了时可以直接用 IP：ssh <netid>@10.72.0.14）
ssh <netid>@discovery.usc.edu
cd /project2/xiangren_1987/group4/AI-watermark-evidence-robustness

# 2. 用共享环境跑脚本（只做 CPU 小任务时可以在登录节点上跑）
/project2/xiangren_1987/group4/envs/wm/bin/python verify_kgw.py data/wm.jsonl data/human.jsonl

# 3. GPU 任务用 Slurm 提交（课程账号、共享环境、共享模型缓存都已经在脚本里设置好）
sbatch run_generate.slurm --limit 10      # pilot
sbatch run_generate.slurm                 # 全部 200 条（断点续跑）
sbatch run_generate.slurm --no_watermark  # 不加水印的模型文本，作为额外对照
```

`squeue --me` 查看排队状态。**不要指定卡型**：A100/A40 要排很久，不指定的话通常几秒就能分到 P100/V100。

（A1 如需重跑，不需要 GPU：`python prepare_data.py`。）

## 本地跑通（uv + API/ollama，不需要 CARC）

检测和统计不需要大模型：green list 只用 token id 在 CPU 上算（见 `kgw_cpu.py`）。
攻击和生成各需要一个模型，但都可以换成小模型或 API，所以笔记本上就能把
`生成 → 攻击 → 检测 → 分析` 整条链跑一遍。

```bash
bash setup_local.sh                    # uv 建 .venv（torch+transformers+MarkLLM）+ 下载 Qwen2.5-1.5B-Instruct
.venv/Scripts/python.exe verify_kgw.py data/wm.jsonl data/human.jsonl
# 必须打印 flags_md5=5d98d773dd（wm）和 6375f943cb（human），否则 green list 算错了

# C1 → C4/C6 → C7（约 1 分钟 + 8 分钟 + 10 秒）
.venv/Scripts/python.exe c_eprocess/extract_flags.py data/wm.jsonl data/human.jsonl
.venv/Scripts/python.exe c_eprocess/run_eval.py --null data/flags_human.jsonl --wm data/flags_wm.jsonl --out results/real
.venv/Scripts/python.exe c_eprocess/make_plots.py --dir results/real

# 攻击数据有两条臂（flags 已给出，只想复现分析不用重跑）：
#   主臂 = B 组交付的 data/attacked_wm.jsonl / data/attacked_human.jsonl（200 篇/档，已入库）
#   强臂 = 下面的 attack_rewrite.py，走 dtens API 的 deepseek-v4.1-flash（20 篇/档）
.venv/Scripts/python.exe c_eprocess/extract_flags.py data/attacked_wm.jsonl data/attacked_human.jsonl
export DTENS_API_KEY=sk-...            # Windows: set DTENS_API_KEY=sk-...
.venv/Scripts/python.exe c_eprocess/attack_rewrite.py data/wm.jsonl data/human.jsonl \
    --out data/attacked_api.jsonl --limit 20
.venv/Scripts/python.exe c_eprocess/extract_flags.py data/attacked_api.jsonl \
    --tokenizer models/Qwen2.5-1.5B-Instruct --split      # --split 写出 _wm / _human 两份

# 攻击 → 检测 → 图（两臂一起跑，约 8 分钟）
.venv/Scripts/python.exe c_eprocess/attack_strength.py data/attacked_wm.jsonl data/attacked_api.jsonl \
    --orig data/wm.jsonl --out results/real/attack_strength.csv
.venv/Scripts/python.exe c_eprocess/run_eval.py --null data/flags_human.jsonl --wm data/flags_wm.jsonl \
    --attacked data/flags_attacked_wm.jsonl data/flags_attacked_human.jsonl \
               data/flags_attacked_api_wm.jsonl data/flags_attacked_api_human.jsonl \
    --out results/real
.venv/Scripts/python.exe c_eprocess/make_plots.py --dir results/real   # fig3 = 主臂，fig3_heatmap_strong = 强臂
```

本地生成（可选，用来验证 `generate.py` 这条路也通）：
`.venv/Scripts/python.exe generate.py --model models/Qwen2.5-1.5B-Instruct --limit 20 --out data/wm_local.jsonl`
（`data/wm.jsonl` 本身也是 Qwen2.5-1.5B 生成的（CARC P100，110 tok/s），本地用的是 Instruct 版本、
同规模，所以水印强度直接可比：green 比例 0.629 vs 0.627，z ≈ 22。）

实测速度（RTX 3060 Ti 8GB，torch 2.6.0+cu124）：生成 27–59 tok/s（batch 2–4）；`extract_flags` 700 条 1 分钟（CPU）；
`run_eval` 全量 5 种 e-process × dedup 开关约 8 分钟（OG 最慢）；改写 ~1.3 s/句（API）、~0.5 s/句（本地 1.5B）。

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

- **检测前必须调用 `patch_kgw_cpu(kgw)`**（见 `kgw_cpu.py`）。MarkLLM 原版在 GPU 上生成 green list，而 `torch.randperm` 在 P100 / V100 / A40 上对同一个种子给出的排列各不相同，换一张卡检测，水印文本的 z 会掉到 0 附近。补丁改成在 CPU 上生成（不同机器、torch 2.5–2.14 结果都一样），算法和 MarkLLM 原版完全相同。
- **自查环境**：`python verify_kgw.py data/wm.jsonl data/human.jsonl`，`flags_md5` 应该是 `wm=5d98d773dd`、`human=6375f943cb`。对不上说明 green list 算错了。
- **检测时要用 `prompt_ids` 的最后一个 token 作为前缀**：KGW 用前一个 token 决定 green list，所以第 1 个 token 的 green list 由 prompt 最后一个 token 决定。写法参考 `generate.py` 末尾的 sanity check：
  `kgw.utils.score_sequence(torch.tensor(prompt_ids[-1:] + token_ids))` 返回 `(z, flags)`，`flags[0]` 是前缀，记为 -1，后面每一位是 1（green）或 0（red）。
- **`vocab_size` 必须用 `model.config.vocab_size`（151936）**，不能用 `len(tokenizer)`。检测时也要用同一个值，否则 green list 对不上。
- 改写后的文本只有 `text`，没有 token ids，需要用 `tok(text, add_special_tokens=False)` 重新切分。原始文本的 decode → encode 往返可能和 `token_ids` 有几个 token 的差异，所以干净文本的检测直接用 `token_ids`。
- 生成参数：temperature 0.7、repetition_penalty 1.1、强制恰好 600 个新 token。如果要改参数，在 `logs/` 里记下来，写报告时要用。

## 已经验证过的

`data/wm.jsonl`（CPU green list 版本，2026-10-15 重新生成，P100 上 20 分钟）用 `verify_kgw.py` 在四个环境上检测，结果**逐 token 完全一致**：

| 环境 | wm（200 条） | human（500 条） |
|---|---|---|
| Mac CPU，torch 2.14.1 | z 13.39 / 21.50 / 36.20（最小 / 中位数 / 最大），200 条 z>4，`5d98d773dd` | z −4.90 / −0.75 / 2.73，0 条 z>4，`6375f943cb` |
| CARC P100，torch 2.5.1 | 同上 | 同上 |
| CARC V100，torch 2.5.1 | 同上 | 同上 |
| CARC A100，torch 2.5.1 | 同上 | 同上 |

平均 green 比例：水印文本 0.627，人写文本 0.237（γ = 0.25）。人写文本略低于 γ，是因为常见词组（如 "of the"）每次出现的 green/red 结果都一样，会放大这把 key 的偶然偏差，C 组做去重分析时要关注。
