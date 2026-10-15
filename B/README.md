# B 组：改写攻击

B 组文件统一放在本目录：

```text
B/
  attack.py                    三档嵌套改写攻击
  sanity_attack.py             B9：原始水印及攻击后的检测
  check_attacks.py             字段、三档嵌套、重新分词检查
  apply_attack_review.py       应用人工复核决定
  manual_review.py             随机抽样、对照阅读并交互保存 valid/reason
  run_attack.sh                GPU 攻击入口
  setup_attack_env.sh          创建本机环境
  requirements-attack.txt      安装依赖
  reports/B_attack_results.md  最新结果报告
```

数据、日志和配置统一放在仓库根目录的 `data/`、`logs/`、`configs/`。原始输入来自仓库根目录 `data/wm.jsonl`、`data/human.jsonl`。检测共享根目录的 `kgw_cpu.py`、`configs/KGW_g025.json` 和 `MarkLLM/`。

## 运行

以下命令从仓库根目录执行：

```bash
# 首次配置；已有环境可跳过
bash B/setup_attack_env.sh
conda activate "$PWD/.conda/b-attack"

# 小规模试跑：每种来源前 5 篇，共 30 条输出
bash B/run_attack.sh --limit 5 --out-dir data/attack_pilot
# 全量：200 篇水印 + 前 200 篇人类文本，各三档，共 1200 条
bash B/run_attack.sh

# 生成结束后应用本次人工复核决定
python B/apply_attack_review.py
# 检查全部攻击数据及 token 对应关系
python B/check_attacks.py --verify-tokens
# B9：检测原始水印和攻击后水印
python B/sanity_attack.py --limit 200 --device cpu
# 改写后人类文本的检测对照
python verify_kgw.py --device cpu data/attacked_human.jsonl
```

Python 脚本的默认路径按脚本位置解析，所以在 `B/` 内运行 `python check_attacks.py` 等命令也可以。自定义相对路径参数以当前工作目录为准；`run_attack.sh` 会先切到仓库根目录，因此传给启动脚本的相对路径以仓库根目录为准。

环境继续使用根目录 `.conda/b-attack`，模型缓存共享根目录 `.cache/huggingface`，无需重新下载。B 组 Python 脚本默认设置该缓存路径，启动脚本也会设置；缓存齐全后可用 `HF_HUB_OFFLINE=1`。本机 RTX 5090 使用 PyTorch 2.8 / CUDA 12.8；CARC P100 请用根目录 `setup_env.sh` 的 torch 2.5.1/cu121。

## 攻击和质量规则

默认攻击模型 Qwen2.5-1.5B-Instruct，不加水印，temperature=0.7，seed=123，每批 32 句。检测用 token_ids 由 Qwen2.5-1.5B base tokenizer 生成，保留原始 prompt_ids。

逐句改写一次，再生成 25% / 50% / 100% 三档嵌套版本。保留空白和末尾半句；句子选择数为 ceil(完整句子数 × 比例)。清洗前缀后要求非空、单行、完整结尾、词数比例 0.4–2、数字字符串不变、逐句相似度至少 0.75；失败重试一次，仍失败或与原文相同则保留原句。

全文相似度用 all-MiniLM-L6-v2 分块编码、均值池化后计算，避免只比较开头。valid 要求相似度至少 0.75、重新分词长度比在 [0.5, 2]、至少成功改写一句。actual_attack_frac 以完整句子为分母，只计实际成功改写的句子。token_edit_ratio 是 1 − SequenceMatcher 的 token 匹配比例（关闭 autojunk），不是 Levenshtein 距离。

输出按 `(id, attack_frac)` 断点续跑，逐条 flush。攻击参数变化时拒绝追加；路径变化不视为算法变化。输入文本更新时应先移走旧输出，避免同 ID 被错误跳过。不要让多个进程同时写同一文件，也不要在生成时应用人工复核。

当前结果对应 A 组 2026-10-15 的 CPU 绿名单水印版本。人类原文未变，沿用已完成的人类攻击。新版前五篇水印的人工复核决定见 `configs/attack_manual_review.json`，只适用于本次输入和固定参数；换数据或参数后需重新复核。自动相似度不能证明所有事实不变，其余样本仍可能存在语义漂移。

## 检测和交付

交互人工复核（无需 GPU，从仓库根目录运行）：

```bash
python B/manual_review.py                              # 随机抽 1 条
python B/manual_review.py --source wm --count 5         # 抽 5 条水印攻击
python B/manual_review.py --source human --attack-frac 1.0
```

脚本只对照显示实际改写的句子，未改写句子隐藏；相邻改写句子合并为一个片段，避免改写后拆句或合句造成错位。输入 true/false 及 reason，决定仍针对该条攻击记录整体；f 查看全文，s 跳过，q 退出。默认跳过已有复核记录。每条决定立即写入根目录 `configs/attack_manual_review.json`，并同步更新对应攻击数据，无需再次运行 apply_attack_review.py。人工通过不会覆盖自动过滤失败。样本有效性改变后需重新汇总检测统计及报告。

sanity_attack.py 使用 A 组的 patch_kgw_cpu，词表大小 151936，并以 prompt 最后一个 token 为前缀。原始水印未全部超过 z=4 时返回状态码 2；summary_valid 只汇总有效记录。

攻击文本不补齐、不截断成 600 token；C 组按实际长度取前缀，只使用 valid=true。固定长度、去重、统一误报率、peeking 和 e-process 的正式分析由 C 组完成。

最新结果见 [B 组结果报告](reports/B_attack_results.md)。后续扩展：7B/DIPPER、多轮改写、回译。
