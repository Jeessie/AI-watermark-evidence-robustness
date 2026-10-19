# TODO

## 现在（midterm 前）：先把 KGW pipeline 跑通
- [x] A1：`prepare_data.py` 已跑完，`data/prompts.jsonl` 和 `data/human.jsonl` 已提交到仓库
- [x] A2：`data/wm.jsonl` 已生成（200 条 × 600 token）。10/15 改成 CPU green list 后重新生成，任何机器检测结果一致（见 README「已经验证过的」）
- [x] B：攻击数据两条臂
  - 主臂（B 组）：`B/attack.py` + `data/attacked_wm.jsonl` / `data/attacked_human.jsonl`，
    200 篇/档，语义相似度 ≥0.75 + 人工复核，token_kept 0.883 / 0.774 / 0.553
  - 强臂（我）：`c_eprocess/attack_rewrite.py` 走 dtens API `deepseek-v4.1-flash`，20 篇/档，
    token_kept 0.832 / 0.685 / 0.404
  - 攻击强度客观指标：`c_eprocess/attack_strength.py`（两臂统一 token_kept / unigram_new 口径）
- [x] C：逐 token 的 green flags → 固定长度 z-test / peeking z-test / e-process
  - [x] e-process 代码：`c_eprocess/eprocess.py`（论文 §4 的四条构造 + KGW 专用 bet 版）
  - [x] C1：`data/flags_{wm,human}.jsonl` 已在本地跑出，`flags_md5` 与 README 一致（5d98d773dd / 6375f943cb）
  - [x] C4：500 条人写文本实测 FPR —— 5 种 e-process 全部 ≤ 0.008（Wilson 上界 ≤ 0.020），peeking z 0.160（α=0.05 时 0.434）
  - [x] C6/C7：长度曲线、停止时间、攻击热图 + 5 张图（`results/real/fig*.png`，每个攻击臂一张热图）
  - [x] 新增强检测器 `betmix`（betting e-process 对 λ 取混合）：中位停止 17 token（avg 31），
        100% 强改写时 0.49 vs avg 0.36；并给出"固定 λ 上界 0.77"说明自适应的代价
  - [x] C4 扩展：多 key 扫描做了 6 把（`results/keys/summary.csv`）；计划 20 把，未跑满
- [x] C：e-process 只对第一次出现的 (前一个 token, 当前 token) 计分（去重，`first_occurrence` + `eprocess(dedup=True)`）

## 以后再做
- [ ] 第二种水印：Gumbel-max / EXP（e-process 论文 Su, Wang & Zhao 2026, arXiv 2602.14286 主要针对的就是它）。看一下 MarkLLM 的 EXP 实现，确认能拿到每个 token 的统计量
- [x] 对比去重前后（`run_eval.py` 的 `dedup` 列）：人写文本 FPR 都是 0.008（没区别）；强改写文本检出率 0.49（去重）vs 0.62（不去重）
      → 去重是理论必需、实测有代价，默认开 + 报敏感性分析（`docs/C_report.md` §6.3）
- [ ] 多 key 扫描扩到 20 把并画每把 key 的 FPR 分布（`bet` 对 γ 偏置最敏感，值得单独看）
- [ ] 扩大规模，加 Unigram / SynthID，加入其他领域的数据（ELI5、WritingPrompts）
- [ ] 根据 `logs/throughput.jsonl` 估算 GPU 小时和费用（回应 TA）
- [ ] 在 related work 中引用 Kirchenbauer 2024（arXiv 2306.04634），说明我们和它的区别
