# TODO

## 现在（midterm 前）：先把 KGW pipeline 跑通
- [x] A1：`prepare_data.py` 已跑完，`data/prompts.jsonl` 和 `data/human.jsonl` 已提交到仓库
- [ ] A2：在 CARC 上跑 `generate.py`（先跑 10 条 pilot，再跑全部 200 条）
- [ ] B：改写攻击脚本（改写 25% / 50% / 100% 的句子）+ 语义相似度过滤
- [ ] C：逐 token 的 green flags → 固定长度 z-test / peeking z-test / e-process
- [ ] C：e-process 只对第一次出现的 (前一个 token, 当前 token) 计分（去重，几行代码）

## 以后再做
- [ ] 第二种水印：Gumbel-max / EXP（e-process 论文 Su, Wang & Zhao 2026, arXiv 2602.14286 主要针对的就是它）。看一下 MarkLLM 的 EXP 实现，确认能拿到每个 token 的统计量
- [ ] 对比去重前后人写文本的 FPR，作为"去重为什么必要"的证据
- [ ] 扩大规模，加 Unigram / SynthID，加入其他领域的数据（ELI5、WritingPrompts）
- [ ] 根据 `logs/throughput.jsonl` 估算 GPU 小时和费用（回应 TA）
- [ ] 在 related work 中引用 Kirchenbauer 2024（arXiv 2306.04634），说明我们和它的区别
