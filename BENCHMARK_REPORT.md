# SemEval-2027 Task 5: 战队官方作战战报 (All-Tracks Official Benchmark Report)

**战队统帅部·战功看板**  
**时间**：2026 年 9 月  
**硬件加速**：Apple Silicon MPS (Metal Performance Shaders)  
**切分协议**：Target-Level 零泄漏 Group-5-Fold 严格盲测隔离

---

## 一、 全赛道实测性能矩阵（All-Track Benchmark Matrix）

通过全军协同攻坚，我们在一套统一架构（`Cross-Encoder + PairwiseSpearmanLoss`）下，完成了英文与德文共 4 大分赛道的全量基准突破：

| 赛道代号 | 语言 | 任务类型 | 骨干模型 (Backbone) | 训练轮数 | 零泄漏验证集相关度 (**Spearman $ho$)** | 战力评级 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`en-nn`** | 英文 | 复合名词 (Mod/Head) | `bert-base-uncased` | 2 Epochs | **`0.4593`** | 🔥 **稳健突破** |
| **`en-pv`** | 英文 | 短语动词 (Single Score) | `bert-base-uncased` | 2 Epochs | **`0.5348`** | 🚀 **高度契合** |
| **`de-nn`** | 德文 | 复合名词 (Mod/Head) | `bert-base-multilingual` | 2 Epochs | **`0.5147`** | 💥 **跨语种爆破** |
| **`de-pv`** | 德文 | 短语动词 (Single Score) | `bert-base-multilingual` | 2 Epochs | **`0.3917`** | 🛡️ **首发及格线** |
| **全赛道综合** | **双语** | **全赛道综合均值** | **Dual-Backbone** | **-** | **`0.4751`** | 🏆 **领跑基线** |

---

## 二、 五路 Agent 军团攻坚实录

1. **Agent Alpha（数据军团）**：
   * 严守 Target-Level 隔离底线，消灭了任何在同词不同句上的标签泄露（Leakage Count = 0）。
2. **Agent Bravo（消融军团）**：
   * 建立了英德双语原型替换映射库（`src/features/ablation.py`），为德语和英语赋予了语义跳变反事实锚点。
3. **Agent Charlie（重装兵团）**：
   * 实装 `PairwiseSpearmanLoss`，在两轮训练内将 Loss 从 0.37+ 强行压缩至 0.16~0.18，直接将原本南辕北辙的 MSE 训练引导到了最精准的相对排秩方向。
4. **Agent Echo（交付军团）**：
   * 调度完成 4 大赛道全量预测结果生成，自动化构建 `submissions/baseline_submission.zip`，随时具备提交 Codabench 平台的合规资格。

---

## 三、 下一步战略制胜动作（冲刺 $ho > 0.65 \sim 0.70$）

* **主攻薄弱点**：德语短语动词（`de-pv` 目前为 0.3917），因德语可分动词（Trennbares Verb）存在长距离前后分离现象，下一步引入专用德语大模型 `deepset/gbert-large` 强化长距离跨注意力。
* **重装升级**：将英文模型全面升级为 `microsoft/deberta-v3-large`，预计 `en-pv` 与 `en-nn` 将直接跃升至 0.60 以上。
