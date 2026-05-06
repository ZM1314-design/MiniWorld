# 0.4B基模 农业语料继续预训练（DAPT / Continued Pretraining）

这个仓库用于在 **单卡 RTX 4090（24GB）** 上，对本地的 **0.4B 基座模型**进行**继续预训练（Domain-Adaptive Pretraining, DAPT）**，语料为农业领域纯文本（多 `.txt` 文件，UTF-8）。

训练方式是 **continued pretraining（非 SFT 指令微调）**：直接对原始领域文本做语言模型训练。

## 你当前的本地路径约定（本项目默认）

- **模型目录**：`E:\models\MiniWorld\`
- **原始语料目录（多 txt）**：`E:\data\agricultur_txt\`
- **生成的 JSONL**：`E:\data\agri_pretrain.jsonl`
- **训练输出**：`E:\out\qwen-0.5b-agri-dapt\`

如果你想换盘/换目录，直接改 `configs/qwen_0_5b_agri_dapt_4090.yml` 里的路径即可。

## 目录与关键文件

- `scripts/txt_folder_to_jsonl.py`：把“多 txt 文件夹”转换成 Axolotl pretraining 可用的 JSONL（每行 `{"text": ...}`）
- `configs/qwen_0_5b_agri_dapt_4090.yml`：4090 单卡 continued pretraining 的默认配置（seq_len=2048、bf16、gradient checkpointing 等）

## 1) 数据准备：txt → jsonl

Axolotl 的非流式 continued pretraining 推荐 JSONL：每行一个样本 `{"text": "..."}`。

在 PowerShell 里执行：

```powershell
cd "e:\继续预训练\axolotl-main"
python scripts\txt_folder_to_jsonl.py --txt_dir "E:\data\agricultur_txt" --out_jsonl "E:\data\agri_pretrain.jsonl"
```

脚本会递归读取 `txt_dir` 下所有 `.txt`，并写入到 `out_jsonl`。

## 2) 训练：continued pretraining（completion）

本项目用 `datasets: type: completion` 做“非流式”继续预训练（适合单机单卡 + 中小规模语料）。

```bash
axolotl train configs/qwen_0_5b_agri_dapt_4090.yml
```

配置要点（已在 YAML 中预设）：

- `sequence_len: 2048`
- `bf16: true`（4090 支持）
- `micro_batch_size: 1` + `gradient_accumulation_steps: 16`
- `gradient_checkpointing: true`（省显存）
- `sample_packing: true`（提升吞吐）
- `max_steps: 5000`（可按语料规模调整）

## 3) 断点续训（Resume）

训练过程中会在 `output_dir` 下生成 `checkpoint-*`。保持 `output_dir` 不变即可从已有 checkpoint 继续。

## Windows/WSL2 提示

- **建议优先用 WSL2 跑训练**：Windows 原生环境下某些依赖（尤其是编译相关、bitsandbytes/attention 优化组件）更容易踩坑；WSL2 通常更稳定省时。
- 本项目配置默认 `flash_attention: false`，等环境稳定后再开启优化项。

## 致谢

训练框架基于上游 Axolotl：`https://github.com/axolotl-ai-cloud/axolotl`。

