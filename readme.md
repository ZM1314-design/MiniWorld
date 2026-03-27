# miniworld: 从零训练个人千万级参数大模型（最小闭环）

这是一个面向个人开发者的训练任务仓库：  
**不依赖外部大模型 API**，在本地完成「Tokenizer -> 预训练 -> SFT -> 推理」全流程，得到你自己的模型权重。

> 目标优先级：先跑通，再提效果。  
> 你最终会得到一个约 **2600 万参数（26M）** 级别的个人模型权重（`hidden_size=512, layers=8` 配置）。

---

## 1. 你将得到什么

- 本地可训练、可推理的 `MiniMind` 小模型
- 一套完整训练产物：
  - `model/tokenizer.json`
  - `out/pretrain_tiny_512.pth`
  - `out/full_sft_tiny_512.pth`
- 一个可运行的对话入口：
  - `eval_llm.py`

---

## 2. 从零开始的最快步骤（Windows）

### Step 0: 安装 pixi

```powershell
powershell -ExecutionPolicy ByPass -c "irm -useb https://pixi.sh/install.ps1 | iex"
```

重新打开 PowerShell 验证：

```powershell
where.exe pixi
pixi --version
```

### Step 1: 安装环境

```powershell
cd d:\minimind-learn-master
pixi install
```

### Step 2:（推荐）安装 GPU 版 PyTorch

> 如果网络较慢，可先 CPU 跑通流程，后续再补 GPU。

```powershell
pixi run python -m pip install --no-cache-dir `
  --index-url https://download.pytorch.org/whl/cu124 `
  "torch==2.6.0+cu124" "torchvision==0.21.0+cu124"
```

验证：

```powershell
pixi run python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

---

## 3. 最快生成 tiny 数据集（离线，不用下载大语料）

在项目根目录执行：

```powershell
cd d:\minimind-learn-master
$ds="dataset"
mkdir $ds -Force | Out-Null

# 预训练 tiny 数据
$pre="$ds\pretrain_tiny.jsonl"
Remove-Item $pre -Force -ErrorAction SilentlyContinue
1..80 | ForEach-Object {
  (@{text="第$_条预训练样本：用于从零跑通 MiniMind 训练闭环。"} | ConvertTo-Json -Compress)
} | Set-Content $pre -Encoding utf8

# SFT tiny 数据
$sft="$ds\sft_tiny.jsonl"
Remove-Item $sft -Force -ErrorAction SilentlyContinue
1..40 | ForEach-Object {
  $a = $_ % 10; if($a -eq 0){$a=1}
  $b = ($_*7) % 10; if($b -eq 0){$b=2}
  $conv = @(
    @{role="system"; content="你是一个简洁友好的助手。"},
    @{role="user"; content="请计算：$a+$b=?"},
    @{role="assistant"; content="$($a+$b)"}
  )
  @{conversations=$conv} | ConvertTo-Json -Compress
} | Set-Content $sft -Encoding utf8
```

---

## 4. 训练流程（最小可跑通）

### 4.1 Tokenizer（可选）

如果 `model/tokenizer.json` 已存在，可跳过。

```powershell
pixi run python scripts/train_tokenizer.py
```

### 4.2 预训练（Pretrain）

```powershell
pixi run python trainer/train_pretrain.py `
  --data_path dataset/pretrain_tiny.jsonl `
  --save_dir out --save_weight pretrain_tiny `
  --epochs 1 --batch_size 2 --learning_rate 1e-4 `
  --dtype float16 --accumulation_steps 1 `
  --hidden_size 512 --num_hidden_layers 8 --use_moe 0 `
  --max_seq_len 128 --from_weight none --from_resume 0 `
  --num_workers 0 --log_interval 5 --save_interval 50
```

### 4.3 SFT（监督微调）

```powershell
pixi run python trainer/train_full_sft.py `
  --data_path dataset/sft_tiny.jsonl `
  --save_dir out --save_weight full_sft_tiny `
  --epochs 1 --batch_size 1 --learning_rate 5e-6 `
  --dtype float16 --accumulation_steps 1 `
  --hidden_size 512 --num_hidden_layers 8 --use_moe 0 `
  --max_seq_len 128 --from_weight pretrain_tiny --from_resume 0 `
  --num_workers 0
```

### 4.4 推理验证（你自己的模型）

```powershell
pixi run python eval_llm.py `
  --load_from model --save_dir out `
  --weight full_sft_tiny `
  --hidden_size 512 --num_hidden_layers 8 --use_moe 0 `
  --max_new_tokens 128
```

---

## 5. 最容易出现的操作失误（重点）

1. **装成 CPU 版 torch**
   - 现象：`torch.__version__` 显示 `+cpu`，`CUDA available: False`
   - 修复：用 `cu124` 源重装 GPU 版 wheel

2. **pixi 在 Windows 平台不支持**
   - 现象：`unsupported-platform win-64`
   - 修复：`pixi.toml` 的 `platforms` 加上 `win-64`

3. **在系统 Python 里装包，而不是 pixi 环境**
   - 现象：装了 GPU torch，但 `pixi run check-gpu` 还是 CPU
   - 修复：所有安装命令都用 `pixi run python -m pip ...`

4. **数据字段格式不匹配**
   - `PretrainDataset` 需要：`{"text": "..."}`
   - `SFTDataset` 需要：`{"conversations":[{"role":"user","content":"..."}, ...]}`

5. **显存不足 OOM**
   - 修复顺序：先减 `batch_size`，再减 `max_seq_len`，再增 `accumulation_steps`

---

## 6. 时间与规模预期

- 最小闭环总时长（不含大文件下载）：约 **10~45 分钟**
- 参数规模（`hidden_size=512, layers=8`）：约 **26M（2600 万）**
- 这属于「可训练、可验证、可迭代」的个人模型起点

---

## 7. 进阶路线（跑通后）

- 把 tiny 数据替换为更大高质量语料
- 提高 `max_seq_len`、`epochs`、`layers`
- 增加 LoRA / DPO / PPO / 蒸馏 / MoE 实验
- 逐步从“能跑通”走向“能用且好用”