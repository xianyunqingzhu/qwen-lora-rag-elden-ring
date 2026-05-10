# Qwen 大模型工程实践

# 艾尔登法环 AI 攻略助手
## 基于 Qwen2.5-7B + LoRA + RAG 的本地知识问答系统

本项目实现了一个能够回答《艾尔登法环》游戏攻略问题的 AI 助手。

整个系统基于：

- Qwen2.5-7B 大语言模型
- LoRA 微调
- RAG（检索增强生成）
- 本地向量知识库
- FastAPI 后端
- Web 前端交互页面

用户提问后，系统会先从本地知识库中检索相关游戏资料，再交给大模型生成最终答案。

---

# 整体架构

```text
用户问题
   ↓
前端页面（HTML + JS）
   ↓
FastAPI 后端接口
   ↓
RAG 检索知识库
   ↓
拼接 Prompt
   ↓
Qwen2.5 大模型生成答案
   ↓
返回网页显示
```

---

# 一、环境与服务器配置

本项目在autodl租用服务器实现，使用配置：

```text
GPU：NVIDIA GeForce RTX 5090
显存：32GB
CUDA：13.0
```

## 环境与依赖：
```bash
conda create -n qwen python=3.11
```
```bash
pip install transformers
pip install vllm
pip install fastapi uvicorn openai
```

这些库的作用：

| 库名 | 作用 |
|---|---|
| transformers | HuggingFace 模型加载库 |
| vllm | 高性能大模型推理框架 |
| fastapi | 构建后端 API |
| uvicorn | FastAPI 的运行服务器 |
| openai | 使用 OpenAI 风格接口调用模型 |

---

# 二、Qwen 模型部署

## 为什么选择 vLLM？

传统方式通常使用 `transformers` 直接推理。

但对于大模型来说：

- 推理速度较慢
- 显存利用率不高
- 并发能力较差

因此本项目使用 `vLLM` 作为模型推理框架。

`vLLM` 是一个专门用于大语言模型部署的高性能推理框架。

它的优势包括：

- 推理速度更快
- 显存利用率更高
- 支持 OpenAI API 格式
- 更适合部署聊天系统
- 支持高并发请求

相比传统推理方式，
vLLM 更适合作为 AI Web 项目的后端服务。



## 模型

先将模型下载到本地服务器：

```bash
hf download Qwen/Qwen2.5-7B-Instruct \
  --local-dir /autodl-fs/data/models/Qwen2.5-7B-Instruct
```

## 启动 vLLM 服务

```bash
vllm serve /autodl-fs/data/models/Qwen2.5-7B-Instruct \
  --served-model-name Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000
```

参数说明：

| 参数 | 作用 |
|---|---|
| --served-model-name | API 中显示的模型名称 |
| --host 0.0.0.0 | 允许外部访问 |
| --port 8000 | 服务端口 |



终端出现：

```text
Application startup complete
```

说明模型服务已经成功运行。

---

# 三、测试模型

## 查看模型列表

新开一个终端：

```bash
curl http://localhost:8000/v1/models
```

如果返回：

```json
{
  "data": [
    {
      "id": "Qwen2.5-7B-Instruct"
    }
  ]
}
```

说明模型 API 已成功启动。


## 测试聊天功能

```bash
curl http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
  "model": "Qwen2.5-7B-Instruct",
  "messages": [
    {
      "role": "user",
      "content": "什么是LoRA微调"
    }
  ]
}'
```

使用 OpenAI 风格接口。

这意味着：

后续前端、RAG、FastAPI 都可以直接按照 OpenAI API 的方式调用模型。

这样能够极大降低开发难度。

# 四、OpenAI API 接口能力

## vLLM 提供的接口

当 Qwen 模型通过 vLLM 启动后，vLLM 会自动提供一组兼容 OpenAI 格式的 API 接口。

常见接口包括：

```text
/v1/chat/completions
/v1/models
/v1/completions
```

这意味着：

```text
本地服务器 ≈ 一个本地版 OpenAI 服务
```

也就是说，我们可以像调用 OpenAI API 一样调用自己部署在服务器上的 Qwen 模型。

目前已经能够实现：

- 本地部署大语言模型
- 通过 HTTP API 调用模型
- 进行文本生成与问答
- 为后续 Web 页面、RAG 系统提供模型能力

---

# 五、FastAPI 封装阶段

## 为什么还需要 FastAPI？

虽然 vLLM 已经提供了 OpenAI 风格接口，例如：

```text
http://localhost:8000/v1/chat/completions
```

但是这个接口更偏底层，调用时需要传入完整的 OpenAI 请求格式。

例如需要写：

```json
{
  "model": "Qwen2.5-7B-Instruct",
  "messages": [
    {
      "role": "user",
      "content": "什么是LoRA微调？"
    }
  ]
}
```

如果直接让前端页面调用 vLLM，会有几个问题：

- 前端请求格式复杂
- 后续不方便加入 RAG 检索逻辑
- 不方便统一管理 prompt
- 不方便控制模型参数
- 不方便隐藏后端实现细节

因此，本项目额外使用 FastAPI 封装一层自己的后端接口。

这样前端只需要请求：

```text
POST /chat
```

并传入一个简单的问题即可。

## FastAPI 在项目中的作用

FastAPI 相当于整个系统的“中间控制层”。

它负责：

- 接收前端传来的用户问题
- 整理成大模型需要的 messages 格式
- 调用 vLLM 提供的接口
- 获取 Qwen 模型生成的结果
- 将结果返回给前端页面

整体流程如下：

```text
用户输入问题
    ↓
前端页面发送请求
    ↓
FastAPI 接收问题
    ↓
FastAPI 调用 vLLM
    ↓
vLLM 调用 Qwen 模型
    ↓
FastAPI 返回模型回答
    ↓
前端页面展示结果
```

---

## 实现接口：POST /chat

本项目在 FastAPI 中实现了一个 `/chat` 接口。

接口形式：

```text
POST /chat
```

请求数据示例：

```json
{
  "question": "什么是LoRA微调？"
}
```

返回数据示例：

```json
{
  "answer": "LoRA 是一种参数高效微调方法..."
}
```

这样设计后，前端调用会非常简单，只需要把用户输入的问题发送给 `/chat` 即可。

---

## main.py 的作用说明
**注意** ：上传项目中的main.py是最终内容，额外内容后续会说。

`main.py` 是 FastAPI 后端服务的核心文件。

它主要完成以下几件事：

### 1. 创建 FastAPI 应用

```python
app = FastAPI(title="Qwen Elden Ring RAG API")
```

这行代码用于创建一个 FastAPI 后端应用。

后续所有接口，例如 `/chat`，都会挂载到这个 `app` 上。


### 2. 定义请求数据格式

通常会使用 Pydantic 定义用户请求的数据结构。

例如：

```python
class ChatRequest(BaseModel):
    question: str
```

这表示 `/chat` 接口接收的数据中必须包含一个 `question` 字段。

也就是说，前端需要发送：

```json
{
  "question": "用户的问题"
}
```

这样做的好处是：

- 请求格式更清晰
- FastAPI 会自动校验数据
- 如果缺少字段，会自动返回错误提示

### 3. 实现 `/chat` 接口

核心接口通常类似：

```python
@app.post("/chat")
def chat(req: ChatRequest):
    ...
```

这表示：

当用户向 `/chat` 发送 POST 请求时，FastAPI 会执行这个函数。

其中：

- `@app.post("/chat")` 表示定义一个 POST 接口
- `req` 表示前端传来的请求数据
- `req.question` 就是用户输入的问题

### 4. 调用 vLLM 模型服务

在 `/chat` 接口内部，FastAPI 会把用户问题转换成 vLLM 需要的格式：

```json
(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "你是一个专业、简洁的中文AI助手。"
            },
            {
                "role": "user",
                "content": req.question
            }
        ],
        temperature=req.temperature,
        max_tokens=req.max_tokens
    )
```

然后请求：

```text
http://localhost:8000/v1/chat/completions
```


### 5. 返回模型回答

vLLM 返回结果后，FastAPI 会从返回数据中提取模型回答，并整理成简单格式：

```json
{
  "answer": "模型生成的回答"
}
```

这样前端就不需要关心 vLLM 的复杂返回结构，只需要读取 `answer` 字段即可。

---

## 启动 FastAPI 服务

进入后端项目目录：

```bash
cd /autodl-fs/data/qwen_api
```

启动 FastAPI：

```bash
uvicorn main:app --host 0.0.0.0 --port 9000
```

命令说明：

| 参数 | 作用 |
|---|---|
| uvicorn | 启动 FastAPI 的服务器 |
| main:app | 表示运行 main.py 中的 app 对象 |
| --host 0.0.0.0 | 允许外部访问 |
| --port 9000 | 指定 FastAPI 服务端口 |

启动成功后，FastAPI 服务会运行在：

```text
http://服务器IP:9000
```

---

## 测试 /chat 接口

新开一个终端，执行：

```bash
curl http://localhost:9000/chat \
-H "Content-Type: application/json" \
-d '{
  "question": "什么是LoRA微调？"
}'
```

如果能够返回模型回答，说明：

- vLLM 模型服务正常运行
- FastAPI 后端服务正常运行
- FastAPI 可以成功调用 Qwen 模型
---
# Lora

# 六、LLaMA-Factory 安装

## 什么是 LLaMA-Factory？

是一个用于大模型微调的开源框架。

它的特点：

- 支持 LoRA 微调
- 支持 Qwen / Llama / ChatGLM 等模型
- 配置简单
- 对新手友好
- 不需要自己手写复杂训练代码

相比从零编写 PyTorch 训练脚本，LLaMA-Factory 更适合个人项目快速实践

### 下载与安装

```text
本地下载 ZIP → 上传服务器 → 解压
```
进入目录：

```bash
cd ~/autodl-fs/LlamaFactory-main
```

安装：

```bash
pip install -e ".[torch,metrics]"
```

参数说明：

| 参数 | 作用 |
|---|---|
| -e | 以开发模式安装 |
| torch | 安装 PyTorch 相关依赖 |
| metrics | 安装训练评估依赖 |

---

# 七、数据集构建


LoRA 微调本质上是：

```text
让模型学习一种新的输出方式
```

因此需要提前准备：

- 问答数据
- 攻略数据
- NPC 信息
- Boss 攻略

---

## 初始数据格式

最开始使用 CSV 格式整理数据：

```csv
instruction,input,output
```

这种格式简单直观，
适合人工编辑。

我创建了一个脚本`txt_to_csv.py`，只需要收集数据将一个小主题内容写成txt文档，可调用deepseek api实现转换成CSV格式内容。需要填写api key。

---

## 数据内容

数据包括：

- 游戏配置
- Boss 攻略
- 新手建议
- Build 配装
- 战技
- NPC
- 地图
- 武器
- 属性加点

目标是让模型逐渐学习：

```text
“艾尔登法环攻略助手”的回答风格
```

---

## 为什么还要转换格式？

虽然 CSV 易于编辑，

但大模型训练通常使用：

```text
JSON / Alpaca 格式
```

因此需要进行数据转换。

---

## 编写 convert.py

项目中编写：

```text
convert.py
```

用于：

```text
CSV → Alpaca JSON
```

---

## 什么是 Alpaca 格式？

Alpaca 是目前最常见的大模型指令微调格式之一。

典型结构：

```json
{
  "instruction": "你是一个专业的艾尔登法环攻略助手",
  "input": "女武神怎么打",
  "output": "..."
}
```

字段说明：

| 字段 | 作用 |
|---|---|
| instruction | 系统指令 |
| input | 用户问题 |
| output | 标准答案 |

---

## 运行数据转换

```bash
python convert.py \
  --input 艾尔登法环.csv \
  --output elden_ring.json
```

转换完成后：

```text
elden_ring.json
```

即可用于 LoRA 微调。

---

# 八、LoRA 训练配置

## dataset_info.json

在开始训练前，
需要告诉 LLaMA-Factory：

```text
数据集在哪里
数据格式是什么
字段如何对应
```

配置文件路径：

```text
/root/autodl-fs/data/dataset_info.json
```

内容：

```json
{
  "elden_ring": {
    "file_name": "elden_ring.json",
    "formatting": "alpaca",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  }
}
```

---

## 配置含义

字段映射：

| 字段 | 含义 |
|---|---|
| prompt | 系统提示词 |
| query | 用户输入 |
| response | 模型目标输出 |

也就是说：

```text
instruction → prompt
input → query
output → response
```

---

# 九、训练 YAML 配置

## 为什么使用 YAML？

LLaMA-Factory 使用 YAML 管理训练参数。

优点：

- 配置清晰
- 不需要手写复杂命令
- 更容易复现训练过程

配置文件路径：

```text
/root/autodl-fs/LlamaFactory-main/examples/train_lora/qwen2_5_7b_elden_lora.yaml
```

---

## 核心参数解释

```yaml
model_name_or_path: /root/autodl-fs/models/Qwen2.5-7B-Instruct

finetuning_type: lora

lora_rank: 8
lora_alpha: 16
lora_dropout: 0.05

cutoff_len: 2048

learning_rate: 5e-5

num_train_epochs: 3

bf16: true
```

---

### model_name_or_path

指定基础模型路径。

也就是：

```text
Qwen2.5-7B-Instruct
```

---

### finetuning_type

```yaml
finetuning_type: lora
```

表示：

```text
使用 LoRA 微调
```

而不是全参数训练。

---

### LoRA 参数

```yaml
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.05
```

作用：

| 参数 | 含义 |
|---|---|
| lora_rank | LoRA 低秩矩阵大小 |
| lora_alpha | LoRA 缩放系数 |
| lora_dropout | 防止过拟合 |

这些参数会影响：

- 显存占用
- 学习能力
- 训练稳定性

---

### cutoff_len

```yaml
cutoff_len: 2048
```

表示：

```text
模型一次最多读取 2048 token
```

数值越大：

- 能读取更长文本
- 但显存占用也会增加
---

### num_train_epochs

```yaml
num_train_epochs: 3
```

表示：

```text
完整训练数据重复学习 3 次
```

---

### bf16

```yaml
bf16: true
```

表示使用：

```text
bfloat16 混合精度训练
```

好处：

- 显存占用更低
- 训练速度更快

---

# 十、开始 LoRA 训练

## 启动训练

```bash
cd /root/autodl-fs/LlamaFactory-main

conda activate qwen

llamafactory-cli train \
examples/train_lora/qwen2_5_7b_elden_lora.yaml
```

训练开始后：

- GPU 会持续占用
- Loss 会不断变化
- 模型会逐渐学习游戏问答风格

# 十二、LoRA 微调效果分析
测试内容：pdf链接，第一个回答是lora训练后的，第二个回答是原始模型。
## 成功部分

经过 LoRA 微调后：

模型已经学会：

- 攻略助手语气
- 结构化回答
- 游戏问答风格
- 分析型输出格式

例如：

```text
【问题分析】
【推荐方案】
【注意事项】
```

相比原版模型：

```text
回答更加像“游戏攻略助手”
```

---

## 当前问题

虽然风格学习成功，

但模型仍然存在：

```text
知识准确性不足
```

例如：

- 魔兽世界内容串台
- Boss 信息错误
- NPC 关系错误
- 地图位置错误

---

# 十三、LoRA 的关键认知

在实践后，一个非常重要的结论是：

```text
LoRA 更擅长学习：
- 风格
- 语气
- 输出格式
- 角色设定
```

但：

```text
LoRA 并不擅长注入大量事实知识
```

也就是说：

```text
LoRA ≠ 真正的知识库
```

这也是后续需要引入：

```text
RAG（检索增强生成）
```

的根本原因。

---

# 十四、合并 LoRA 模型

## 为什么需要 merge？

LoRA 训练完成后：

得到的并不是完整模型。

而是：

```text
“增量参数”
```

因此需要：

```text
基础模型 + LoRA 参数
→ 合并成完整模型
```

这样后续部署会更加方便。

---

## 创建 merge YAML

```bash
cat > /root/autodl-fs/LlamaFactory-main/examples/merge_lora.yaml <<'EOF'
model_name_or_path: /root/autodl-fs/models/Qwen2.5-7B-Instruct

adapter_name_or_path: /root/autodl-fs/saves/qwen2.5-7b/elden-ring-lora

template: qwen
finetuning_type: lora

export_dir: /root/autodl-fs/models/qwen2.5-7b-elden-ring-merged

export_size: 2
export_device: cuda
export_legacy_format: false
EOF
```

---

## 开始合并

```bash
cd /root/autodl-fs/LlamaFactory-main

llamafactory-cli export examples/merge_lora.yaml
```

合并完成后得到：

```text
/root/autodl-fs/models/qwen2.5-7b-elden-ring-merged
```

---

# 十五、部署 merged model

## 启动 vLLM

```bash
conda activate qwen

vllm serve \
/root/autodl-fs/models/qwen2.5-7b-elden-ring-merged \
--host 0.0.0.0 \
--port 8000 \
--dtype bfloat16
```

这里的：

```bash
--dtype bfloat16
```

表示：

```text
使用 bfloat16 精度推理
```

这样能够降低显存占用。

---

## 验证模型是否启动成功

```bash
curl http://127.0.0.1:8000/v1/models
```

如果返回模型信息：

说明 merged model 已成功部署。

---

## 测试聊天

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
  "model": "/root/autodl-fs/models/qwen2.5-7b-elden-ring-merged",
  "messages": [
    {
      "role": "user",
      "content": "艾尔登法环里的女武神怎么打？"
    }
  ],
  "temperature": 0.7
}'
```

---




# 十六、为什么需要 RAG？

在完成 LoRA 微调后，
模型已经能够：

- 使用攻略助手语气
- 进行结构化输出
- 模仿游戏问答风格

但是实际测试发现：

```text
模型“会说”，但不一定“说得对”
```

例如：

- NPC 关系错误
- Boss 技能串台
- 地图位置错误
- 不同游戏内容混淆

其根本原因在于：

```text
LoRA 更擅长学习“表达方式”
而不是“记忆大量事实知识”
```

因此：

仅依靠 LoRA 微调，
无法构建一个真正可靠的游戏知识库。
将开始构建：

```text
RAG 本地知识库系统
```

用于真正解决：

```text
“知识准确性”
```

问题。

---

## RAG 的核心思想

RAG 全称：

```text
Retrieval-Augmented Generation
检索增强生成
```

它的核心思想非常简单：

```text
不要让模型“硬记”
而是在回答前：
先去知识库“查资料”
```

整体流程：

```text
用户提问
    ↓
系统先检索知识库
    ↓
找到相关游戏资料
    ↓
把资料一起发送给模型
    ↓
模型基于资料生成答案
```

这样做后：

模型回答会更加：

- 准确
- 稳定
- 可控
- 不容易胡编乱造

---

# 十七、RAG 系统整体流程

在本项目中：

RAG 的整体流程如下：

```text
用户问题
    ↓
问题向量化（Embedding）
    ↓
向量数据库检索
    ↓
找到最相似文本
    ↓
拼接 Prompt
    ↓
发送给 Qwen
    ↓
模型生成最终回答
```

这里最关键的变化是：

```text
模型不再只依赖自身记忆
而是：
“边查资料边回答”
```

---

# 十八、Embedding（向量化）

## 什么是 Embedding？

计算机无法真正理解文本。

因此：

需要先把文本转换成：

```text
数字向量
```

这个过程叫：

```text
Embedding（向量化）
```

例如：

```text
“女武神怎么打”
```

会被转换成：

```text
[0.123, -0.531, 0.842, ...]
```

这样的高维数学向量。

---

## 为什么向量化很重要？

传统搜索：

```text
依赖关键词匹配
```

例如：

```text
搜索“女武神”
```

只能找到包含：

```text
“女武神”
```

这几个字的内容。

但向量搜索不同。

它搜索的是：

```text
语义相似度
```

例如：

```text
“玛莲妮亚打法”
```

即使没有出现：

```text
“女武神”
```

系统仍然能找到相关内容。

---

# 十九、文本切块（Chunk）

## 为什么不能直接把整本资料丢给模型？

因为：

大模型存在：

```text
上下文长度限制
```

例如：

```text
Qwen2.5-7B
```

一次只能读取有限 token。

因此：

必须先把长文本切成小块。

这个过程叫：

```text
Chunk（文本切块）
```

---

## Chunk 的作用

例如：

一本完整攻略：

```text
几十万字
```

会被切成：

```text
一小段一小段文本
```

例如：

```text
Boss 技能介绍
Boss 二阶段打法
武器属性说明
NPC 支线流程
```

这样：

系统才能：

```text
快速检索最相关内容
```

而不是把整本书都发给模型。

---

# 二十、向量数据库

## 什么是向量数据库？

向量数据库负责：

```text
存储所有文本向量
```

并支持：

```text
快速相似度搜索
```

例如：

用户提问：

```text
“女武神怎么打”
```

系统会：

- 将问题转为向量
- 与知识库所有向量进行相似度比较
- 找到最接近的文本

---

## 本项目使用 FAISS

本项目使用：



作为本地向量数据库。

它是 Facebook 开源的向量检索库。

优点：

- 检索速度快
- 占用资源较低
- 非常适合本地 RAG 项目

---

# 二十一、知识库构建流程

本项目知识库构建流程：

```text
TXT / Markdown 游戏资料
        ↓
文本切块（Chunk）
        ↓
Embedding 向量化
        ↓
写入 FAISS
        ↓
生成本地知识库
```

最终：

系统会得到：

```text
“可检索的游戏知识数据库”
```

---

# 二十二、RAG 与 LoRA 的区别

这是整个项目里非常关键的认知。

---

## LoRA 的作用

LoRA 更擅长：

```text
学习回答方式
```

例如：

- 语气
- 风格
- 输出结构
- 角色扮演

---

## RAG 的作用

RAG 更擅长：

```text
提供真实知识
```

例如：

- Boss 数据
- NPC 关系
- 地图位置
- 武器属性
- 游戏剧情

---

## 两者配合

因此：

最佳方案通常是：

```text
LoRA 负责“怎么说”
RAG 负责“说什么”
```

这也是目前大多数 AI 应用的典型架构。

---

# 二十三、项目架构升级

在加入 RAG 后，

项目整体架构升级为：

```text
用户问题
    ↓
FastAPI 后端
    ↓
RAG 检索知识库
    ↓
获取相关游戏资料
    ↓
拼接 Prompt
    ↓
Qwen2.5-7B + LoRA
    ↓
GPU 推理
    ↓
返回最终回答
```

相比之前：

模型回答：

- 更准确
- 更稳定
- 更像真正的游戏攻略助手

---

# 二十四、当前项目阶段

截至目前：

项目已经完成：

- 本地 Qwen 部署
- FastAPI API 服务
- LoRA 微调
- merged model 部署
- RAG 架构设计
- 向量知识库方案

下一阶段：

将正式开始：

```text
Embedding + FAISS 本地知识库实现
```

真正构建：

```text
“可检索”的艾尔登法环游戏数据库
```

# 二十五、扩展知识库

在确认 LoRA 更适合学习“回答风格”，而不适合记忆大量事实知识后，
项目开始进入 RAG 阶段。

这一阶段的目标是：

```text
把整理好的艾尔登法环资料构建成本地知识库，
让模型在回答前先检索相关资料，再基于资料生成答案。
```

---

# 二十六、扩展后的知识库结构

知识库目录命名为：

```text
lore
```

整体结构如下：

```text
lore
├── 人物志
│   ├── “半狼”布莱泽.md
│   ├── “废石”托普斯.md
│   ├── 亵渎君王拉卡德.md
│   ├── 初始之王葛孚雷.md
│   ├── 刮起风暴的女王——涅斐丽.露.md
│   ├── 呆萌少女菈雅.md
│   ├── 咖列.md
│   ├── 圣树米凯拉.md
│   ├── 壶哥亚历山大.md
│   ├── 夏玻利利.md
│   ├── 女武神玛莲妮亚.md
│   ├── 接肢葛瑞克.md
│   ├── 月之公主菈妮.md
│   ├── 末代君王蒙葛特.md
│   ├── 梅琳娜.md
│   ├── 死眠少女菲雅.md
│   ├── 狄亚罗斯·霍斯劳.md
│   ├── 白金之子.md
│   ├── 白面具梵雷.md
│   ├── 盲女海妲.md
│   ├── 碎星将军拉塔恩.md
│   ├── 米莉森.md
│   ├── 维克.md
│   ├── 罗德莉卡.md
│   ├── 罗杰尔.md
│   ├── 葛德文.md
│   ├── 贝纳尔.md
│   └── 鲜血君王蒙格.md
├── 游戏结局.md
├── 背景剧情.md
└── 黄金树幽影DLC.md
```

---

## 为什么使用 Markdown 管理知识库？

本项目选择使用 `.md` 文件管理知识库，原因是：

- 方便人工阅读和修改
- 支持标题、列表、段落等结构
- 适合整理剧情、人物、Boss、支线等长文本内容
- 后续可以直接被脚本递归读取并构建索引

相比 CSV 问答数据，Markdown 更适合保存：

```text
大段背景资料
人物故事
剧情解析
DLC 设定
```

---

# 二十七、构建 RAG 系统

## 安装 RAG 相关依赖

```bash
pip install langchain
pip install langchain-community
pip install langchain-huggingface
pip install chromadb
pip install sentence-transformers
pip install markdown
```

依赖说明：

| 依赖 | 作用 |
|---|---|
| langchain | 构建 RAG 流程 |
| langchain-community | 提供 Chroma 等社区组件 |
| langchain-huggingface | 接入 HuggingFace embedding 模型 |
| chromadb | 本地向量数据库 |
| sentence-transformers | 加载 embedding 模型 |
| markdown | 处理 Markdown 文本 |

---

# 二十八、编写 build_index.py

## 文件路径

```text
/root/autodl-fs/qwen_api/rag/build_index.py
```

---

## build_index.py 的作用

`build_index.py` 是知识库构建脚本。

它的核心任务是：

```text
把 Markdown 文档转换成可以被检索的向量数据库。
```

主要功能包括：

- 自动递归读取 `lore` 目录下的 Markdown 文件
- 提取每篇文档的文本内容
- 使用 `RecursiveCharacterTextSplitter` 进行文本切块
- 使用 `BAAI/bge-m3` 生成 embedding
- 使用 Chroma 构建本地向量数据库
- 将向量索引保存到本地磁盘

---

## 为什么要递归读取 Markdown？

因为知识库不是一个单独文件，而是多个目录、多篇文章组成。

例如：

```text
lore/人物志/月之公主菈妮.md
lore/背景剧情.md
lore/黄金树幽影DLC.md
```

递归读取可以自动遍历所有 `.md` 文件，
后续新增资料时不需要手动修改脚本。

---

## 为什么要切块？

一篇 Markdown 文档可能很长。

如果直接整篇放入向量数据库，会导致：

- 检索粒度太粗
- 召回内容不精准
- Prompt 过长
- 模型容易抓不到重点

因此需要切成多个较小文本块。

例如：

```text
月之公主菈妮.md
    ↓
菈妮身份背景 chunk
菈妮支线流程 chunk
菈妮结局相关 chunk
菈妮与双指关系 chunk
```

这样用户问：

```text
菈妮是谁？
```

系统就能更精准地召回“身份背景”相关片段。

---

## 使用 RecursiveCharacterTextSplitter

本项目使用：

```text
RecursiveCharacterTextSplitter
```

进行文本切分。

它会尽量按照：

```text
段落 → 句子 → 字符
```

的优先级进行切分。

相比简单按固定字数切分，
这种方式更容易保留语义完整性。

---

# 二十九、Embedding 模型

## 使用 BAAI/bge-m3

本项目使用的 embedding 模型是：

```text
BAAI/bge-m3
```

它的作用是：

```text
把用户问题和知识库文本转换为向量。
```

向量化之后，
系统就可以计算两段文本之间的语义相似度。

---

## 为什么不用 Qwen 本身做检索？

Qwen 负责生成回答。

而 embedding 模型负责语义检索。

两者分工不同：

| 模型 | 作用 |
|---|---|
| Qwen2.5-7B | 生成最终回答 |
| BAAI/bge-m3 | 将文本转换成向量 |
| Chroma | 存储和检索向量 |

这样架构更清晰，也更容易维护。

---

# 三十、使用 Chroma 构建向量数据库

## 向量数据库路径

```text
/root/autodl-fs/qwen_api/vectory_db
```

> 注意：这里目录名是 `vectory_db`。如果后续重新整理项目，建议改成更常见的 `vector_db`。

---

## 构建向量索引

进入项目目录：

```bash
cd /root/autodl-fs/qwen_api
```

执行：

```bash
python rag/build_index.py
```

构建成功后，会生成：

```text
vectory_db
├── chroma.sqlite3
└── xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```

其中：

| 文件/目录 | 作用 |
|---|---|
| chroma.sqlite3 | Chroma 的索引数据库 |
| UUID 目录 | 存储向量相关数据 |

---

# 三十一、编写 Retriever 检索模块

## 文件路径

```text
/root/autodl-fs/qwen_api/rag/retriever.py
```

---

## retriever.py 的作用

`retriever.py` 负责从向量数据库中检索相关知识。

它主要完成：

- 加载 Chroma 向量库
- 加载 BGE embedding 模型
- 将用户问题转换为向量
- 在向量库中进行相似度检索
- 返回最相关的 top_k 个文本块

---

## 什么是 top_k？

`top_k` 表示：

```text
返回最相似的前 k 条文本片段。
```

例如：

```bash
--top_k 3
```

表示返回最相关的 3 个 chunk。

top_k 太小：

```text
可能漏掉关键信息
```

top_k 太大：

```text
可能引入无关内容，并增加 Prompt 长度
```

因此通常可以从：

```text
3 ~ 5
```

开始测试。

---

## 测试 Retriever

```bash
python rag/retriever.py \
  --query "菈妮是谁？" \
  --top_k 3
```

如果输出了与菈妮相关的文本片段，
说明向量库构建和检索流程正常。

---

# 三十二、编写 RAG Chat 模块

## 文件路径

```text
/root/autodl-fs/qwen_api/rag/rag_chat.py
```

---

## rag_chat.py 的作用

`rag_chat.py` 是完整 RAG 问答流程的测试脚本。

它把：

```text
检索模块 + 大模型接口
```

连接起来。

整体流程：

```text
用户问题
    ↓
Retriever 检索知识库
    ↓
获取 top_k 相关 chunk
    ↓
拼接成 Prompt
    ↓
请求 vLLM
    ↓
Qwen2.5-7B merged model
    ↓
返回最终回答
```

---

## Prompt 拼接思路

RAG 的关键是：

```text
把检索到的资料放进 Prompt，
要求模型只能基于资料回答。
```

示例结构：

```text
你是一个专业的艾尔登法环攻略助手。
请根据下面的资料回答用户问题。

【参考资料】
1. ...
2. ...
3. ...

【用户问题】
古龙时代有什么故事？

【回答要求】
- 优先基于参考资料回答
- 不要编造资料中没有的信息
- 如果资料不足，请明确说明
```

这种写法可以减少模型胡编乱造。

---

## 测试 RAG Chat

```bash
python rag/rag_chat.py \
  --query "古龙时代有什么故事？" \
  --top_k 3
```

如果回答内容明显引用了知识库资料，
说明 RAG 流程已经打通。

---

# 三十三、解决 CUDA OOM 问题

## 遇到的问题

在测试 RAG 时，出现了 CUDA 显存不足问题：

```text
CUDA out of memory
```

原因是：

```text
embedding 模型和 vLLM 同时占用 GPU
```

也就是：

- vLLM 正在 GPU 上运行 Qwen2.5-7B merged model
- BAAI/bge-m3 也尝试加载到 GPU
- 两者同时占用显存，导致 OOM

---

## 解决方案

将 embedding 模型放到 CPU 上运行：

```python
model_kwargs={"device": "cpu"}
```

这样：

- GPU 专门负责 Qwen 推理
- CPU 负责 embedding 检索
- 避免显存冲突
- 系统更稳定

缺点是：

```text
检索速度会稍慢
```

但对于个人项目和单用户问答系统来说，
这个方案完全可接受。

---

# 三十四、合并后的模型部署

## merged model 路径

```text
/root/autodl-fs/models/qwen2.5-7b-elden-ring-merged
```

---

## 启动 vLLM

```bash
vllm serve /root/autodl-fs/models/qwen2.5-7b-elden-ring-merged \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16
```

---

## 测试 vLLM

```bash
curl http://127.0.0.1:8000/v1/models
```

如果能够返回模型列表，
说明合并后的模型服务已经正常启动。

---

# 三十五、FastAPI 接入 RAG

## 文件路径

```text
/root/autodl-fs/qwen_api/main.py
```

---

## main.py 当前功能

在加入 RAG 后，
`main.py` 不再只是简单转发用户问题给 vLLM。

它现在负责完整的后端问答流程：

- 接收前端问题
- 调用 Retriever 检索知识库
- 将检索结果拼接进 Prompt
- 请求 vLLM 模型服务
- 获取模型回答
- 返回给前端

---

## 新增接口：POST /rag/chat

接口：

```text
POST /rag/chat
```

请求示例：

```json
{
  "question": "蒙葛特伟大的灵魂怎么体现？",
  "top_k": 4
}
```

字段说明：

| 字段 | 作用 |
|---|---|
| question | 用户输入的问题 |
| top_k | 检索最相关的文本片段数量 |

---

## 为什么保留 top_k 参数？

保留 `top_k` 可以方便调试。

例如：

```json
{
  "question": "菈妮是谁？",
  "top_k": 3
}
```

表示：

```text
只检索最相关的 3 条资料
```

如果发现回答不完整，
可以尝试：

```json
{
  "question": "菈妮是谁？",
  "top_k": 5
}
```

通过调整 `top_k`，
可以观察不同检索数量对回答质量的影响。

---

## 测试 RAG 接口

```bash
curl -X POST http://127.0.0.1:9000/rag/chat \
-H "Content-Type: application/json" \
-d '{
  "question": "蒙葛特伟大的灵魂怎么体现？",
  "top_k": 4
}'
```

如果返回的回答能够结合知识库内容，
说明：

- FastAPI 正常运行
- Retriever 正常加载
- Chroma 向量库可用
- vLLM 模型服务可用
- RAG 问答链路已经打通

---

# 三十六、加入 RAG 后的项目架构

目前系统架构已经升级为：

```text
用户问题
    ↓
FastAPI /rag/chat
    ↓
Retriever 检索模块
    ↓
Chroma 向量数据库
    ↓
BGE-M3 Embedding
    ↓
召回相关知识片段
    ↓
拼接 Prompt
    ↓
vLLM 模型服务
    ↓
Qwen2.5-7B merged model
    ↓
返回最终回答
```

---

# 三十七、当前阶段总结

通过这一阶段，项目已经完成了从：

```text
“能聊天的本地大模型”
```

到：

```text
“能查资料再回答的本地知识问答系统”
```

的升级。

这一阶段的关键收获是：

```text
RAG 解决知识准确性问题
LoRA 解决回答风格问题
FastAPI 负责系统接口封装
vLLM 负责模型推理服务
Chroma 负责知识库检索
```

最终形成的完整定位是：

```text
基于 Qwen2.5-7B + LoRA + RAG 的艾尔登法环本地 AI 攻略助手
```

# 三十八、解决浏览器跨域问题（CORS）

在前端页面接入 FastAPI 后，可能会遇到浏览器跨域问题。

这是因为：

```text
前端页面端口：7860
FastAPI 接口端口：9000
```

虽然它们都运行在同一台服务器上，
但在浏览器看来：

```text
不同端口 = 不同源
```

因此浏览器可能会拦截请求。

---

## 什么是 CORS？

CORS 全称：

```text
Cross-Origin Resource Sharing
跨源资源共享
```

简单理解：

```text
浏览器为了安全，不允许网页随便请求另一个地址的接口。
```

例如：

```text
http://127.0.0.1:7860
```

请求：

```text
http://127.0.0.1:9000/rag/chat
```

就可能触发跨域限制。

---

## 解决方法

在 `main.py` 中加入：

```python
from fastapi.middleware.cors import CORSMiddleware
```

然后添加中间件：

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 参数说明

| 参数 | 作用 |
|---|---|
| allow_origins | 允许哪些前端地址访问 |
| allow_credentials | 是否允许携带 Cookie 等认证信息 |
| allow_methods | 允许哪些请求方法 |
| allow_headers | 允许哪些请求头 |

当前项目处于学习和本地测试阶段，
因此使用：

```python
allow_origins=["*"]
```

表示允许所有来源访问。

如果后续正式部署到公网，
建议改成固定前端地址，例如：

```python
allow_origins=["http://127.0.0.1:7860"]
```

这样更加安全。

---

# 三十九、服务工程化

前面所有服务都可以通过手动命令启动。

但随着项目变复杂，
手动启动会出现几个问题：

- 命令太长，容易输错
- vLLM 和 FastAPI 需要分别启动
- 关闭终端后服务可能中断
- 查看日志不方便
- 服务器重启后需要重新执行多条命令

因此需要将启动、停止、重启流程脚本化。

---

## 创建 scripts 和 logs 目录

```bash
mkdir -p /root/autodl-fs/qwen_api/scripts
mkdir -p /root/autodl-fs/qwen_api/logs
```

目录作用：

| 目录 | 作用 |
|---|---|
| scripts | 存放启动、停止、重启脚本 |
| logs | 存放服务运行日志 |

---

# 四十、start_vllm.sh

## 脚本作用

`start_vllm.sh` 用于后台启动 vLLM 模型服务。

它主要负责：

- 自动激活 conda 环境
- 启动 merged model
- 指定服务端口 8000
- 将运行日志写入 `logs/vllm.log`
- 让模型服务在后台运行

---

## 启动命令

```bash
bash /root/autodl-fs/qwen_api/scripts/start_vllm.sh
```

---

## 查看日志

```bash
tail -f /root/autodl-fs/qwen_api/logs/vllm.log
```

如果日志中出现：

```text
Application startup complete
```

说明 vLLM 已经启动完成。

---

# 四十一、start_api.sh

## 脚本作用

`start_api.sh` 用于后台启动 FastAPI 服务。

它主要负责：

- 自动激活 conda 环境
- 进入 qwen_api 项目目录
- 启动 `main.py`
- 指定服务端口 9000
- 将运行日志写入 `logs/api.log`

---

## 启动命令

```bash
bash /root/autodl-fs/qwen_api/scripts/start_api.sh
```

---

## 查看日志

```bash
tail -f /root/autodl-fs/qwen_api/logs/api.log
```

如果日志中出现：

```text
Uvicorn running on http://0.0.0.0:9000
```

说明 FastAPI 已经启动成功。

---

# 四十二、stop_all.sh

## 脚本作用

`stop_all.sh` 用于停止当前项目相关服务。

它主要负责停止：

- FastAPI 服务
- vLLM 服务

---

## 执行命令

```bash
bash /root/autodl-fs/qwen_api/scripts/stop_all.sh
```

执行后可以检查端口是否还被占用：

```bash
lsof -i:8000
lsof -i:9000
```

如果没有输出，
说明端口已经释放。

---

# 四十三、restart_all.sh

## 脚本作用

`restart_all.sh` 是最常用的启动脚本。

它的作用是：

- 停止旧服务
- 启动 vLLM
- 等待 vLLM 初始化
- 启动 FastAPI
- 自动检测服务是否正常
- 输出最终启动结果

---

## 执行命令

```bash
bash /root/autodl-fs/qwen_api/scripts/restart_all.sh
```

最终成功输出：

```text
[SUCCESS] 所有服务启动完成
```

---

## 为什么要等待 vLLM 初始化？

vLLM 启动大模型需要一定时间。

如果 FastAPI 太早调用 vLLM，
可能会出现：

```text
Connection refused
```

或者：

```text
模型接口暂时不可用
```

因此 `restart_all.sh` 中需要先等待 vLLM 完成初始化，
再启动或检测 FastAPI。

---

# 四十四、搭建网页前端

后端接口完成后，
还需要一个网页页面用于交互。

这样用户就不需要使用 `curl` 命令，
可以直接在浏览器中提问。

---

## 创建 web 目录

```bash
mkdir -p /root/autodl-fs/qwen_api/web
```

---

## 编写 index.html

文件路径：

```text
/root/autodl-fs/qwen_api/web/index.html
```

页面功能包括：

- 用户输入框
- 发送按钮
- AI 回答展示区域
- Loading 动画
- Markdown 渲染
- 深色模式 UI

---

## 前端请求流程

网页中的 JavaScript 会将用户输入发送到 FastAPI：

```text
用户输入问题
    ↓
点击发送按钮
    ↓
fetch 请求 /rag/chat
    ↓
FastAPI 调用 RAG
    ↓
返回 answer
    ↓
页面渲染 Markdown
```

---

# 四十五、启动网页服务

进入前端目录：

```bash
cd /root/autodl-fs/qwen_api/web
```

启动一个简单的静态文件服务：

```bash
python -m http.server 7860 --bind 0.0.0.0
```

这里使用的是 Python 自带的 HTTP 服务。

它的作用是：

```text
把 index.html 作为网页提供给浏览器访问。
```

服务端口为：

```text
7860
```

---

# 四十六、修复 Markdown 渲染问题

## 遇到的问题

模型返回的内容中可能包含：

```text
\n
```

但网页没有正确换行。

例如返回内容显示成：

```text
第一段\n第二段\n第三段
```

而不是：

```text
第一段
第二段
第三段
```

---

## 解决方法

在前端渲染前进行替换：

```javascript
.replace(/\\n/g, "\n")
```

它的作用是：

```text
将字符串中的 "\\n" 转换成真正的换行符
```

这样 Markdown 渲染器才能正确识别段落和换行。

---

# 四十七、SSH 连接服务器（MobaXterm）

由于项目运行在远程 GPU 服务器上，
本地电脑需要通过 SSH 连接服务器。

本项目使用：

```text
MobaXterm
```

进行远程连接。

---

## MobaXterm 的作用

MobaXterm 可以用于：

- SSH 登录服务器
- 文件拖拽上传
- 多终端管理
- 创建 SSH Tunnel
- 查看远程文件目录

下载地址：

```text
https://mobaxterm.mobatek.net/download.html
```

---

## 创建 SSH Session

AutoDL 提供的 SSH 命令类似：

```bash
ssh -p 54797 root@connect.bjb1.seetacloud.com
```

在 MobaXterm 中对应填写：

| MobaXterm 字段 | 内容 |
|---|---|
| Remote host | connect.bjb1.seetacloud.com |
| Username | root |
| Port | 54797 |

连接成功后，
就可以在 MobaXterm 终端中操作服务器。

---

# 四十八、SSH Tunnel 映射

## 为什么需要 SSH Tunnel？

项目中的服务运行在远程服务器上：

| 服务 | 服务器端口 |
|---|---:|
| 前端网页 | 7860 |
| FastAPI | 9000 |
| vLLM | 8000 |

但本地浏览器不能直接访问服务器内部端口。

因此需要使用：

```text
SSH Tunnel
```

将服务器端口映射到本地电脑。

简单理解：

```text
把服务器上的 7860 / 9000 端口，
“搬到”本地电脑上访问。
```

---

## 映射网页端口

| 项 | 内容 |
|---|---|
| Forwarded port | 7860 |
| Remote server | 127.0.0.1 |
| Remote port | 7860 |
| SSH server | connect.bjb1.seetacloud.com |
| SSH login | root |
| SSH port | 54797 |

映射成功后，
本地浏览器访问：

```text
http://127.0.0.1:7860
```

即可打开网页。

---

## 映射 API 端口

| 项 | 内容 |
|---|---|
| Forwarded port | 9000 |
| Remote server | 127.0.0.1 |
| Remote port | 9000 |
| SSH server | connect.bjb1.seetacloud.com |
| SSH login | root |
| SSH port | 54797 |

映射成功后，
本地前端即可请求：

```text
http://127.0.0.1:9000/rag/chat
```

---

## 是否需要映射 vLLM 端口？

通常不需要。

因为：

```text
浏览器 → FastAPI → vLLM
```

浏览器只需要访问 FastAPI。

vLLM 只需要被服务器内部的 FastAPI 调用即可。

因此一般只映射：

```text
7860
9000
```

不需要把：

```text
8000
```

暴露到本地。

---

# 四十九、最终系统架构

目前完整系统架构如下：

```text
本地浏览器
    ↓
SSH Tunnel 映射 7860
    ↓
index.html 前端页面
    ↓
SSH Tunnel 映射 9000
    ↓
FastAPI /rag/chat
    ↓
Retriever 检索模块
    ↓
Chroma Vector DB
    ↓
BGE-M3 Embedding
    ↓
召回相关知识片段
    ↓
拼接 Prompt
    ↓
vLLM (8000)
    ↓
Qwen2.5-7B merged model
    ↓
RTX 5090 推理
    ↓
返回答案到网页
```

---

# 五十、当前项目完成效果

经过以上步骤，
项目已经实现了一个完整的本地 AI 攻略问答系统。

目前具备：

- 网页端提问
- FastAPI 后端接口
- RAG 本地知识库检索
- Chroma 向量数据库
- BGE-M3 语义检索
- Qwen2.5-7B + LoRA merged model
- vLLM 高性能推理
- SSH Tunnel 本地访问
- 脚本化启动与停止

最终用户可以在浏览器中输入：

```text
菈妮是谁？
```

系统会：

```text
先检索本地艾尔登法环知识库，
再调用 Qwen 模型生成结构化回答。
```

这标志着项目已经从：

```text
单纯调用本地大模型
```

升级为：

```text
具备本地知识库能力的 RAG 问答应用
```