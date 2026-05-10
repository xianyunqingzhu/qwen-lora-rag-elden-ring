import os
import csv
import json
from openai import OpenAI

# =========================
# 配置区域
# =========================

# DeepSeek API Key
API_KEY = "your api key here"

# DeepSeek API 地址
BASE_URL = "https://api.deepseek.com"

# 输入txt文件
INPUT_TXT = "elden_ring.txt"

# 输出csv文件
OUTPUT_CSV = "elden_ring_train-7.csv"

# # 每次发送的文本长度（太长容易超token）
# CHUNK_SIZE = 12000

# 使用模型
MODEL_NAME = "deepseek-v4-flash"

# =========================
# 初始化客户端
# =========================

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

# =========================
# Prompt模板
# =========================

SYSTEM_PROMPT = """
你是一个专业的数据整理助手。

你的任务：
把游戏攻略文本整理成适用于LoRA微调的高质量训练数据。

输出格式必须严格为JSON数组。

每一项必须包含：

[
  {
    "instruction": "...",
    "input": "...",
    "output": "..."
  }
]

要求：

1. instruction：
必须是明确的系统身份描述。
例如：
"你是一个专业的艾尔登法环攻略助手，请用结构化方式回答。"

2. input：
必须是玩家会真实提出的问题。

3. output：
必须：
- 结构化
- 分点
- 逻辑清晰
- 使用\\n换行
- 内容专业
- 不要太短

4. 不允许：
- markdown
- ```json
- 解释
- 注释
- 多余文本

5. 输出必须是合法JSON数组。

6. 输出内容必须适合训练中文游戏攻略助手。

7. 根据输入内容生成1~5条高质量训练数据。不要遗漏关键信息。
"""

# =========================
# 按行读取txt
# 一行 = 一个知识点
# =========================

with open(INPUT_TXT, "r", encoding="utf-8") as f:

    full_text = f.read()

chunks = [
    chunk.strip()
    for chunk in full_text.split("<<<END>>>")
    if chunk.strip()
]

print(f"共读取 {len(chunks)} 个知识块")

# =========================
# 写CSV头
# =========================

csv_exists = os.path.exists(OUTPUT_CSV)

csv_file = open(
    OUTPUT_CSV,
    "a",
    newline="",
    encoding="utf-8-sig"
)

writer = csv.writer(csv_file)

if not csv_exists:
    writer.writerow(["instruction", "input", "output"])

# =========================
# 开始处理
# =========================

for idx, chunk in enumerate(chunks):

    print(f"\n正在处理第 {idx + 1}/{len(chunks)} 块...")

    user_prompt = f"""
请把下面游戏攻略文本整理为训练数据：

{chunk}
"""

    try:

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            temperature=0.7,
            max_tokens=4000
        )

        result = response.choices[0].message.content.strip()

        # =========================
        # JSON解析
        # =========================

        data = json.loads(result)

        count = 0

        for item in data:

            instruction = item.get("instruction", "").strip().replace("\n", "\\n")

            input_text = item.get("input", "").strip().replace("\n", "\\n")

            output = item.get("output", "").strip().replace("\n", "\\n")


            # 过滤空内容
            if not instruction or not input_text or not output:
                continue

            writer.writerow([
                instruction,
                input_text,
                output
            ])
            count += 1

        print(f"成功写入 {count} 条数据")

    except Exception as e:

        print("处理失败：", e)

# =========================
# 关闭文件
# =========================

csv_file.close()

print("\n全部完成")
print(f"输出文件：{OUTPUT_CSV}")