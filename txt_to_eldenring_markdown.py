from openai import OpenAI
from pathlib import Path
import os

# =========================
# 配置区
# =========================

INPUT_DIR = Path("txt_inputs")
OUTPUT_DIR = Path("md_outputs")

MODEL_NAME = "deepseek-v4-pro"

SLEEP_SECONDS = 1



client = OpenAI(
    api_key="sk-380c7c6ef697433d9a44dab70436f2d6",
    base_url="https://api.deepseek.com"
)
# =========================
# 工具函数
# =========================

def read_txt_file(file_path: Path) -> str:
    """读取 txt 文件，自动兼容常见编码"""
    encodings = ["utf-8", "utf-8-sig", "gbk", "gb18030"]

    for enc in encodings:
        try:
            return file_path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue

    raise ValueError(f"无法识别文件编码：{file_path}")


def clean_markdown_output(text: str) -> str:
    """去掉模型可能输出的 ```markdown 包裹"""
    text = text.strip()

    if text.startswith("```markdown"):
        text = text[len("```markdown"):].strip()

    if text.startswith("```"):
        text = text[len("```"):].strip()

    if text.endswith("```"):
        text = text[:-3].strip()

    return text


def build_prompt(text: str, file_stem: str) -> str:
    return f"""
你是一个《艾尔登法环》游戏资料整理助手。

现在我会提供一个完整 TXT 文档中的全部内容。
这个 TXT 是一个完整专题资料，不需要分段处理。

你的任务是：
通读全文，综合所有内容，将原始资料整理成结构清晰、适合长期保存与后续 RAG 使用的 Markdown 文档。

当前文件名：{file_stem}

资料可能包含：
- 世界观背景
- 游戏剧情
- 时间线
- Boss 信息
- NPC 支线
- 地图区域
- 武器、防具、护符、战灰、骨灰
- 魔法、祷告、道具
- 职业信息
- 任务流程
- 打法攻略
- 配装推荐
- 地点路线
- 掉落奖励
- 人物志
- 人物生平
- 人物关系
- 人物支线

整理要求：

1. 只输出 Markdown 正文，不要输出解释说明。
2. 使用清晰的 Markdown 标题层级：#、##、###。
3. 保留原文中的关键游戏信息，不要随意删减重要内容。
4. 不要编造原文没有的信息。
5. 如果原文没有提到某项信息，不要强行补充。
6. 删除重复、口语化、无关内容。
7. 不要生成问答内容。
8. 不要生成“适合RAG检索的问答”章节。
9. 不要输出 Q1、Q2、Q3 等形式内容。
10. 不要把内容改写成“用户提问 + 回答”格式。
11. 对专有名词尽量保留原称呼，例如：
    - Boss 名称
    - NPC 名称
    - 地图名称
    - 道具名称
    - 法术名称
12. 如果出现流程类内容，请整理为步骤列表。
13. 如果出现打法、配装、路线，请整理为要点列表。
14. 如果出现代码、命令、表格型资料，请保留为 Markdown 表格或代码块。
15. 不要加入“根据原文”“以下是整理结果”等说明性废话。
16. 内容需要适合后续 RAG 使用，因此：
    - 标题必须明确
    - 段落不要过长
    - 一个知识点尽量独立
    - 时间线尽量清晰
17. 如果资料属于世界观或剧情：
    - 优先整理时间线
    - 梳理人物关系
    - 梳理势力关系
    - 区分“原作明确设定”和“玩家推测”
18.如果资料属于人物志，请优先整理人物身份、生平经历、人物关系、关键事件、相关势力与争议说法。
19. 如果多篇文章存在重复内容，请自动合并去重。
20. 如果不同资料存在矛盾说法，请单独整理为：
    - “争议内容”
    - “不同说法”
    - “社区推测”
21. 不要强制套模板，要根据资料类型自动调整结构。
22. 最终输出必须是一份完整统一的 Markdown 文档，不要出现“分段”“chunk”“第几段”等痕迹。

根据资料类型，可参考下面的结构：

========================
【世界观 / 剧情类】
========================

# {file_stem}

## 一、剧情概述

## 二、世界观背景

## 三、时间线梳理

## 四、主要势力

## 五、关键人物

## 六、关键事件

## 七、重要概念与设定

## 八、争议与不同说法

## 九、补充说明

========================
【Boss 类】
========================

# Boss 名称

## 基本信息

## 所在区域

## 触发条件

## 攻击特点

## 战斗机制

## 推荐打法

## 掉落奖励

## 相关剧情

## 注意事项

========================
【NPC 支线类】
========================

# NPC 名称

## 基本信息

## 初次位置

## 支线流程

## 关键选择

## 奖励

## 相关人物

## 相关地图

## 注意事项

========================
【装备 / 道具 / 魔法类】
========================

# 名称

## 基本信息

## 类型

## 获取位置

## 获取方法

## 属性与效果

## 使用建议

## 适合流派

## 注意事项

========================
【职业 / Build 类】
========================

# 职业名称

## 初始属性

## 初始装备

## 职业特点

## 推荐加点

## 推荐武器

## 推荐流派

## 优缺点分析

## 新手推荐程度

========================
【人物志类】
========================
# 人物名称

## 一、人物概述

## 二、身份与背景

## 三、生平经历

## 四、关键事件

## 五、人物关系

## 六、相关势力

## 七、相关地点

## 八、相关 Boss / 道具 / 支线

## 九、争议与不同说法

## 十、补充说明

原始资料如下：

{text}
"""


def convert_chunk_to_markdown(text: str, file_stem: str) -> str:
    prompt = build_prompt(text, file_stem)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "你擅长将杂乱游戏资料整理成适合 RAG 检索的高质量 Markdown 知识库。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.1,
    )

    md = response.choices[0].message.content
    return clean_markdown_output(md)


def process_one_txt(txt_path: Path):
    print(f"\n[INFO] 开始处理：{txt_path}")

    text = read_txt_file(txt_path)

    if not text.strip():
        print(f"[WARN] 文件为空，跳过：{txt_path}")
        return

    file_stem = txt_path.stem

    relative_path = txt_path.relative_to(INPUT_DIR)
    output_path = OUTPUT_DIR / relative_path.with_suffix(".md")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("[INFO] 正在将整个 TXT 一次性发送给 DeepSeek...")
    print(f"[INFO] 文本长度：{len(text)} 字符")

    final_md = convert_chunk_to_markdown(
        text=text,
        file_stem=file_stem
    )

    output_path.write_text(final_md, encoding="utf-8")

    print(f"[DONE] 已生成：{output_path}")


def main():
    INPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    txt_files = sorted(INPUT_DIR.rglob("*.txt"))

    if not txt_files:
        print(f"[WARN] 未找到 txt 文件，请把 txt 放到文件夹：{INPUT_DIR}")
        return

    print(f"[INFO] 共找到 {len(txt_files)} 个 txt 文件")

    for txt_path in txt_files:
        try:
            process_one_txt(txt_path)
        except Exception as e:
            print(f"[ERROR] 处理失败：{txt_path}")
            print(f"[ERROR] 原因：{e}")

    print("\n[ALL DONE] 全部处理完成")


if __name__ == "__main__":
    main()