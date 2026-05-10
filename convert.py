import csv
import json
import argparse


def safe_strip(value):
    """安全处理 None"""
    if value is None:
        return ""
    return str(value).strip()


def csv_to_alpaca_json(csv_path, json_path):
    data = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        print("检测到字段名:", reader.fieldnames)

        for i, row in enumerate(reader):
            instruction = safe_strip(row.get("instruction"))
            input_text = safe_strip(row.get("input"))
            output = safe_strip(row.get("output"))

            # 跳过无效数据
            if not output:
                print(f"[Warning] 第 {i} 行 output 为空，已跳过")
                continue

            item = {
                "instruction": instruction,
                "input": input_text,
                "output": output
            }

            data.append(item)

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"转换完成，共 {len(data)} 条数据 -> {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    args = parser.parse_args()

    csv_to_alpaca_json(args.input, args.output)