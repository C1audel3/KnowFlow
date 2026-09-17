"""Generate the deterministic 12-document, 120-question phase-7 benchmark."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import PROJECT_ROOT


CORPUS_DIR = PROJECT_ROOT / "data" / "benchmarks" / "phase7" / "documents"
DATASET_PATH = PROJECT_ROOT / "data" / "evaluation" / "phase7_benchmark.jsonl"

PROJECTS = [
    ("Aster", "赵宁", "成都", "2027 Q1", "S-17", "21", "93", "NOVA", "IMG-A17", "F1", "2PR/(P+R)"),
    ("Birch", "孙悦", "苏州", "2027 Q2", "S-24", "18", "96", "LYRA", "IMG-B24", "IoU", "交集/并集"),
    ("Cedar", "周凯", "武汉", "2027 Q3", "S-31", "27", "91", "VEGA", "IMG-C31", "MAE", "|y-ŷ|平均值"),
    ("Dawn", "李雯", "西安", "2027 Q4", "S-42", "16", "97", "ORBIT", "IMG-D42", "RMSE", "平方误差均值开方"),
    ("Ember", "陈浩", "杭州", "2028 Q1", "S-53", "25", "92", "POLAR", "IMG-E53", "MAPE", "绝对百分比误差均值"),
    ("Fjord", "王琪", "青岛", "2028 Q2", "S-68", "19", "95", "SIRIUS", "IMG-F68", "R2", "1-SSE/SST"),
    ("Grove", "郑宇", "南京", "2028 Q3", "S-75", "23", "94", "ALTAIR", "IMG-G75", "Dice", "2|A∩B|/(|A|+|B|)"),
    ("Harbor", "刘佳", "厦门", "2028 Q4", "S-86", "15", "98", "MIRA", "IMG-H86", "BLEU", "几何均值×长度惩罚"),
    ("Iris", "何川", "天津", "2029 Q1", "S-93", "29", "90", "RIGEL", "IMG-I93", "AUC", "ROC曲线下面积"),
    ("Juniper", "郭琳", "重庆", "2029 Q2", "S-105", "20", "95", "DENEB", "IMG-J105", "NDCG", "DCG/IDCG"),
    ("Kite", "马骁", "合肥", "2029 Q3", "S-116", "17", "97", "ARCTURUS", "IMG-K116", "MRR", "首个相关结果排名的倒数均值"),
    ("Lumen", "唐雪", "长沙", "2029 Q4", "S-128", "24", "93", "CAPELLA", "IMG-L128", "Recall", "TP/(TP+FN)"),
]


def evidence_id(index: int, modality: str) -> str:
    return f"p{index + 1:02d}-{modality}"


def write_documents() -> None:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(PROJECTS):
        project, owner, city, quarter, sensor, latency, accuracy, gateway, image_code, metric, formula = row
        content = f"""# Project {project} 技术档案

[EVIDENCE:{evidence_id(index, 'text')}]
{project} 项目的负责人是{owner}，试点城市为{city}，目标发布时间为 {quarter}。档案编号 KF-{index + 1:02d}。

[EVIDENCE:{evidence_id(index, 'table')}]
| 项目 | 传感器 | 延迟 | 准确率 |
|---|---|---:|---:|
| {project} | {sensor} | {latency} ms | {accuracy}% |

[EVIDENCE:{evidence_id(index, 'image')}]
架构图文字说明：{project} 的视觉网关为 {gateway}，图像校验码为 {image_code}。流程从 CAMERA 指向 {gateway}，最终进入 KNOWLEDGE BASE。

[EVIDENCE:{evidence_id(index, 'equation')}]
评测公式卡：{metric} 的定义为 {formula}。该指标仅用于 {project} 项目本轮验收。
"""
        (CORPUS_DIR / f"project_{index + 1:02d}_{project.lower()}.md").write_text(content, encoding="utf-8")


def case(identifier: str, question: str, kind: str, answer: str, keywords: list[str], evidence: list[str], difficulty: str) -> dict:
    return {
        "id": identifier,
        "question": question,
        "type": kind,
        "difficulty": difficulty,
        "reference_answer": answer,
        "expected_keywords": keywords,
        "evidence_ids": evidence,
    }


def build_cases() -> list[dict]:
    cases: list[dict] = []
    difficulties = ["easy", "medium", "hard"]
    serial = 0

    def add(*args):
        nonlocal serial
        cases.append(case(*args, difficulties[serial % 3]))
        serial += 1

    for i, row in enumerate(PROJECTS):
        project, owner, city, quarter, sensor, latency, accuracy, gateway, image_code, metric, formula = row
        text_id, table_id = evidence_id(i, "text"), evidence_id(i, "table")
        image_id, equation_id = evidence_id(i, "image"), evidence_id(i, "equation")
        add(f"text-{i+1:02d}-owner", f"{project} 项目的负责人是谁？", "text", owner, [owner], [text_id])
        add(f"text-{i+1:02d}-release", f"{project} 项目的目标发布时间是什么？", "text", quarter, quarter.split(), [text_id])
        if i == 0:
            add("text-01-city", "Aster 项目的试点城市在哪里？", "text", city, [city], [text_id])
        add(f"table-{i+1:02d}-latency", f"{project} 项目所用传感器的延迟是多少？", "table", f"{latency} ms", [latency, "ms"], [table_id])
        if i < 8:
            add(f"table-{i+1:02d}-accuracy", f"{sensor} 的准确率是多少？", "table", f"{accuracy}%", [f"{accuracy}%"], [table_id])
        add(f"image-{i+1:02d}-gateway", f"{project} 架构图中的视觉网关叫什么？", "image", gateway, [gateway], [image_id])
        if i < 8:
            add(f"image-{i+1:02d}-code", f"{project} 的图像校验码是什么？", "image", image_code, [image_code], [image_id])
        add(f"equation-{i+1:02d}-metric", f"{project} 验收使用的评测指标是什么？", "equation", metric, [metric], [equation_id])
        if i < 3:
            definition_keywords = ["|y-ŷ|", "平均值"] if metric == "MAE" else [formula]
            add(f"equation-{i+1:02d}-definition", f"请给出 {metric} 在档案中的定义。", "equation", formula, definition_keywords, [equation_id])
        other = PROJECTS[(i + 1) % len(PROJECTS)]
        other_text = evidence_id((i + 1) % len(PROJECTS), "text")
        add(f"cross-{i+1:02d}-owner-release", f"请同时给出 {project} 的负责人和 {other[0]} 的发布时间。", "cross_document", f"{owner}；{other[3]}", [owner, *other[3].split()], [text_id, other_text])
        if i < 8:
            add(f"cross-{i+1:02d}-gateway-sensor", f"请同时给出 {project} 的视觉网关和 {other[0]} 的传感器编号。", "cross_document", f"{gateway}；{other[4]}", [gateway, other[4]], [image_id, evidence_id((i + 1) % len(PROJECTS), "table")])
        add(f"refusal-{i+1:02d}-budget", f"{project} 项目的预算是多少？若档案未提供请明确说明。", "refusal", "档案未提供，无法确定", ["未提供"], [text_id])
        if i < 8:
            add(f"refusal-{i+1:02d}-vendor", f"{project} 项目的云厂商是哪家？若档案未提供请明确说明。", "refusal", "档案未提供，无法确定", ["未提供"], [text_id])
    assert len(cases) == 120
    return cases


def main() -> None:
    write_documents()
    cases = build_cases()
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DATASET_PATH.open("w", encoding="utf-8") as output:
        for item in cases:
            output.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"generated {len(PROJECTS)} documents and {len(cases)} cases")


if __name__ == "__main__":
    main()
