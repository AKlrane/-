import pandas as pd
import json
from typing import List

# === 定义 Sector 类 ===
class Sector:
    def __init__(self, id: int, name: str, suppliers: List[str], consumers: List[str]) -> None:
        self.id: int = id
        self.name: str = name
        self.suppliers: List[str] = suppliers
        self.consumers: List[str] = consumers

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "suppliers": self.suppliers,
            "consumers": self.consumers
        }

# === 分类函数 ===
def classify(text: str) -> str:
    if "整车" in text or "汽车制造" in text:
        return "OEM"
    elif "电池" in text or "电机" in text:
        return "Battery/Motor"
    elif "传感" in text or "电子" in text or "控制系统" in text:
        return "Electronics"
    elif "零部件" in text or "配件" in text or "车身" in text or "挂车" in text:
        return "Parts"
    elif "钢铁" in text or "橡胶" in text or "塑料" in text or "化工" in text:
        return "Raw"
    elif "销售" in text or "维修" in text or "租赁" in text:
        return "Service"
    else:
        return "Other"

# === 主程序 ===
if __name__ == "__main__":
    # 1. 读取 Excel
    file_path = r"C:\Users\31398\Desktop\5数据资料 5-天津新能源智能网联汽车产业集群数据集.xlsx"
    df = pd.read_excel(file_path, sheet_name="集群企业名单")

    # 2. 分类
    df["Category"] = df.apply(lambda row: classify(str(row["国标行业小类"]) + str(row["经营范围"])), axis=1)

    # 3. 保存企业分类结果（CSV）
    df[["企业名称", "Category"]].to_csv("企业分类结果.csv", index=False, encoding="utf-8-sig")
    print("✅ 已生成：企业分类结果.csv")

    # 4. 定义环节关系
    relation_map = {
        "Raw": {"suppliers": [], "consumers": ["Parts","Electronics","Battery/Motor"]},
        "Parts": {"suppliers": ["Raw"], "consumers": ["OEM"]},
        "Electronics": {"suppliers": ["Raw"], "consumers": ["OEM"]},
        "Battery/Motor": {"suppliers": ["Raw"], "consumers": ["OEM"]},
        "OEM": {"suppliers": ["Parts","Electronics","Battery/Motor"], "consumers": ["Service"]},
        "Service": {"suppliers": ["OEM"], "consumers": []},
        "Other": {"suppliers": [], "consumers": []}
    }

    # 5. 构建 Sector 对象列表
    sector_relations = []
    for i, category in enumerate(relation_map.keys()):
        sector_relations.append(
            Sector(
                id=i,
                name=category,
                suppliers=relation_map[category]["suppliers"],
                consumers=relation_map[category]["consumers"]
            )
        )

    # 6. 保存 Sector 对象（JSON）
    with open("环节上下游关系.json", "w", encoding="utf-8") as f:
        json.dump([s.to_dict() for s in sector_relations], f, ensure_ascii=False, indent=2)
    print("✅ 已生成：环节上下游关系.json")

    # 7. 保存企业分类结果（JSON）
    with open("企业分类结果.json", "w", encoding="utf-8") as f:
        json.dump(df[["企业名称","Category"]].to_dict(orient="records"), f, ensure_ascii=False, indent=2)
    print("✅ 已生成：企业分类结果.json")
