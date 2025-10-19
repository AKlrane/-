import pandas as pd
import json

class Sector:
    def __init__(self, id: int, name: str, suppliers: list[str], consumers: list[str]) -> None:
        self.id = id
        self.name = name
        self.suppliers = suppliers
        self.consumers = consumers

    def __repr__(self):
        return f"Sector(id={self.id}, name='{self.name}', suppliers={self.suppliers}, consumers={self.consumers})"
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "suppliers": self.suppliers,
            "consumers": self.consumers
        }


file_path = "5数据资料 5-天津新能源智能网联汽车产业集群数据集.xlsx"
sheet_name = "集群企业名单"
df = pd.read_excel(file_path, sheet_name=sheet_name)



def classify_v4(text: str) -> str:
    # OEM 优先级最高
    if any(k in text for k in ["整车","汽车制造","新能源车","电动车","观光车","客车制造","整车装配"]):
        return "OEM"
    elif any(k in text for k in ["钢铁","橡胶","塑料","化工","原料","压延"]):
        return "Raw"
    elif any(k in text for k in ["紧固件","螺丝","螺栓","螺母","轴承","散热器","刮水器","附件","装饰件"]):
        return "Small-Parts"
    elif any(k in text for k in ["车身","底盘","挂车","客车","改装","玻璃"]):
        return "Large-Parts"
    elif any(k in text for k in ["电池","蓄电池","电机","电控","控制系统","传感","芯片","电子","软件","原动设备"]):
        return "Electronics"
    elif any(k in text for k in ["销售","维修","修理","租赁","清障","洗车","服务"]):
        return "Service"
    else:
        return "Other"



df["Category"] = df.apply(lambda row: classify_v4(str(row["国标行业小类"]) + str(row["经营范围"])), axis=1)

company_classification = df[["企业名称","Category"]]




category_counts = df["Category"].value_counts()




sector_relations: list[Sector] = [
    Sector(id=0, name="Raw", suppliers=[], consumers=["Small-Parts","Large-Parts","Electronics"]),
    Sector(id=1, name="Small-Parts", suppliers=["Raw"], consumers=["OEM"]),
    Sector(id=2, name="Large-Parts", suppliers=["Raw"], consumers=["OEM"]),
    Sector(id=3, name="Electronics", suppliers=["Raw"], consumers=["OEM"]),
    Sector(id=4, name="OEM", suppliers=["Small-Parts","Large-Parts","Electronics"], consumers=["Service"]),
    Sector(id=5, name="Service", suppliers=["OEM"], consumers=[]),
    Sector(id=6, name="Other", suppliers=[], consumers=[]),
]




