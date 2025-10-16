"""
Supply Chain Classification Script

Reads company data from Excel and classifies companies by industry sector,
then generates supply chain relationship files with English key names and capital mapping.
"""

import pandas as pd
import json
import numpy as np
import re
import math
from typing import List, Optional, Tuple

# === Sector Class Definition ===
class Sector:
    """Represents an industry sector with supply chain relationships."""
    
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

# === Classification Function ===
def classify(text: str) -> str:
    """
    Classify company into industry sector based on text description.
    
    Chinese keywords:
    - 整车/汽车制造 -> OEM
    - 电池/电机 -> Battery/Motor
    - 传感/电子/控制系统 -> Electronics
    - 零部件/配件/车身/挂车 -> Parts
    - 钢铁/橡胶/塑料/化工 -> Raw
    - 销售/维修/租赁 -> Service
    """
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

# === Capital Mapping Function ===
def map_size_to_capital(size: str) -> float:
    """
    Map company size to initial capital amount.
    
    Size categories and capital ranges (in CNY):
    - L (大型/Large): 500,000 - 1,000,000
    - M (中型/Medium): 100,000 - 500,000
    - S (小型/Small): 20,000 - 100,000
    - XS (微型/Micro): 5,000 - 20,000
    - Default/Unknown: 10,000 - 100,000
    """
    size_capital_map = {
        "L(大型)": (500000, 1000000),
        "M(中型)": (100000, 500000),
        "S(小型)": (20000, 100000),
        "XS(微型)": (5000, 20000),
    }
    
    if size in size_capital_map:
        min_cap, max_cap = size_capital_map[size]
        # Use log-uniform distribution for more realistic capital distribution
        return np.exp(np.random.uniform(np.log(min_cap), np.log(max_cap)))
    else:
        # Default range for unknown sizes
        return np.exp(np.random.uniform(np.log(10000), np.log(100000)))

def sample_missing_sizes(df: pd.DataFrame, size_col: str) -> pd.DataFrame:
    """
    Sample company sizes for rows with missing data based on the distribution
    of specified sizes.
    
    Args:
        df: DataFrame with company data
        size_col: Name of the size column
        
    Returns:
        DataFrame with sampled sizes for missing values
    """
    # Get the distribution of non-missing sizes
    size_distribution = df[df[size_col] != "-"][size_col].value_counts(normalize=True)
    
    # Find rows with missing sizes
    missing_mask = (df[size_col] == "-") | df[size_col].isna()
    n_missing = missing_mask.sum()
    
    if n_missing > 0 and len(size_distribution) > 0:
        # Sample from the distribution
        sampled_sizes = np.random.choice(
            size_distribution.index, 
            size=n_missing, 
            p=size_distribution.values
        )
        
        # Fill in the missing values
        df.loc[missing_mask, size_col] = sampled_sizes
        print(f"✓ Sampled {n_missing} missing company sizes from distribution:")
        print(size_distribution.to_string())
    
    return df


def parse_registration_status(status: str) -> Tuple[bool, Optional[str]]:
    """
    Parse registration status to extract active state and death date.
    
    Args:
        status: Registration status string (e.g., "存续", "注销", "吊销（2017-06-16）")
        
    Returns:
        Tuple of (is_active, death_date)
        - is_active: True if status is "存续" (active), False otherwise
        - death_date: Date string if found in parentheses, None otherwise
    """
    if pd.isna(status):
        return False, None
    
    status_str = str(status)
    
    # Check if active (存续 means "continuing/active")
    is_active = status_str == "存续"
    
    # Extract date from parentheses if present (e.g., "注销（2017-06-16）")
    date_match = re.search(r'（(\d{4}-\d{2}-\d{2})）', status_str)
    death_date = date_match.group(1) if date_match else None
    
    return is_active, death_date


def convert_latlon_to_xy(df: pd.DataFrame, lat_col: str, lon_col: str) -> pd.DataFrame:
    """
    Convert latitude/longitude coordinates to x,y coordinates in km.
    Uses the same projection method as plot_coordinates.py.
    
    Args:
        df: DataFrame with lat/lon columns
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        
    Returns:
        DataFrame with added 'x' and 'y' columns (in km, relative to center)
    """
    # Filter out rows with missing coordinates
    valid_coords = df[[lat_col, lon_col]].notna().all(axis=1)
    
    if valid_coords.sum() == 0:
        print("⚠ No valid coordinates found, skipping coordinate conversion")
        df['x'] = None
        df['y'] = None
        return df
    
    # Calculate center point (using only valid coordinates)
    lat_center = df.loc[valid_coords, lat_col].mean()
    lon_center = df.loc[valid_coords, lon_col].mean()
    
    print(f"✓ Center point (lat, lon): {lat_center:.6f}, {lon_center:.6f}")
    
    # Earth radius in km
    R = 6371
    lat0 = math.radians(lat_center)
    lon0 = math.radians(lon_center)
    
    def project(lat, lon):
        """Project lat/lon to x,y coordinates in km relative to center."""
        if pd.isna(lat) or pd.isna(lon):
            return None, None
        
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        x = R * (lon_rad - lon0) * math.cos(lat0)
        y = R * (lat_rad - lat0)
        return x, y
    
    # Apply projection to all rows
    coordinates = df.apply(
        lambda row: project(row[lat_col], row[lon_col]), 
        axis=1
    )
    
    df['x'], df['y'] = zip(*coordinates)
    
    # Print statistics
    valid_x = df['x'].notna().sum()
    print(f"✓ Converted {valid_x} / {len(df)} companies to x,y coordinates")
    
    if valid_x > 0:
        print(f"  X range: {df['x'].min():.2f} to {df['x'].max():.2f} km")
        print(f"  Y range: {df['y'].min():.2f} to {df['y'].max():.2f} km")
    
    return df


# === Main Program ===
if __name__ == "__main__":
    # 1. Read Excel file
    file_path = "data.xlsx"  # Update this path to your Excel file
    sheet_name = "集群企业名单"  # Update this to match your sheet name
    
    # Column names (update these to match your Excel file)
    industry_col = "国标行业小类"
    business_scope_col = "经营范围"
    company_name_col = "企业名称"
    company_size_col = "企业规模"  # Company size column
    establishment_date_col = "成立日期"  # Establishment date column
    registration_status_col = "登记状态"  # Registration status column
    lat_col = "纬度"  # Latitude column
    lon_col = "经度"  # Longitude column
    
    print("="*70)
    print("COMPANY DATA PROCESSING")
    print("="*70)
    print(f"Reading from: {file_path}")
    print(f"Sheet: {sheet_name}")
    
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    print(f"✓ Loaded {len(df)} companies")

    # 2. Classify companies by sector
    df["Category"] = df.apply(
        lambda row: classify(str(row[industry_col]) + str(row[business_scope_col])), 
        axis=1
    )

    # 3. Process company size column
    if company_size_col in df.columns:
        print(f"✓ Found '{company_size_col}' column")
        
        # Replace various forms of missing data with "-"
        df[company_size_col] = df[company_size_col].fillna("-")
        
        # Sample missing sizes based on distribution
        df = sample_missing_sizes(df, company_size_col)
    else:
        print(f"⚠ Column '{company_size_col}' not found, skipping size processing")
        df[company_size_col] = "XS(微型)"  # Default to micro if no size column
    
    # 4. Generate capital based on company size
    print("\n✓ Generating initial capital based on company size...")
    np.random.seed(42)  # For reproducibility
    df["initial_capital"] = df[company_size_col].apply(map_size_to_capital)
    
    # 5. Convert latitude/longitude to x,y coordinates
    print("\n✓ Converting latitude/longitude to x,y coordinates...")
    if lat_col in df.columns and lon_col in df.columns:
        df = convert_latlon_to_xy(df, lat_col, lon_col)
    else:
        print(f"⚠ Columns '{lat_col}' or '{lon_col}' not found, skipping coordinate conversion")
        df['x'] = None
        df['y'] = None
    
    # Prepare output data with English keys
    print("\n✓ Preparing output data...")
    output_data = []
    for _, row in df.iterrows():
        # Parse registration status to get active state and death date
        is_active, death_date = parse_registration_status(row[registration_status_col])
        
        company_dict = {
            "company_name": row[company_name_col],
            "sector": row["Category"],
            "company_size": row[company_size_col],
            "initial_capital": round(row["initial_capital"], 2),
            "establishment_date": str(row[establishment_date_col]) if pd.notna(row[establishment_date_col]) else None,
            "active": is_active,
            "death_date": death_date
        }
        
        # Add coordinates if available
        if pd.notna(row['x']) and pd.notna(row['y']):
            company_dict["x"] = round(row['x'], 4)
            company_dict["y"] = round(row['y'], 4)
        else:
            company_dict["x"] = None
            company_dict["y"] = None
        
        output_data.append(company_dict)
    
    # 6. Save company classification results (CSV)
    print("\n✓ Saving results...")
    output_csv = "company_classification.csv"
    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"✅ Generated: {output_csv} ({len(output_df)} companies)")

    # 7. Define supply chain relationships
    relation_map = {
        "Raw": {"suppliers": [], "consumers": ["Parts", "Electronics", "Battery/Motor"]},
        "Parts": {"suppliers": ["Raw"], "consumers": ["OEM"]},
        "Electronics": {"suppliers": ["Raw"], "consumers": ["OEM"]},
        "Battery/Motor": {"suppliers": ["Raw"], "consumers": ["OEM"]},
        "OEM": {"suppliers": ["Parts", "Electronics", "Battery/Motor"], "consumers": ["Service"]},
        "Service": {"suppliers": ["OEM"], "consumers": []},
        "Other": {"suppliers": [], "consumers": []}
    }

    # 8. Build Sector object list
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

    # 9. Save sector relationships (JSON)
    output_relations = "supply_chain_relations.json"
    with open(output_relations, "w", encoding="utf-8") as f:
        json.dump([s.to_dict() for s in sector_relations], f, ensure_ascii=False, indent=2)
    print(f"✅ Generated: {output_relations}")

    # 10. Save company classification results (JSON)
    output_json = "company_classification.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"✅ Generated: {output_json}")
    
    # 11. Print summary statistics
    print("\n" + "="*70)
    print("Classification Summary")
    print("="*70)
    print(df["Category"].value_counts().to_string())
    print("="*70)
    
    # 12. Print company size distribution
    print("\n" + "="*70)
    print("Company Size Distribution (After Sampling)")
    print("="*70)
    print(df[company_size_col].value_counts().to_string())
    print("="*70)
    
    # 13. Print capital statistics by size
    print("\n" + "="*70)
    print("Capital Statistics by Company Size")
    print("="*70)
    capital_by_size = df.groupby(company_size_col)["initial_capital"].agg([
        ("count", "count"),
        ("mean", "mean"),
        ("median", "median"),
        ("min", "min"),
        ("max", "max")
    ])
    print(capital_by_size.to_string())
    print("="*70)
    
    # 14. Cross-tabulation: Category vs Size
    print("\n" + "="*70)
    print("Category vs Company Size")
    print("="*70)
    crosstab = pd.crosstab(df["Category"], df[company_size_col], margins=True)
    print(crosstab.to_string())
    print("="*70)
    
    # 15. Print total capital by sector
    print("\n" + "="*70)
    print("Total Capital by Sector")
    print("="*70)
    capital_by_sector = df.groupby("Category")["initial_capital"].agg([
        ("count", "count"),
        ("total", "sum"),
        ("mean", "mean")
    ])
    print(capital_by_sector.to_string())
    print("="*70)
    
    # 16. Print coordinate statistics
    if 'x' in df.columns and 'y' in df.columns:
        valid_coords = df[['x', 'y']].notna().all(axis=1).sum()
        print("\n" + "="*70)
        print("Coordinate Statistics")
        print("="*70)
        print(f"Companies with valid coordinates: {valid_coords} / {len(df)} ({valid_coords/len(df)*100:.1f}%)")
        if valid_coords > 0:
            print(f"X range: {df['x'].min():.2f} to {df['x'].max():.2f} km (span: {df['x'].max() - df['x'].min():.2f} km)")
            print(f"Y range: {df['y'].min():.2f} to {df['y'].max():.2f} km (span: {df['y'].max() - df['y'].min():.2f} km)")
        print("="*70)
