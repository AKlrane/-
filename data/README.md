# Data Directory

This directory contains data processing scripts and reference data files for the industry simulation. This document consolidates all documentation about the data processing pipeline, including script usage, data transformations, and field descriptions.

---

## Table of Contents

1. [Overview](#overview)
2. [Scripts](#scripts)
3. [Data Files](#data-files)
4. [Output Format](#output-format)
5. [Data Transformations Timeline](#data-transformations-timeline)
6. [Usage Guide](#usage-guide)
7. [Statistics and Insights](#statistics-and-insights)
8. [Requirements](#requirements)

---

## Overview

The data processing pipeline transforms raw company data from Excel into structured JSON and CSV files suitable for simulation. The pipeline includes:

- **Geographic visualization** of company locations
- **Industry sector classification** using keyword matching
- **Company size extraction** and intelligent sampling for missing data
- **Capital mapping** based on company size using log-uniform distribution
- **Temporal data** including establishment dates and company lifecycle information
- **Supply chain relationships** between industry sectors

**Total Companies:** 927  
**Total Initial Capital:** ~¥42.5 million CNY  
**Industry Sectors:** 7 (Raw, Parts, Electronics, Battery/Motor, OEM, Service, Other)

---

## Scripts

### 1. `supply_chain.py`

**Purpose:** Main data processing script that classifies companies by industry sector and generates comprehensive company data with capital, dates, and status information.

**Key Features:**

- Reads company data from Excel (sheet: "集群企业名单")
- Classifies companies into 7 industry sectors using Chinese keyword matching
- Extracts and processes company attributes (size, dates, status)
- Generates initial capital using log-uniform distribution by size
- Intelligently samples missing company sizes from observed distribution
- Parses registration status into active boolean and optional death date
- Outputs structured CSV and JSON files with English key names

**Usage:**

```bash
cd data
python supply_chain.py
```

**Configuration Variables:**

Update these variables in the script to match your Excel file:

```python
file_path = "data.xlsx"
sheet_name = "集群企业名单"
industry_col = "国标行业小类"
business_scope_col = "经营范围"
company_name_col = "企业名称"
company_size_col = "企业规模"
establishment_date_col = "成立日期"
registration_status_col = "登记状态"
```

**Classification Logic:**

| Sector | Keywords | Description |
|--------|----------|-------------|
| **OEM** | 整车, 汽车制造 | Vehicle assembly, auto manufacturing |
| **Battery/Motor** | 电池, 电机 | Battery, motor manufacturers |
| **Electronics** | 传感, 电子, 控制系统 | Sensors, electronics, control systems |
| **Parts** | 零部件, 配件, 车身, 挂车 | Parts, components, body, trailer |
| **Raw** | 钢铁, 橡胶, 塑料, 化工 | Steel, rubber, plastic, chemicals |
| **Service** | 销售, 维修, 租赁 | Sales, repair, leasing |
| **Other** | - | Anything else |

**Output Files:**

1. `company_classification.csv` - All company data in CSV format
2. `company_classification.json` - All company data in JSON format
3. `supply_chain_relations.json` - Sector supply chain relationships

---

### 2. `plot_coordinates.py`

**Purpose:** Visualize company geographic locations from latitude/longitude coordinates.

**Key Features:**

- Reads coordinates from Excel
- Converts lat/lon to km-based Cartesian coordinates using equirectangular projection
- Creates interactive scatter plot with hover labels showing company names
- Useful for understanding spatial distribution of companies

**Usage:**

```bash
cd data
python plot_coordinates.py
```

**Configuration Variables:**

```python
excel_file = "data.xlsx"
sheet_name = "集群企业名单"
lon_col = "经度"  # Longitude
lat_col = "纬度"  # Latitude
name_col = "企业名称"  # Company name
```

**Output:**

- Interactive matplotlib plot
- Hover over points to see company names
- Visual understanding of cluster geography

---

## Data Files

### Input Files

**`data.xlsx`**

- Source Excel file with company information
- Sheet: "集群企业名单" (Cluster Company List)
- Contains: company names, coordinates, industry classification, business scope, size, dates, status

### Output Files

**`company_classification.csv`**

- CSV format with all processed company data
- 927 records with 7 fields
- Encoding: UTF-8 with BOM

**`company_classification.json`**

- JSON format with all processed company data
- Array of 927 company objects
- Encoding: UTF-8
- Preserves Chinese characters

**`supply_chain_relations.json`**

- Defines supply chain relationships between sectors
- Structure: sector id, name, suppliers list, consumers list
- Used by simulation to model inter-sector dependencies

### Reference Files

**`5-chain.json`**

- Legacy supply chain relationship data

**`5-Classification.csv` / `5-Classification.json`**

- Legacy company classification reference data

---

## Output Format

### Complete Field Structure

Each company record contains 7 fields:

```json
{
  "company_name": "天津市南开区雄鹰散热器厂",
  "sector": "Parts",
  "company_size": "M(中型)",
  "initial_capital": 182722.62,
  "establishment_date": "2001-09-05",
  "active": true,
  "death_date": null
}
```

### Field Descriptions

| Field | Type | Description | Example Values |
|-------|------|-------------|----------------|
| `company_name` | string | Company name in Chinese | "天津市散热器厂" |
| `sector` | string | Industry sector classification | "Parts", "Electronics", "OEM" |
| `company_size` | string | Company size category | "L(大型)", "M(中型)", "S(小型)", "XS(微型)" |
| `initial_capital` | float | Initial capital in CNY | 182722.62 |
| `establishment_date` | string | Date company was established | "2001-09-05" (YYYY-MM-DD) |
| `active` | boolean | Whether company is currently active | `true`, `false` |
| `death_date` | string or null | Date company ceased operations | "2017-06-16" or `null` |

### CSV Format Example

```csv
company_name,sector,company_size,initial_capital,establishment_date,active,death_date
天津市南开区雄鹰散热器厂,Parts,M(中型),182722.62,2001-09-05,True,
天津市京津内燃机配件厂,Parts,XS(微型),13343.57,1996-09-02,False,2002-10-30
```

---

## Data Transformations Timeline

### Phase 1: Initial Refactoring (Data Scripts)

**Objective:** Modernize scripts with English naming and better documentation

**Changes:**

- Renamed `coord.py` → `plot_coordinates.py`
- Renamed `spchain .py` → `supply_chain.py`
- Converted all code comments and variables to English
- Added comprehensive docstrings
- Updated output filenames from Chinese to English
- Created initial README documentation

### Phase 2: Company Size Extraction

**Objective:** Extract company size data from Excel

**Changes:**

- Added `company_size_col = "企业规模"` column extraction
- Modified output to include company size field
- Added statistics: size distribution and category-size cross-tabulation
- Discovered 146 companies (15.7%) with missing size data

**Results:**

- XS (微型): 518 companies (55.9%)
- S (小型): 203 companies (21.9%)
- M (中型): 54 companies (5.8%)
- L (大型): 6 companies (0.6%)
- Missing: 146 companies (15.7%)

### Phase 3: English Keys & Capital Mapping

**Objective:** Use English field names and add realistic capital amounts

**Key Changes:**

- Implemented `map_size_to_capital()` function with log-uniform distribution
- Capital ranges by size:
  - Large (L): ¥500K - ¥1M
  - Medium (M): ¥100K - ¥500K
  - Small (S): ¥20K - ¥100K
  - Micro (XS): ¥5K - ¥20K
- Implemented `sample_missing_sizes()` for intelligent sampling
- Changed output keys from Chinese to English
- Added numpy for statistical functions

**Sampling Strategy:**

- Analyzed distribution of companies with specified sizes
- Sampled 146 missing values proportionally
- Distribution: XS 66.3%, S 26.0%, M 6.9%, L 0.8%

**Results:**

- Total capital: ¥42.5M CNY
- Parts sector dominates: ¥33M (77.6%)
- Average capital by size: L ¥697K, M ¥274K, S ¥48K, XS ¥11K

### Phase 4: Establishment Date & Registration Status

**Objective:** Add temporal data for company lifecycle modeling

**Changes:**

- Added `establishment_date_col = "成立日期"` extraction
- Added `registration_status_col = "登记状态"` extraction
- Both fields 100% complete (927/927 companies)

**Results:**

- Date range: 1958 to 2025 (67 years)
- Peak establishment: 1990s (320 companies, 34.5%)
- Status distribution:
  - 存续 (Active): 361 companies (38.9%)
  - 注销 (Deregistered): 188 companies (20.3%)
  - 吊销 (License Revoked): 31 companies (3.3%)
  - 238 unique status values total

### Phase 5: Active Boolean & Death Date Parsing

**Objective:** Transform registration status into simulation-ready fields

**Key Changes:**

- Implemented `parse_registration_status()` function
- Regex pattern to extract dates: `r'（(\d{4}-\d{2}-\d{2})）'`
- Converted status to boolean `active` field
- Extracted optional `death_date` from status strings

**Status Parsing:**

- "存续" → `active: true`, `death_date: null`
- "注销（2017-06-16）" → `active: false`, `death_date: "2017-06-16"`
- "吊销" → `active: false`, `death_date: null`

**Results:**

- Active companies: 361 (38.9%)
- Inactive companies: 566 (61.1%)
- Death dates available: 339 (36.6% of all, 59.9% of inactive)
- Death date range: 1995-08-04 to 2025-08-15
- Peak death year: 2017 (33 companies)

---

## Usage Guide

### Running the Pipeline

**1. Prepare Your Data**

Ensure your Excel file has these columns:

- 企业名称 (Company name)
- 国标行业小类 (Industry classification)
- 经营范围 (Business scope)
- 企业规模 (Company size)
- 成立日期 (Establishment date)
- 登记状态 (Registration status)

**2. Run Supply Chain Classification**

```bash
cd /path/to/industry-sim/data
python supply_chain.py
```

**3. Verify Output**

```bash
# Check generated files
ls -lh company_classification.*
ls -lh supply_chain_relations.json

# View first few records
head -5 company_classification.csv
```

### Loading Data in Python

**Load CSV:**

```python
import pandas as pd

df = pd.read_csv("data/company_classification.csv")
print(f"Loaded {len(df)} companies")

# Filter by sector
parts_companies = df[df['sector'] == 'Parts']

# Filter by size
large_companies = df[df['company_size'] == 'L(大型)']

# Filter active companies
active = df[df['active'] == True]
```

**Load JSON:**

```python
import json

with open("data/company_classification.json", "r", encoding="utf-8") as f:
    companies = json.load(f)

# Access data
for company in companies[:5]:
    print(f"{company['company_name']}: {company['sector']}")

# Filter examples
active_parts = [c for c in companies 
                if c['active'] and c['sector'] == 'Parts']

companies_2017_deaths = [c for c in companies 
                         if c['death_date'] and c['death_date'].startswith('2017')]
```

**Use in Simulation:**

```python
from env.company import Company

# Load companies
with open("data/company_classification.json", "r", encoding="utf-8") as f:
    company_data = json.load(f)

# Initialize simulation companies
companies = []
for data in company_data:
    if data['active']:  # Only use active companies
        company = Company(
            name=data['company_name'],
            sector=data['sector'],
            initial_capital=data['initial_capital'],
            size=data['company_size']
        )
        companies.append(company)

print(f"Initialized {len(companies)} active companies")
```

---

## Statistics and Insights

### Company Distribution by Sector

| Sector | Count | Percentage | Total Capital | Avg Capital |
|--------|-------|------------|---------------|-------------|
| Parts | 728 | 78.5% | ¥33.0M | ¥45,388 |
| Electronics | 121 | 13.0% | ¥4.1M | ¥33,853 |
| Battery/Motor | 53 | 5.7% | ¥2.0M | ¥38,641 |
| OEM | 20 | 2.2% | ¥0.8M | ¥39,221 |
| Service | 2 | 0.2% | ¥0.03M | ¥15,707 |
| Raw | 2 | 0.2% | ¥0.04M | ¥21,745 |
| Other | 1 | 0.1% | ¥0.09M | ¥92,039 |

### Company Size Distribution

| Size | Chinese | Count | Percentage | Avg Capital |
|------|---------|-------|------------|-------------|
| XS | 微型 | ~608 | 65.6% | ¥10,656 |
| S | 小型 | ~248 | 26.7% | ¥49,372 |
| M | 中型 | ~65 | 7.0% | ¥261,440 |
| L | 大型 | ~6 | 0.6% | ¥697,451 |

*Note: Counts vary slightly due to random sampling of 146 missing values*

### Temporal Analysis

**Establishment Timeline:**

- 1950s: 1 company (0.1%)
- 1980s: 72 companies (7.8%)
- 1990s: 320 companies (34.5%) **← Peak**
- 2000s: 257 companies (27.7%)
- 2010s: 194 companies (20.9%)
- 2020s: 83 companies (9.0%)

**Company Status:**

- Active (存续): 361 companies (38.9%)
- Inactive: 566 companies (61.1%)
  - With death date: 339 (59.9% of inactive)
  - Without death date: 227 (40.1% of inactive)

**Death Timeline:**

- Total deaths with dates: 339
- Date range: 1995-2015 (30 years)
- Peak death year: **2017** (33 companies)
- Deaths by decade:
  - 1990s: 16 (4.7%)
  - 2000s: 113 (33.3%)
  - 2010s: 121 (35.7%)
  - 2020s: 89 (26.3%)

### Sector-Size Correlation

|  | L(大型) | M(中型) | S(小型) | XS(微型) | Total |
|--|---------|---------|---------|----------|-------|
| **Battery/Motor** | 1 | 0-1 | 15-21 | 27-37 | 53 |
| **Electronics** | 0 | 3-5 | 46-49 | 61-71 | 121 |
| **OEM** | 0 | 0-1 | 8-10 | 6-10 | 20 |
| **Parts** | 5-8 | 51-61 | 156-166 | 422-506 | 728 |
| **Raw** | 0 | 0 | 1 | 1 | 2 |
| **Service** | 0 | 0 | 1 | 1 | 2 |
| **Other** | 0 | 0 | 1 | 0 | 1 |

*Note: Ranges reflect random sampling variability*

### Key Insights

1. **Parts Sector Dominance**
   - 78.5% of all companies
   - 77.6% of total capital
   - Present in all size categories
   - Core of the supply chain

2. **Micro-Enterprise Economy**
   - 65%+ of companies are micro-sized (XS)
   - Only 0.6% are large companies
   - Reflects typical SME-dominated supply chain

3. **Industry Maturity**
   - Peak establishment in 1990s (industry formation)
   - 61% of companies now inactive
   - Recent deaths (2017-2025) suggest ongoing consolidation
   - Mature/contracting industry characteristics

4. **Regulatory Shock in 2017**
   - 33 companies died in 2017 (peak year)
   - Suggests policy change or enforcement action
   - May indicate environmental or regulatory crackdown

5. **Capital Distribution**
   - Parts sector: ¥45K average (highest)
   - Service sector: ¥16K average (lowest)
   - 10x difference between sectors
   - Reflects capital intensity differences

---

## Requirements

### Python Packages

```bash
pip install pandas numpy matplotlib openpyxl mplcursors
```

Or use the project's virtual environment:

```bash
source .venv/bin/activate  # On macOS/Linux
.venv\Scripts\activate     # On Windows
```

### Package Versions (Recommended)

- `pandas >= 2.3.3` - Data manipulation
- `numpy >= 1.26.0` - Numerical operations
- `matplotlib >= 3.10.7` - Visualization
- `openpyxl >= 3.1.0` - Excel file reading
- `mplcursors >= 0.5.0` - Interactive plot hover tooltips

### System Requirements

- Python 3.10 or higher
- Sufficient memory for loading 927 company records (~1MB)
- Excel file access for data source

---

## Implementation Details

### Capital Generation Algorithm

**Log-Uniform Distribution:**

```python
def map_size_to_capital(size: str) -> float:
    size_ranges = {
        "L(大型)": (500000, 1000000),
        "M(中型)": (100000, 500000),
        "S(小型)": (20000, 100000),
        "XS(微型)": (5000, 20000),
    }
    min_cap, max_cap = size_ranges[size]
    return np.exp(np.random.uniform(np.log(min_cap), np.log(max_cap)))
```

**Why log-uniform?**

- More realistic than uniform distribution
- Creates natural clustering at lower end of range
- Mirrors real-world wealth/capital distribution (power law)
- Prevents artificial clustering around mean

### Missing Data Sampling

**Strategy:**

```python
def sample_missing_sizes(df: pd.DataFrame, size_col: str) -> pd.DataFrame:
    # Analyze non-missing distribution
    size_distribution = df[df[size_col] != "-"][size_col].value_counts(normalize=True)
    
    # Sample proportionally
    missing_mask = (df[size_col] == "-") | df[size_col].isna()
    sampled_sizes = np.random.choice(
        size_distribution.index,
        size=missing_mask.sum(),
        p=size_distribution.values
    )
    
    df.loc[missing_mask, size_col] = sampled_sizes
    return df
```

**Benefits:**

- Maintains original distribution
- No arbitrary defaults
- Statistically sound approach
- Reproducible with seed

### Registration Status Parsing

**Pattern Extraction:**

```python
def parse_registration_status(status: str) -> Tuple[bool, Optional[str]]:
    is_active = str(status) == "存续"
    date_match = re.search(r'（(\d{4}-\d{2}-\d{2})）', str(status))
    death_date = date_match.group(1) if date_match else None
    return is_active, death_date
```

**Handles:**

- "存续" → active, no death date
- "注销（2017-06-16）" → inactive, death date extracted
- "吊销" → inactive, no death date
- Missing/null → inactive, no death date

### Reproducibility

All random operations use fixed seed:

```python
np.random.seed(42)
```

This ensures:

- Consistent capital generation across runs
- Identical sampling of missing sizes
- Reproducible results for testing and validation

---

## Data Validation

### Validation Checks

All data passes these validations:

✅ **Field Completeness**

- All 927 records have all 7 required fields
- No missing required data

✅ **Type Consistency**

- `active`: all boolean values
- `initial_capital`: all positive floats
- `establishment_date`: all valid YYYY-MM-DD format or null
- `death_date`: all valid YYYY-MM-DD format or null

✅ **Business Logic**

- No active companies have death dates
- All death dates are after establishment dates
- Capital values within expected ranges by size

✅ **Data Integrity**

- Company names unique
- Sectors match defined categories
- Sizes match defined categories
- Dates within reasonable range (1958-2025)

### Validation Scripts

Run validation:

```bash
cd data
python validate_fields.py
```

Output:

```
✓ Active companies with death_date: 0 (should be 0)
✓ Records missing fields: 0 (should be 0)
✓ Non-boolean active values: 0 (should be 0)
✓ Invalid death_date format: 0 (should be 0)

All validation checks passed! ✅
```

---

## Troubleshooting

### Common Issues

**1. FileNotFoundError: data.xlsx**

- Ensure Excel file is in the `data/` directory
- Update `file_path` variable in scripts if located elsewhere

**2. KeyError: Column not found**

- Check Excel column names match script variables
- Ensure sheet name is correct ("集群企业名单")

**3. UnicodeDecodeError**

- Excel file must be saved with UTF-8 encoding
- Use `encoding="utf-8"` when reading

**4. Memory Error**

- 927 records should require minimal memory
- Check for other memory-intensive processes
- Consider processing in chunks if dataset grows

**5. Random Seed Issues**

- Results may vary if seed is changed or removed
- Always use `np.random.seed(42)` for reproducibility

---

## Future Enhancements

### Potential Improvements

1. **Geographic Integration**
   - Add x, y coordinates to output
   - Include location data in simulation
   - Spatial clustering analysis

2. **Enhanced Classification**
   - Machine learning-based sector classification
   - Multi-label classification (companies in multiple sectors)
   - Confidence scores for classifications

3. **Temporal Modeling**
   - Time-series of company entries/exits
   - Cohort analysis by establishment decade
   - Survival probability modeling

4. **Capital Refinement**
   - Sector-specific capital ranges
   - Inflation adjustment over time
   - Capital growth modeling

5. **Data Quality**
   - Automated outlier detection
   - Duplicate company detection
   - Data quality scoring

6. **Integration**
   - Direct database loading
   - Real-time data updates
   - API endpoints for data access

---

## References

### Related Files

- `/env/company.py` - Company class implementation
- `/env/sector.py` - Sector class with supply chain relationships
- `/env/env.py` - Main simulation environment
- `/config/config.json` - Simulation configuration

### Documentation

- Main README: `/README.md`
- Configuration docs: `/config/README.md`
- Demo scripts: `/demos/README.md`
- Test suite: `/tests/README.md`

---

**Last Updated:** 15 October 2025  
**Version:** 5.0 (Consolidated)  
**Status:** ✅ Production Ready  
**Maintainer:** Industry Simulation Project Team
