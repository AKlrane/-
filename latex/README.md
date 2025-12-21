# LaTeX 实验报告

## 文件说明

- `report.tex`: 主报告文件

## 编译方法

### 使用 XeLaTeX（推荐，支持中文）

```bash
cd latex
xelatex report.tex
xelatex report.tex  # 需要编译两次以生成目录和交叉引用
```

### 使用 pdfLaTeX

```bash
cd latex
pdflatex report.tex
pdflatex report.tex  # 需要编译两次
```

## 需要的 LaTeX 包

报告使用了以下 LaTeX 包：
- `ctex`: 中文支持
- `amsmath`, `amssymb`: 数学公式
- `graphicx`: 图片插入
- `hyperref`: 超链接
- `geometry`: 页面设置
- `enumitem`: 列表格式
- `booktabs`: 表格格式
- `float`: 浮动体控制

如果缺少包，可以使用以下命令安装（以TeX Live为例）：

```bash
tlmgr install ctex amsmath amssymb graphicx hyperref geometry enumitem booktabs float
```

## 报告结构

1. **背景介绍**
   - 产业集群模拟任务说明
   - 天津产业集群数据介绍
   - 研究目标

2. **方法**
   - 环境设定（产业链、交易流程、成本设计等）
   - 强化学习设计（模型架构、观察空间、动作空间、算法框架、奖励设定、Loss设计）

3. **实验**（待补充）

4. **结论**（待补充）

## 注意事项

- 报告使用UTF-8编码
- 需要支持中文的LaTeX发行版（如TeX Live、MiKTeX）
- 如果编译时出现中文显示问题，请确保使用XeLaTeX编译器

