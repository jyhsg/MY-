# 对极几何实践作业（必做）

项目包含两个必做任务：给定匹配点估计 F 并绘制极点极线；自动特征检测、匹配、外点处理与 F 估计。

## 运行

建议 Python 3.10 或更新版本。

```powershell
python -m pip install -r requirements.txt
python run_experiments.py --mode all
```

本机已有依赖的环境为 Anaconda，可直接使用：

```powershell
& 'D:\soft\anaconda\python.exe' run_experiments.py --mode all
```

`--mode given` 只运行第一个任务，`--mode auto` 只运行第二个任务。默认参数为 `--ratio 0.75 --threshold 1.5 --seed 42`，其中 threshold 是 Sampson 距离平方根的像素阈值。模式 all 同时更新 `report/results.tex`。

## 数据与坐标

`data/left_original.jpg` 和 `right_original.jpg` 为从作业 DOCX 内提取的 384×288 原图，与给定匹配点范围一致。另附的约 134×100 JPG 为缩略图，不用于计算。`data/h_matches.txt` 为原始匹配数据的副本，共 168 行。

虽然题目将坐标称为“行、列”，实际第一、三列超过图像高度但小于图像宽度，故按 `xL yL xR yR` 读取。保留文件原值，不附加像素原点偏移。统一采用 `xR.T @ F @ xL = 0`；右极线为 `F @ xL`，左极线为 `F.T @ xR`。

准备好的 data 目录可直接运行。缺失时，程序会从原始作业材料中提取所需数据，不改动原始材料。

## 代码

- `src/fundamental.py`：归一化八点法、Sampson 距离、极点与基本误差统计。
- `src/ransac.py`：第二个任务的外点处理与一致集重估。
- `src/feature_matching.py`：SIFT、双向比率检验和匹配筛选。
- `src/visualization.py`：特征点、匹配、极线与极点绘制。
- `run_experiments.py`：运行两个必做任务并保存结果。

## 结果

`outputs/given_matches/` 对应第一个任务：

- `F.txt`：基础矩阵。
- `epilines.png`、`epilines_all.png`：16 对展示点和全部给定点的极线。
- `epipole_left.png`、`epipole_right.png`：左右独立极点图，红色为极点、蓝色为极线。极点在原图外，故扩展显示范围。

`outputs/automatic_matching/` 对应第二个任务：

- `keypoints.png`：SIFT 检测结果。
- `ratio_matches.png`：比率检验后的候选匹配（Candidate match）。
- `ransac_matches.png`：RANSAC 内点与外点。
- `matches_with_inlier_mask.txt`：匹配坐标和最终内点掩码。
- `F.txt`：一致集重估后的基础矩阵。
- `epilines.png`、`epilines_all.png`：内点的极线结果。

`outputs/metrics.json` 保存两个任务的矩阵、点数与基本误差统计。内点掩码是基于几何阈值的分类，不是人工真值。

## 报告与打包

在项目目录执行：

```powershell
python build_report.py
python package_submission.py
```

需要已安装 XeLaTeX 和中文字体。`report/main.tex` 是报告正文，`report/results.tex` 保存实际运行数字，`report/main.pdf` 是编译后的报告。源码引用 outputs 中的图片，需保留目录结构。

`submission_compulsory.zip` 包含代码、数据、必做实验结果、LaTeX 源码和 PDF。提交前按课程要求改为“学号+姓名+对极几何实践作业（必做）.zip”。
