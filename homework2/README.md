# 机器视觉作业：单幅图像测量

正式报告为 `实验报告.pdf`，包含可点击的自动目录（章节、小节、附录和参考文献）。行人尺度固定为1.70 m，路灯估计高度约4.53 m。分工说明单独保存在 `分工说明.txt`。

## 文件说明

- `1.jpg`：原始实验图片。
- `analyze.py`：测量计算、像素误差实验和作图。
- `verify_geometry.py`：合成投影几何验证。
- `requirements.txt`：Python依赖。
- `results/`：标注、测量数据、模拟数据及支撑图片，全部保留以便复现。
- `实验报告.pdf`：正式报告成品。

- `分工说明.txt`：本人和AI的工作分工。


## 重新计算

在项目目录执行：

```powershell
python -m pip install -r requirements.txt
python analyze.py
python verify_geometry.py
```


