# import os
# from rdkit import Chem
# from rdkit.Chem import Draw
# output_dir = './SVG'
# motifs = ["C1CC2=C(C=CS2)C3=C1C=NN3"]
# for j, motif in enumerate(motifs):
#     motif_mol = Chem.MolFromSmiles(motif)
#     if motif_mol is not None:
#         # 设置图片尺寸 (宽度, 高度)
#         size = (500, 500)
#         drawer = Draw.MolDraw2DSVG(size[0], size[1])
#         drawer.DrawMolecule(motif_mol)
#         drawer.FinishDrawing()
#         svg = drawer.GetDrawingText().replace('svg:', '')
#         img_path = os.path.join(output_dir, f"test_{motif}.svg")
#         with open(img_path, 'w') as f:
#             f.write(svg)

import matplotlib.pyplot as plt
import numpy as np

# 数据
methods = ["Only Bipartite\nNetwork",
           "No Feature\nFusion",
           "No Contrastive\nLearning",
           "AGCL-DTA"]
mse_data = [0.193, 0.177, 0.171, 0.166]  # MSE
r_m2_data = [0.726, 0.751, 0.774, 0.794]  # R_m^2

# 误差条数据
mse_error = [0.022, 0.012, 0.015, 0.009]
r_m2_error = [0.011, 0.010, 0.009, 0.005]

# 设置样式参数
bar_width = 0.3
x = np.arange(len(methods))  # X轴的位置

# 学术展示配色
colors = ['#1F77B4', '#FF7F0E']  # MSE: 深蓝色, R_m^2: 橙色
error_colors = ['#87CEEB', '#FFD700']  # 柔和误差条颜色

# 设置全局字体大小
plt.rcParams.update({
    'font.size': 30,  # 全局字体大小
    'font.family': 'Times New Roman',  # 字体风格
    'axes.labelsize': 24,  # 坐标轴标签字体大小
    'axes.titlesize': 28,  # 标题字体大小
    'xtick.labelsize': 24,  # X轴刻度字体大小
    'ytick.labelsize': 24,  # Y轴刻度字体大小
    'legend.fontsize': 28  # 图例字体大小
})

# 创建图表
fig, ax1 = plt.subplots(figsize=(12, 8))  # 增大图像尺寸以适配更大的字体

# 绘制MSE条形图
ax1.bar(x - bar_width / 2, mse_data, width=bar_width, color=colors[0], label="MSE", yerr=mse_error,
        error_kw={'ecolor': error_colors[0], 'capsize': 4, 'elinewidth': 1.5, 'alpha': 0.8})
ax1.set_ylabel("MSE", color=colors[0], fontsize=22)
ax1.set_xlabel("Methods", fontsize=22)
ax1.set_xticks(x)
ax1.set_xticklabels(methods)
ax1.set_ylim(0, 0.25)

# 创建第二个Y轴，绘制R_m^2条形图
ax2 = ax1.twinx()
ax2.bar(x + bar_width / 2, r_m2_data, width=bar_width, color=colors[1], label="r_m^2", yerr=r_m2_error,
        error_kw={'ecolor': error_colors[1], 'capsize': 4, 'elinewidth': 1.5, 'alpha': 0.8})
ax2.set_ylabel("r_m^2", color=colors[1], fontsize=22)
ax2.set_ylim(0.7, 0.8)

# 添加标题
fig.suptitle("Comparison of MSE and r_m^2 of AGCL-DTA model variants", y=0.97, fontsize=30, fontweight='bold')

# 显示全局图例
# 调整图例的位置，使其更贴近标题并保持居中
fig.legend(labels=["MSE", "r_m^2"], loc='upper center', fontsize=20, ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.9))

# 调整布局
plt.tight_layout(rect=[0, 0, 1, 0.86])  # 缩减图例和图案的距离，同时避免重叠

# 保存图表为svg格式
plt.savefig("performance_comparison_with_agcl_dta.svg", format='svg', bbox_inches='tight', dpi=300)

# 显示图表
plt.show()





