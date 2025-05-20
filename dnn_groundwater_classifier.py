# See uploaded source for full content
# This file contains the full DNN-based pollution classification model.
# Includes training, visualization, evaluation, and export.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# 数据加载
data = pd.read_csv('M:/Programs/PythonProject/pythonProject/Groundwater pollution data DNN.csv')

# 定义污染物名称列表
pollutants = ['Pb', 'Cd', 'Hg', 'Se', 'Cu', 'As', 'Tl', 'Mo', 'Be', 'Ba', 'Ni', 'Sb', 'Ag',
              'Cr', 'Al', 'V', 'Co', 'B', 'Mn', 'Zn']

# 根据中国地下水污染标准划分污染等级
def classify_pollution(row):
    levels = []
    # 逐一对每个污染物进行标准比对（略去具体代码，与之前一致）
    # Pb
    if row['Pb'] <= 0.005:
        levels.append(0)
    elif 0.005 < row['Pb'] <= 0.01:
        levels.append(1)
    elif 0.01 < row['Pb'] <= 0.1:
        levels.append(2)
    elif 0.1 < row['Pb'] <= 0.5:
        levels.append(3)
    else:
        levels.append(4)

    # Cd
    if row['Cd'] <= 0.001:
        levels.append(0)
    elif 0.001 < row['Cd'] <= 0.005:
        levels.append(1)
    elif 0.005 < row['Cd'] <= 0.01:
        levels.append(2)
    elif 0.01 < row['Cd'] <= 0.05:
        levels.append(3)
    else:
        levels.append(4)

    # Hg
    if row['Hg'] <= 0.00005:
        levels.append(0)
    elif 0.00005 < row['Hg'] <= 0.0001:
        levels.append(1)
    elif 0.0001 < row['Hg'] <= 0.001:
        levels.append(2)
    elif 0.001 < row['Hg'] <= 0.002:
        levels.append(3)
    else:
        levels.append(4)

    # Se
    if row['Se'] <= 0.005:
        levels.append(0)
    elif 0.005 < row['Se'] <= 0.008:
        levels.append(1)
    elif 0.008 < row['Se'] <= 0.01:
        levels.append(2)
    elif 0.01 < row['Se'] <= 0.1:
        levels.append(3)
    else:
        levels.append(4)

    # Cu
    if row['Cu'] <= 0.01:
        levels.append(0)
    elif 0.01 < row['Cu'] <= 0.05:
        levels.append(1)
    elif 0.05 < row['Cu'] <= 1:
        levels.append(2)
    elif 1 < row['Cu'] <= 1.5:
        levels.append(3)
    else:
        levels.append(4)

    # As
    if row['As'] <= 0.0005:
        levels.append(0)
    elif 0.0005 < row['As'] <= 0.001:
        levels.append(1)
    elif 0.001 < row['As'] <= 0.01:
        levels.append(2)
    elif 0.01 < row['As'] <= 0.1:
        levels.append(3)
    else:
        levels.append(4)

    # Tl
    if row['Tl'] <= 0.00005:
        levels.append(0)
    elif 0.00005 < row['Tl'] <= 0.00008:
        levels.append(1)
    elif 0.00008 < row['Tl'] <= 0.0001:
        levels.append(2)
    elif 0.0001 < row['Tl'] <= 0.001:
        levels.append(3)
    else:
        levels.append(4)

    # Mo
    if row['Mo'] <= 0.001:
        levels.append(0)
    elif 0.001 < row['Mo'] <= 0.01:
        levels.append(1)
    elif 0.01 < row['Mo'] <= 0.07:
        levels.append(2)
    elif 0.07 < row['Mo'] <= 0.15:
        levels.append(3)
    else:
        levels.append(4)

    # Be
    if row['Be'] <= 0.00005:
        levels.append(0)
    elif 0.00005 < row['Be'] <= 0.0001:
        levels.append(1)
    elif 0.0001 < row['Be'] <= 0.002:
        levels.append(2)
    elif 0.002 < row['Be'] <= 0.06:
        levels.append(3)
    else:
        levels.append(4)

    # Ba
    if row['Ba'] <= 0.01:
        levels.append(0)
    elif 0.01 < row['Ba'] <= 0.10:
        levels.append(1)
    elif 0.10 < row['Ba'] <= 0.70:
        levels.append(2)
    elif 0.70 < row['Ba'] <= 4:
        levels.append(3)
    else:
        levels.append(4)

    # Ni
    if row['Ni'] <= 0.001:
        levels.append(0)
    elif 0.001 < row['Ni'] <= 0.002:
        levels.append(1)
    elif 0.002 < row['Ni'] <= 0.02:
        levels.append(2)
    elif 0.02 < row['Ni'] <= 0.1:
        levels.append(3)
    else:
        levels.append(4)

    # Sb
    if row['Sb'] <= 0.0001:
        levels.append(0)
    elif 0.0001 < row['Sb'] <= 0.0005:
        levels.append(1)
    elif 0.0005 < row['Sb'] <= 0.005:
        levels.append(2)
    elif 0.005 < row['Sb'] <= 0.01:
        levels.append(3)
    else:
        levels.append(4)

    # Ag
    if row['Ag'] <= 0.001:
        levels.append(0)
    elif 0.001 < row['Ag'] <= 0.01:
        levels.append(1)
    elif 0.01 < row['Ag'] <= 0.05:
        levels.append(2)
    elif 0.05 < row['Ag'] <= 0.10:
        levels.append(3)
    else:
        levels.append(4)

    # Cr
    if row['Cr'] <= 0.005:
        levels.append(0)
    elif 0.005 < row['Cr'] <= 0.01:
        levels.append(1)
    elif 0.01 < row['Cr'] <= 0.05:
        levels.append(2)
    elif 0.05 < row['Cr'] <= 0.10:
        levels.append(3)
    else:
        levels.append(4)

    # Al
    if row['Al'] <= 0.01:
        levels.append(0)
    elif 0.01 < row['Al'] <= 0.05:
        levels.append(1)
    elif 0.05 < row['Al'] <= 0.20:
        levels.append(2)
    elif 0.20 < row['Al'] <= 0.50:
        levels.append(3)
    else:
        levels.append(4)

    # V
    if row['V'] <= 0.005:
        levels.append(0)
    elif 0.005 < row['V'] <= 0.01:
        levels.append(1)
    elif 0.01 < row['V'] <= 0.05:
        levels.append(2)
    elif 0.05 < row['V'] <= 0.50:
        levels.append(3)
    else:
        levels.append(4)

    # Co
    if row['Co'] <= 0.0025:
        levels.append(0)
    elif 0.0025 < row['Co'] <= 0.005:
        levels.append(1)
    elif 0.005 < row['Co'] <= 0.05:
        levels.append(2)
    elif 0.05 < row['Co'] <= 0.10:
        levels.append(3)
    else:
        levels.append(4)

    # B
    if row['B'] <= 0.02:
        levels.append(0)
    elif 0.02 < row['B'] <= 0.10:
        levels.append(1)
    elif 0.10 < row['B'] <= 0.50:
        levels.append(2)
    elif 0.50 < row['B'] <= 2.00:
        levels.append(3)
    else:
        levels.append(4)

    # Mn
    if row['Mn'] <= 0.025:
        levels.append(0)
    elif 0.025 < row['Mn'] <= 0.05:
        levels.append(1)
    elif 0.05 < row['Mn'] <= 0.10:
        levels.append(2)
    elif 0.10 < row['Mn'] <= 1.50:
        levels.append(3)
    else:
        levels.append(4)

    # Zn
    if row['Zn'] <= 0.05:
        levels.append(0)
    elif 0.05 < row['Zn'] <= 0.5:
        levels.append(1)
    elif 0.5 < row['Zn'] <= 1:
        levels.append(2)
    elif 1 < row['Zn'] <= 5:
        levels.append(3)
    else:
        levels.append(4)

    return np.mean(levels)

# 应用函数并创建标签列
data['Pollution_Level'] = data.apply(classify_pollution, axis=1)

# 清洗污染物数据
data_cleaned_pollutants = data[pollutants].replace([np.inf, -np.inf], np.nan).dropna()

# 预测部分
X_pollutants = data_cleaned_pollutants  # 特征数据
y = data['Pollution_Level']  # 标签数据

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(X_pollutants, y, test_size=0.2, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 构建 Gradient Boosting Trees 模型
gbt_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gbt_model.fit(X_train, y_train)

# 使用模型预测污染等级
data['Predicted_Pollution_Level'] = np.round(gbt_model.predict(scaler.transform(X_pollutants))).astype(int)

# 绘制基于模型预测的分布图
plt.figure(figsize=(15, 10))

# 使用接近图片中的颜色
box_color = '#FF0044'  # 红色用于箱线图
point_color = '#1E90FF'  # 蓝色用于实心点

for i, pollutant in enumerate(pollutants, 1):
    plt.subplot(5, 4, i)

    # 绘制箱线图，基于模型预测等级
    sns.boxplot(x=data['Predicted_Pollution_Level'], y=data_cleaned_pollutants[pollutant], showfliers=False,
                color=box_color, linewidth=0.5)

    # 绘制实心蓝色点
    sns.stripplot(x=data['Predicted_Pollution_Level'], y=data_cleaned_pollutants[pollutant],
                  jitter=True, color=point_color, alpha=0.8, size=5, marker='o')

    # 删除横坐标和纵坐标刻度标注及标题
    plt.xticks([])  # 删除横坐标刻度
    plt.xlabel('')  # 删除横坐标标题
    plt.ylabel('')  # 删除纵坐标标题

    # 删除网格线
    plt.grid(False)

# 调整布局
plt.tight_layout()

# 显示图像
plt.show()

# 计算均方误差
mse = mean_squared_error(y_test, gbt_model.predict(X_test))
print(f'Mean Squared Error: {mse}')

# 保存分类结果
result = data[['ID', 'Pollution_Level', 'Predicted_Pollution_Level']]
result.to_csv('final_pollution_classification_with_predictions.csv', index=False)
print("分类结果已保存到 'final_pollution_classification_with_predictions.csv'")
