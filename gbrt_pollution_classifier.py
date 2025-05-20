# See uploaded source for full GBRT groundwater pollution classification model
# Includes rule-based label creation, training, evaluation, SHAP, ROC curve, and learning curve.
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# 1. 加载数据
# 假设数据存储在 CSV 文件中
data = pd.read_csv("pollution_data.csv")

# 2. 定义特征和目标
# 特征为 organic 和 inorganic 污染等级
X = data[["inorganic", "organic"]]

# 定义污染分级规则
def generate_pollution_level(row):
    # 重度污染
    if row["inorganic"] == 4 or row["organic"] == 4:
        return 4  # 重度污染
    # 无污染
    elif row["inorganic"] == 1 and row["organic"] == 1:
        return 1  # 无污染
    # 轻度污染
    elif row["inorganic"] <= 2 and row["organic"] <= 2:
        return 2  # 轻度污染
    # 中度污染
    elif row["inorganic"] == 3 or row["organic"] == 3:
        return 3  # 中度污染

# 应用规则生成污染等级
data["pollution_level"] = data.apply(generate_pollution_level, axis=1)

# 检查目标变量的分布
print("目标变量分布：")
print(data["pollution_level"].value_counts())

# 3. 分层抽样划分训练集和测试集
y = data["pollution_level"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 检查训练集和测试集类别分布
print("训练集类别分布：")
print(y_train.value_counts())
print("测试集类别分布：")
print(y_test.value_counts())

# 4. 初始化梯度提升树模型
gbm = GradientBoostingClassifier(
    n_estimators=150,  # 增加树的数量
    learning_rate=0.05,  # 减小学习率
    max_depth=3,
    random_state=42
)

# 5. 训练模型
gbm.fit(X_train, y_train)

# 6. 在测试集上进行预测
y_pred = gbm.predict(X_test)

# 输出分类报告
print("分类报告:\n", classification_report(y_test, y_pred))
print("模型精度: ", accuracy_score(y_test, y_pred))

# 7. 对整个数据集进行预测
data["predicted_pollution_level"] = gbm.predict(X)

# 检查预测结果分布
print("预测结果分布：")
print(data["predicted_pollution_level"].value_counts())

# 8. 将预测结果映射为文本分类
level_mapping = {
    1: "无污染",
    2: "轻度污染",
    3: "中度污染",
    4: "重度污染"
}
data["predicted_pollution_category"] = data["predicted_pollution_level"].map(level_mapping)

# 9. 保存结果为 CSV 文件
output_file = "pollution_level_results.csv"
data.to_csv(output_file, index=False)
print(f"预测结果已保存为 {output_file}")
