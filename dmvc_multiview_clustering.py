# See uploaded script for full DMVC implementation
# This file builds an encoder to extract latent features from multi-view data,
# then applies KMeans clustering and saves cluster assignments and summaries.
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 设置随机种子
np.random.seed(42)
tf.random.set_seed(42)

# 1. 加载数据
data = pd.read_csv("Groundwater class.csv")  # 替换为您的CSV文件路径

# 提取特征矩阵，并确保数据为浮点数
features = data[["vulnerability", "mobility", "hydrochemistry", "land use", "pollution"]].astype(float).values

# 2. 定义简单的深度学习编码器
def build_encoder(input_dim, latent_dim):
    input_layer = tf.keras.layers.Input(shape=(input_dim,))
    encoded = tf.keras.layers.Dense(64, activation='relu')(input_layer)
    encoded = tf.keras.layers.Dense(32, activation='relu')(encoded)
    encoded = tf.keras.layers.Dense(latent_dim, activation='relu')(encoded)
    return tf.keras.Model(inputs=input_layer, outputs=encoded)

# 设置潜在空间维度
latent_dim = 5  # 潜在空间的维度（可以根据需求调整）

# 构建编码器
encoder = build_encoder(features.shape[1], latent_dim)

# 编译模型（仅用于提取潜在特征，不需要训练）
encoder.compile(optimizer='adam', loss='mse')

# 3. 编码数据（提取潜在表示）
encoded_features = encoder.predict(features)

# 检查潜在表示的形状
print("潜在表示的形状：", encoded_features.shape)

# 4. 使用 KMeans 聚类
n_clusters = 5  # 聚类分区数量（可以根据需求调整）
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(encoded_features)

# 将聚类结果保存到原数据中
data["cluster"] = clusters

# 5. 分析每个类别的主要决定因素
cluster_summary = data.groupby("cluster")[["vulnerability", "mobility", "hydrochemistry", "land use", "pollution"]].mean()
print("\n每个类别的特征均值：")
print(cluster_summary)

# 保存类别分析结果
cluster_summary.to_csv("cluster_summary.csv", index=True)

# 保存聚类结果到 CSV 文件
output_file = "pollution_repair_clusters.csv"
data.to_csv(output_file, index=False)
print(f"\n聚类结果已保存为 {output_file}")
print(f"\n类别特征分析结果已保存为 cluster_summary.csv")

# 6. 可视化聚类结果（二维降维可视化）
from sklearn.decomposition import PCA

# 使用 PCA 将潜在空间降维到二维
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(encoded_features)

# 绘制聚类结果
plt.figure(figsize=(10, 6))
for cluster_id in range(n_clusters):
    plt.scatter(reduced_data[clusters == cluster_id, 0],
                reduced_data[clusters == cluster_id, 1],
                label=f"Cluster {cluster_id}")
plt.title("Pollution Repair Clustering Results (PCA Visualization)")
plt.xlabel("PCA Dimension 1")
plt.ylabel("PCA Dimension 2")
plt.legend()
plt.grid()
plt.show()
