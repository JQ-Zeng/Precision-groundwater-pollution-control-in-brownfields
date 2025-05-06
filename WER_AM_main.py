import numpy as np
import pandas as pd
import random
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

# 设置随机种子以确保结果一致
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

# 加载数据
file_path = 'Autoencoder_Groundwater pollution data ALL.csv'
data = pd.read_csv(file_path)

# 提取超标率并转换为数值型
exceedance_rates = data.iloc[0, 4:].astype(float).values
pollutant_names = data.columns[4:]  # 获取污染物名称

# 找出超标率大于 0 的污染物
valid_indices = np.where(exceedance_rates > 0)[0]
exceedance_rates_filtered = exceedance_rates[valid_indices]
pollutant_names_filtered = pollutant_names[valid_indices]

# 提取监测井浓度数据并过滤超标率为 0 的污染物
data_concentrations = data.iloc[1:, 4:].astype(float).values  # 从第五列开始是污染物浓度数据
data_concentrations_filtered = data_concentrations[:, valid_indices]  # 仅保留超标率大于 0 的污染物

# 超标率加权
alpha = 1.0  # 加权系数
weights = exceedance_rates_filtered * alpha  # 直接将超标率作为权重
weighted_data = data_concentrations_filtered * weights  # 将数据按超标率加权

# 数据标准化
scaler = StandardScaler()
weighted_data = scaler.fit_transform(weighted_data)

# 自动编码器模型
input_dim = weighted_data.shape[1]
encoding_dim = 10  # 编码层维度，降维至10个主要特征

input_layer = Input(shape=(input_dim,))
encoded = Dense(20, activation="relu")(input_layer)
encoded = Dense(encoding_dim, activation="relu")(encoded)
decoded = Dense(20, activation="relu")(encoded)
decoded = Dense(input_dim, activation="sigmoid")(decoded)  # 重构层

# 构建自动编码器模型
autoencoder = Model(inputs=input_layer, outputs=decoded)
encoder = Model(inputs=input_layer, outputs=encoded)  # 提取编码层模型
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

# 训练自动编码器
autoencoder.fit(weighted_data, weighted_data, epochs=100, batch_size=32, shuffle=True, validation_split=0.1, verbose=0)

# 提取编码层特征
encoded_data = encoder.predict(weighted_data)

# 计算重构误差
reconstructed_data = autoencoder.predict(weighted_data)
reconstruction_error = np.mean(np.square(weighted_data - reconstructed_data), axis=0)

# 构建超标率和重构误差的DataFrame
feature_importance = pd.DataFrame({
    "Pollutant": pollutant_names_filtered,
    "ExceedanceRate": exceedance_rates_filtered,
    "ReconstructionError": reconstruction_error
})

# 计算综合得分并进行排序
beta = 0.7  # 设置权重因子，调整 `ReconstructionError` 和 `ExceedanceRate` 的影响比例
feature_importance['Score'] = beta * feature_importance['ReconstructionError'] + (1 - beta) * feature_importance['ExceedanceRate']

# 按综合得分降序排序
feature_importance_sorted = feature_importance.sort_values(by="Score", ascending=False)

# 保存到 CSV 文件
output_file_path = 'feature_importance_combined_sorted.csv'
feature_importance_sorted.to_csv(output_file_path, index=False)

print(f"包含综合得分的污染物列表已保存至 {output_file_path}")
