# 加载必要的包
library(kohonen)

# 设置随机数种子，确保结果可重复
set.seed(123)

# 设置数据文件路径
data_path <- "M:/R Programs/Groundwater chemical data.csv"

# 读取数据文件
data <- read.csv(data_path, header = TRUE)

# 提取ID列并标准化数据（去除第一列的ID）
sample_ids <- data[,1]
X <- scale(data[,-1])

# 设置SOM模型的网格结构（8x12的六边形拓扑结构）
som_grid <- somgrid(xdim = 8, ydim = 12, topo = "hexagonal")

# 训练SOM模型
som_model <- som(X, grid = som_grid, rlen = 1000, alpha = c(0.05, 0.01))

# 定义自定义的渐变色带（从蓝到红的渐变色带）
custom_palette <- colorRampPalette(c("#0A3D92", "#1652A5", "#2A90D0", "#5EC5D9", "#85C789", 
                                     "#C2DA26", "#FAD509", "#EF7105", "#E61210", "#B0141E"))

# 为 Codes plot 添加矩阵边框
plot(som_model, type = "codes", main = "Codes plot", shape = "straight", border = "black")

# 生成每个特征的平面图（组件平面图，使用翻转后的自定义色带，并使网格线条淡化）
for (i in 1:ncol(X)) {
  plot(som_model, type = "property", property = som_model$codes[[1]][, i], 
       main = colnames(X)[i], palette.name = custom_palette, shape = "straight", border = "white")
}

# 设置随机种子，确保K-means结果一致
set.seed(123)

# 使用k-means对SOM的权重进行聚类
som_codes <- som_model$codes[[1]]
k <- 4  # 设置聚类数量
kmeans_result <- kmeans(som_codes, centers = k)

# 显示K-means聚类结果，使用自定义的渐变色带区分不同的聚类
plot(som_model, type = "mapping", bgcol = custom_palette(k)[kmeans_result$cluster], 
     main = "K-means Clustering on SOM", shape = "straight", border = "white")
add.cluster.boundaries(som_model, kmeans_result$cluster)  # 添加边界

# 输出k-means聚类的结果
print(kmeans_result$cluster)

# 为每个样本找到最佳匹配单元（BMU）
bmus <- map(som_model, X)$unit.classif

# 根据BMU，将每个样本分配到其对应的聚类
sample_clusters <- kmeans_result$cluster[bmus]

# 将样本ID和聚类结果结合起来
data_with_clusters <- data.frame(ID = sample_ids, Cluster = sample_clusters)

# 打印每个样本ID对应的分类结果
print(data_with_clusters)

# 导出带有聚类结果的数据为CSV文件
write.csv(data_with_clusters, file = "sample_clusters_with_ID.csv", row.names = FALSE)

# 可视化：将每个样本的聚类结果投射到SOM网格中
plot(som_model, type = "mapping", bgcol = custom_palette(k)[kmeans_result$cluster], 
     main = "K-means Clustering on SOM", shape = "straight", border = "white")
add.cluster.boundaries(som_model, kmeans_result$cluster)

# ---------------- 手动绘制饼图 ----------------
# 显示每个类别的样本数量
plot(som_model, type="counts", border = "white")  # 淡化网格线条

# 显示SOM聚类结果（右图展示了每个节点的特征值）
plot(som_model, type="codes", border = "white")  # 淡化网格线条

# 查看每个节点的权重值
print(som_model$codes)

# 查看最佳匹配单元（BMU）
print(som_model$unit.classif)

