## model
### backbone 
提取特征
#### 点云 3D 检测模型
1. 基于体素的模型通常需要 Encoder 来对点云体素化，如 HardVFE 和 PointPillarScatter等，采用的稀疏卷积或者 Pillars 的方法从点云中生成 2D 特征图，然后基本可以套用 2D 检测流程进行 3D 检测。
2. 基于原始点云模型通常直接采用 3D Backbone (Pointnet / Pointnet++ 等) 提取点的特征，再针对提取到的点云特征采用 RoI 或者 Group 等方式回归 3D bounding box。
#### 单目 3D 检测模型

#### 多模态 3D 检测模型

#### 点云 3D 语义分割模型

### neck
特征融合和增强


### head
输出所需结果


