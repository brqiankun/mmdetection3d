### 常用链接
https://mmdetection3d.readthedocs.io/zh_CN/latest/index.html

``` python
conda create -n mmdet3d_1_0 python=3.8
conda activate mmdet3d_1_0
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
git checkout mmdet3d/learn
pip install openmim
mim install mmcv-full
mim install mmdet
mim install mmsegmentation
# 安装mmdet3d
pip install -v -e . 

```

### 模型训练
```
# 单机单卡  nuscence可以
python tools/train.py configs/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d.py 
# 单机多卡
./tools/dist_train.sh configs/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d.py 8

```

### 数据集
常用数据集有：lidar-pools,public,kitti-mmdet3d，lidar-open-dataset, nuscenes-mmdet3d,waymo_kitti，lidar-robosense（kitti对应config文件为 ...../kitti/config_mdet3d/）

/share/public/public
/share/public/public/kitti
/share/public/lidar-open-dataset
/share/public/lidar-open-dataset/kitti_openpcdet  #

#### kitti数据集格式(自定义数据集)
1. ImageSets包含数据集划分文件，用来划分训练、验证、测试集
2. calib包含每数据样本的标定信息
3. image_2和velodyne包含图像数据和点云数据
4. label_2包含于3D目标检测相关的标注文件

tools/waymo_converter.py 将waymo转换为kitti格式
mmdet3d/datasets/waymo_dataset.py

准备配置文件来帮助数据集的读取和使用
需要三个配置文件: 数据集配置文件(configs/\_base\_/datasets/kitti-3d-3class.py) + 模型配置文件(configs/\_base\_/models/hv_pointpillars_secfpn_kitti.py) ==> 整体配置文件(configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py)

自定义数据集, 重写新的数据集类后(mmdet3d/datasets/my_dataset.py), 修改数据集配置文件来调用my_dataset数据集类

这是一个链接 [自定义数据集](https://mmdetection3d.readthedocs.io/zh_CN/latest/tutorials/customize_dataset.html)

数据集包装器:  源码见(mmdet3d/datasets/dataset_wrappers.py)

1. RepeatDataset：简单地重复整个数据集

2. ClassBalancedDataset：以类别平衡的方式重复数据集 类别均衡(class balance): 避免某些类别被过拟合或忽略  进行重复的数据集需要实例化函数 self.get_cat_ids(idx)

3. ConcatDataset：拼接多个数据集

修改数据集类别 
可以对现有数据集类别名进行修改, 实现全部标注的子集标注的训练。


#### 自定义数据预处理流程
[数据预处理流程](https://mmdetection3d.readthedocs.io/zh_CN/latest/tutorials/data_pipeline.html) <br>
使用Dataset和DataLoader来调用多个进程进行数据加载

Dataset 将会返回与模型前向传播的参数所对应的数据项构成的字典。在 MMCV 中引入一个 [DataContainer](https://github.com/open-mmlab/mmcv/blob/master/mmcv/parallel/data_container.py)类型，来帮助收集和分发不同尺寸的数据

数据预处理流程和数据集之间是互相分离的两个部分。
数据集定义了如何处理标注信息，而数据预处理流程定义了准备数据项字典的所有步骤
数据集预处理流程包含一系列的操作，每个操作将一个字典作为输入，并输出应用于下一个转换的一个新的字典.<br>
见(mmdetection3d/configs/_base_/datasets/kitti-3d-3class.py)  train_pipeline

pipelines(比如LoadPointsFromFile)实现见mmdetection3d/mmdet3d/datasets/pipelines目录下

1. 数据加载
LoadPointsFromFile

添加：points

LoadPointsFromMultiSweeps

更新：points

LoadAnnotations3D

添加：gt_bboxes_3d, gt_labels_3d, gt_bboxes, gt_labels, pts_instance_mask, pts_semantic_mask, bbox3d_fields, pts_mask_fields, pts_seg_fields

2. 预处理
GlobalRotScaleTrans

添加：pcd_trans, pcd_rotation, pcd_scale_factor

更新：points, *bbox3d_fields

RandomFlip3D

添加：flip, pcd_horizontal_flip, pcd_vertical_flip

更新：points, *bbox3d_fields

PointsRangeFilter

更新：points

ObjectRangeFilter

更新：gt_bboxes_3d, gt_labels_3d

ObjectNameFilter

更新：gt_bboxes_3d, gt_labels_3d

PointShuffle

更新：points

PointsRangeFilter

更新：points

3. 格式化
DefaultFormatBundle3D

更新：points, gt_bboxes_3d, gt_labels_3d, gt_bboxes, gt_labels

Collect3D

添加：img_meta （由 meta_keys 指定的键值构成的 img_meta）

移除：所有除 keys 指定的键值以外的其他键值

4. 测试时的数据增强
MultiScaleFlipAug

更新: scale, pcd_scale_factor, flip, flip_direction, pcd_horizontal_flip, pcd_vertical_flip （与这些指定的参数对应的增强后的数据列表）

扩展并使用自定义数据集预处理方法

在pipeline.py目录下创建my_pipeline.py 
输入输出都是字典
```
from mmdet.datasets import PIPELINES

@PIPELINES.register_module()
class MyTransform:

    def __call__(self, results):
        results['dummy'] = True
        return results
```
在mmdetection3d/mmdet3d/datasets/pipelines/\_\_init\_\_.py 中 <br>
```
from .my_pipeline import MyTransform
```
在配置文件中添加
dict(type='MyTransform'),


### 自定义模型
模型包括:
1. 编码器(encoder)  包括voxel layer， voxel encoder和middle encoder等进入backbone前使用的基于voxel的方法
2. 骨干网络(backbone)  使用FCN来提取特征图 如ResNet
3. 颈部网络(neck)  FPN和SECONDFPN
4. 检测头(head)   特定任务的组成, 如检测框预测和掩码预测
5. RoI提取器(RoI extractor) Region of Interest  用于目标检测和实例分割等任务，用于从图像或特征图中识别和提取特定区域
6. 损失函数(loss)

#### 以HardVFE为例搭建encoder
HardVFE（Hard Voxel Feature Encoding）是一种三维点云处理方法，主要用于自动驾驶和机器人领域的对象检测和分割任务。它将输入的点云数据分解成固定大小的体素（voxels），然后对每个体素内的点进行特征提取和编码。

HardVFE可以有效地降低点云数据的复杂性，同时保留空间和结构信息。它与二维图像处理中的卷积神经网络类似，具有较强的特征学习能力。

#### 以SECOND为例搭建backbone
SECOND (Sparsely Embedded Convolutional Detection) 是一种基于三维点云数据的物体检测算法，主要用于自动驾驶领域。它使用稀疏卷积神经网络（Sparse Convolutional Neural Network, SCNN）对输入点云进行有效处理，以加快计算速度并降低内存消耗。

SECOND 的核心步骤如下：

1. 将输入点云数据分解为固定大小的体素（Voxelization）。
2. 使用 Voxel Feature Encoding（VFE）对每个体素的点进行特征提取和编码。
3. 通过稀疏卷积层处理编码后的体素特征来学习高级语义信息。
4. 应用区域提议网络（RPN, Region Proposal Network）生成候选边界框。
5. 对 RPN 提出的候选区域进行非极大值抑制（NMS, Non-Maximum Suppression），筛选出最终结果。<br>
SECOND 在保持高精度的同时，能够显著提高物体检测任务的计算效率，适用于实时场景中的自动驾驶应用。

#### 添加新建 necks (SECONDFPN)

#### 添加新建 heads 
1. 单阶段检测器(mmdetection3d/mmdet3d/models/dense_heads)  
单阶段检测器（One-stage Detector）是一类用于目标检测的深度学习算法。与两阶段检测器（如Faster R-CNN）相比，单阶段检测器直接在原始图像上生成目标边界框和类别概率，无需经过区域建议网络（RPN）生成候选框的步骤。因此，它们通常具有更高的计算效率和实时性，但在某些情况下可能牺牲一定的准确性。YOLO
2. 双阶段检测器(mmdetection3d/mmdet3d/models/roi_heads)
在mmdetection3d/mmdet3d/models/roi_heads下新建自定义bbox head

用户需要在 mmdet3d/models/roi_heads/bbox_heads/__init__.py 和 mmdet3d/models/roi_heads/__init__.py 中添加新模块，使得对应的注册器能够发现并加载该模块。
或者在配置文件中添加
```
custom_imports=dict(
    imports=['mmdet3d.models.roi_heads.part_aggregation_roi_head', 'mmdet3d.models.roi_heads.bbox_heads.parta2_bbox_head'])
```


#### 新建loss
在mmdetection3d/mmdet3d/models/losses中新建loss, 按照对应模板
之后将新建loss添加到mmdetection3d/mmdet3d/models/losses/\_\_init\_\_.py
也可在配置文件中添加custom_imports=dict()
需要在对应的head中修改loss_bbox的值


#### 自定义运行时配置 自定义优化器
支持所有Pytorch的优化器，改变配置文件中的 optimizer 字段


#### 模型复杂度分析
```
python tools/analysis_tools/get_flops.py configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py
```


### pointnet++
10.9.1.8上可以运行mmdet3d

```python
python demo/pc_seg_demo.py demo/data/scannet/scene0000_00.bin configs/pointnet2/pointnet2_ssg_16x2_cosine_200e_scannet_seg-3d-20class.py checkpoints/pointnet2_ssg_16x2_cosine_200e_scannet_seg-3d-20class_20210514_143644-ee73704a.pth --out-dir brtest_output/

python demo/pcd_demo.py demo/data/kitti/kitti_000008.bin configs/second/hv_second_secfpn_6x8_80e_kitti-3d-car.py checkpoints/hv_second_secfpn_6x8_80e_kitti-3d-car_20200620_230238-393f000c.pth --out-dir ./brtest_output/

python demo/multi_modality_demo.py demo/data/kitti/kitti_000008.bin demo/data/kitti/kitti_000008.png demo/data/kitti/kitti_000008_infos.pkl configs/mvxnet/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class.py checkpoints/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class_20210831_060805-83442923.pth --out-dir ./brtest_output/
```

```
nsys profile python demo/pc_seg_demo.py demo/data/scannet/scene0000_00.bin configs/pointnet2/pointnet2_ssg_16x2_cosine_200e_scannet_seg-3d-20class.py checkpoints/pointnet2_ssg_16x2_cosine_200e_scannet_seg-3d-20class_20210514_143644-ee73704a.pth --out-dir brtest_output/
```

### pointpillars
预训练模型的demo验证
```python
python demo/pcd_demo.py demo/data/kitti/kitti_000008.bin configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car.py checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth --out-dir brtest_output/
```

模型训练  可以训练
1. 单GPU
```
python tools/train.py configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py 
```

2. 多GPU
```
./tools/dist_train.sh configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car.py 4
```
输出日志保存在work_dir下

mmdet中最复杂的是head模块  
mmdet 训练部分最核心的是runner部分  
mmdet 训练部分最核心的是runner部分数据流  

#### pipeline
按照插入顺序运行的数据处理模块，包括数据前处理，数据增强，数据收集等。接受数据字典，输出经过处理后的字典。代码位于mmdet3d/dataset/pipeline目录下


#### DataParallel 和 Model
mmdet中的dataloader包括了DataContainer对象，来提高内存效率。使用mmdet自带的MMDataParallel 和 MMDistributedDataParallel 来处理DataContainer对象

#### Runner 和 Hooks
Runner中包括优化器，学习率设置，权重保存等组件。 Runner封装了各个框架的训练和验证流程，负责管理训练和验证过程的整个生命周期， 通过预定义回调函数，插入定制化Hook，实现定制化需求


#### 训练和测试流程
训练和验证调用 tools/train.py脚本，进行Dataset、Model相关类初始化。之后构建runner，模型训练和验证在runner内部，runner实际调用了model内部的train_step和val_step

#### 以pointpillars为例分析mmdet3d中model的训练和测试
![train_and_test](./docs/picture/v2-a8c9de0156a19b7ddc84ab550ea3419a_r.jpg "train_and_test")


### 坐标系
框坐标表示: (x, y, z, x_size, y_size, z_size, yaw)  x, y, z 表示框的位置, x_size, y_size, z_size表示框的尺寸, yaw表示框的朝向
坐标系中，x轴位参考方向，包围框的旋转角度只考虑朝向角yaw，不考虑俯仰角pitch和翻滚角roll

### 构建Box
将某个场景所有的物体 3D 框 Boxes 封装成一个类，提供 BaseInstance3DBoxes这个基类，再分别基于此为三种坐标系构建 LiDARInstance3DBoxes、CameraInstance3DBoxes、DepthInstance3DBoxes 三种 Boxes 类，相关的代码位于/mmdet3d/core/bbox/structures目录下。


origin 的理解问题，我们可以在各个不同的坐标系下将任意一个 3D 框通过放缩和平移变换为一个坐标值在 0-1 之间的正方体，而此时各个坐标系下 3D 框底部中心点的坐标值就是原 3D 框底部中心点对应的 origin：
