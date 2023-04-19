### 常用链接
https://mmdetection3d.readthedocs.io/zh_CN/latest/index.html


### 数据集
常用数据集有：lidar-pools,public,kitti-mmdet3d，lidar-open-dataset, nuscenes-mmdet3d,waymo_kitti，lidar-robosense（kitti对应config文件为 ...../kitti/config_mdet3d/）

/share/public/public
/share/public/public/kitti
/share/public/lidar-open-dataset
/share/public/lidar-open-dataset/kitti_openpcdet  #


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
