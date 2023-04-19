### 构建Box
将某个场景所有的物体 3D 框 Boxes 封装成一个类，提供 BaseInstance3DBoxes这个基类，再分别基于此为三种坐标系构建 LiDARInstance3DBoxes、CameraInstance3DBoxes、DepthInstance3DBoxes 三种 Boxes 类，相关的代码位于/mmdet3d/core/bbox/structures目录下。