### open3D 在线可视化
基于 Open3D 构建了一个在线 Visualizer，用于在有 GUI 界面的情况下提供实时可视化结果，Open3D 提供了非常丰富的功能。相关代码位于 mmdet3d/core/visualizer/open3d_vis.py 

### 使用 MeshLab 可视化
对于 MeshLab 来说，可视化需要提供相应的 obj 文件，文件内包含点云信息、分割结果、检测结果等等。而在目前 MMDetection3D 中，我们提供下述方法，可以将模型输出结果转换为 obj 文件。mmdet3d/core/visualizer/show_result.py

点云场景 3D 框可视化 show_result

### Demo 可视化
在 demo 可视化的时候，通过使用训练好的模型得到推理结果后，直接根据需要使用show_result.py中的三个可视化函数可视化三件套。这种情况下通常输入是某个点云场景 bin 文件或者一张图片。

### 推理过程可视化
对于某个模型，在某个数据集的验证集/测试集推理的时候可视化推理结果，调用的是该模型内部实现的 model.show_results 方法，在 MMDetection3D 中，我们为三种模型的基类分别实现了相应的可视化方法 show_results。
点云 3D 检测（多模态 3D 检测）模型推理的时候，其可视化方法 show_results 调用的是三件套中的 show_result，点云分割模型调用的则是 show_seg_result，单目 3D 检测模型调用的是 show_multi_modality_result。
同时这些检测模型的 model.show_results 方法基本都提供了 score_thr 参数，用户可以更改该参数调整推理结果可视化时的检测框的阈值，获得更好的可视化效果。

### 结果可视化
MMDetection3D 提供了 tools/misc/visualize_results.py 脚本，用于可视化检测结果。通常来说，模型完成测试集/验证集的推理后，通常会生成保存检测结果的 pkl 格式的文件，该 pkl 文件的内部的具体存储格式则因数据集而异，所以通常对于每个数据集类，也会实现对应的 dataset.show 方法

pkl 文件只会保存检测到的 3D 框的信息，所以我们需要借助于数据集本身通过 pipeline 的方式获取点云（或者图片信息）


### 数据集及标签可视化
MMDetection3D 提供 tools/misc/browse_dataset.py 脚本，browse_dataset 可以对 datasets 吐出的数据进行可视化检查
在可视化数据集的时候需要在命令行指定 task，包括 det、multi_modality-det、mono-det、seg 四种
