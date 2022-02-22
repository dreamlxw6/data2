## 一、项目背景介绍
在施工现场，对于来往人员，以及工作人员而言，安全问题至关重要。而安全帽更是保障施工现场在场人员安全的第一防线，因此需要对场地中的人员进行安全提醒。当人员未佩戴安全帽进入施工场所时，人为监管耗时耗力，而且不易实时监管，过程繁琐、消耗人力且实时性较差。针对上述问题，希望通过视频监控->目标检测->智能督导的方式智能、高效的完成此任务:  

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/63b901ca44ca482abb31511b8b99faed4cdbad7d9d7c467e8cdd169181895bb4" width = "500"></center>
<center><br>图1：安全施工图 </br></center>
<br></br>

# 二、数据处理
**2.1PaddleX简介：**  
PaddleX是飞桨全流程开发工具，集飞桨核心框架、模型库、工具及组件等深度学习开发所需全部能力于一身，打通深度学习开发全流程，并**提供简明易懂的Python API**，方便用户根据实际生产需求进行直接调用或二次开发，为开发者提供飞桨全流程开发的最佳实践。目前，该工具代码已开源于GitHub，同时可访问PaddleX在线使用文档，快速查阅读使用教程和API文档说明。  
[PaddleX代码GitHub链接](https://github.com/PaddlePaddle/PaddleX/tree/develop)  
[PaddleX文档链接](https://paddlex.readthedocs.io/zh_CN/develop/index.html)  
**2.2安装PaddleX**  
**2.4划分数据集**  
**2.3挂载数据集**
需要在data文件夹下 对解压出来的 annotatios文件夹重命名成---Annotations
images文件夹重命名成JPEGImages文件夹
# 三、模型选择和调参
**3.1 YOLOv3模型设计思想**

YOLOv3算法的基本思想可以分成两部分：

* 按一定规则在图片上产生一系列的候选区域，然后根据这些候选区域与图片上物体真实框之间的位置关系对候选区域进行标注。跟真实框足够接近的那些候选区域会被标注为正样本，同时将真实框的位置作为正样本的位置目标。偏离真实框较大的那些候选区域则会被标注为负样本，负样本不需要预测位置或者类别。
* 使用卷积神经网络提取图片特征并对候选区域的位置和类别进行预测。这样每个预测框就可以看成是一个样本，根据真实框相对它的位置和类别进行了标注而获得标签值，通过网络模型预测其位置和类别，将网络预测值和标签值进行比较，就可以建立起损失函数。

YOLOv3算法训练过程的流程图如 **图2** 所示：

<br></br>
<center><img src="https://ai-studio-static-online.cdn.bcebos.com/f2eb2b75bb5a4e518b86a257e0f931de7377dba3bba44d1e846b307036aed41a" width = "800"></center>
<center><br>图2：YOLOv3算法训练流程图 </br></center>
<br></br>


* **图2** 左边是输入图片，上半部分所示的过程是使用卷积神经网络对图片提取特征，随着网络不断向前传播，特征图的尺寸越来越小，每个像素点会代表更加抽象的特征模式，直到输出特征图，其尺寸减小为原图的$\frac{1}{32}$。
* **图2** 下半部分描述了生成候选区域的过程，首先将原图划分成多个小方块，每个小方块的大小是$32 \times 32$，然后以每个小方块为中心分别生成一系列锚框，整张图片都会被锚框覆盖到。在每个锚框的基础上产生一个与之对应的预测框，根据锚框和预测框与图片上物体真实框之间的位置关系，对这些预测框进行标注。
* 将上方支路中输出的特征图与下方支路中产生的预测框标签建立关联，创建损失函数，开启端到端的训练过程。

这里我们直接用PaddleX调用YOLOv3模型  

**3.2配置GPU**  
**3.3 定义图像处理流程transforms**       
定义数据处理流程，其中训练集和验证集需分别定义，训练过程包括了部分测试过程中不需要的数据增强操作，如在本示例中，训练过程使用了MixupImage、RandomDistort、RandomExpand、RandomCrop和RandomHorizontalFlip共5种数据增强方式，更多图像预处理流程transforms的使用可参见paddlex.det.transforms。  

**3.4 定义数据集Dataset**  
目标检测可使用VOCDetection格式和COCODetection两种数据集，此处由于数据集为VOC格式，因此采用pdx.datasets.VOCDetection来加载数据集，该接口的介绍可参见文档paddlex.datasets.VOCDetection。  

**3.5模型选择**  
使用YOLOv3模型，DarkNet53网络  

# 四、模型训练
**4.1配置超参数训练模型**  
<center>step次数 如 **图3** 所示：
<center><img src="https://ai-studio-static-online.cdn.bcebos.com/d9f51b579bd24bfbb517fba3d9492bfedf88bb90609d4068a4ba82ffb9a10bb9" width="70%" height="60%"></center>
<center><br>图3：step次数 </br></center>
<br></br>



**Train samples: 3500, num_epochs=50(这里算1轮的50没乘), train_batch_size=20, Step=175**

**Notebook版本选BML Codecolab 选择output-yolov3_darknet53-vdl_log--里面的XXXX.log文件 启动VisualDL**
<center><img src="https://ai-studio-static-online.cdn.bcebos.com/ff79ecab80724f2f8ab1076babb21efcadb1e8ceb6614639a86261c601c61df7" width="70%" height="60%"></center>
<center><br>图4：调用可视化方法 </br></center>

**训练验证图示如下**
<center>iteration次数 如 **图3** 所示：
<center><img src="https://ai-studio-static-online.cdn.bcebos.com/26b6bd8cc25b41d0a1d734a75cb24fc74cd88621794f46e986cbbeee128ec618" width="90%" height="90%"></center>
<center><br>图5：训练验证图 </br></center>
<br></br>

# 五、可视化模型效果

