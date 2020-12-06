@[TOC](基于卷积神经网络的手写数字识别（附数据集+代码）)
本项目详细讲解见CSDN博客https://blog.csdn.net/ShakalakaPHD/article/details/110694933
# 配置环境
**使用环境：python3.8
平台：Windows10
IDE：PyCharm**

# 1.前言
手写数字识别，作为机器视觉入门项目，无论是基于传统的OpenCV方法还是基于目前火热的深度学习、神经网络的方法都有这不错的训练效果。当然，这个项目也常常被作为大学/研究生阶段的课程实验。可惜的是，目前网络上关于手写数字识别的项目代码很多，但是普遍不完整，对于初学者提出了不小的挑战。为此，博主撰写本文，无论你是希望借此完成课程实验或者学习机器视觉，本文或许对你都有帮助。

# 2.问题描述
本文针对的问题为：随机在黑板上写一个数字，通过调用电脑摄像头实时检测出数字是0-9哪个数字
# 3.解决方案
基于Python的深度学习方法：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201205161558256.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NoYWthbGFrYVBIRA==,size_16,color_FFFFFF,t_70)
检测流程如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201205161624520.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NoYWthbGFrYVBIRA==,size_16,color_FFFFFF,t_70)
# 4.实现步骤
## 4.1数据集选择
手写数字识别经典数据集：本文数据集选择的FishionMint数据集中的t10k，共含有一万张28*28的手写图片（二值图片）
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201205161810634.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NoYWthbGFrYVBIRA==,size_16,color_FFFFFF,t_70)
数据集下载地址见：[https://github.com/ShakalakaPHD/Hand_wrtten/tree/master/dataset](https://github.com/ShakalakaPHD/Hand_wrtten/tree/master/dataset)
## 4.2构建网络
采用Resnt（残差网络），残差网络的优势在于：

 - 更易捕捉模型细微波动
 - 更快的收敛速度
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201205172633204.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NoYWthbGFrYVBIRA==,size_16,color_FFFFFF,t_70)
本文的网络结构如下图所示，代码见第五节：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201205172657252.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NoYWthbGFrYVBIRA==,size_16,color_FFFFFF,t_70)
## 4.3训练网络
本文设置训练次数为100个循环，其实网络的训练过程是这样的：
 - 给网络模型“喂”数据（图像+标签）
 - 网络根据“喂”来的数据不断自我修正权重
 - 本文一共“喂”100次1万张图像
 - RTX2070上耗时2h
训练结果如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020120517304258.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NoYWthbGFrYVBIRA==,size_16,color_FFFFFF,t_70)
## 4.4测试网络
 - 随机选取数据集中37张图片进行检测
 - 正确率为36/37
 - 选取其中6张进行展示
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201205173130786.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NoYWthbGFrYVBIRA==,size_16,color_FFFFFF,t_70)
## 4.5图像预处理
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201205173200466.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NoYWthbGFrYVBIRA==,size_16,color_FFFFFF,t_70)
 - 全部采取传统机器视觉的方法
 - 速度“飞快”，仅做以上操作处理速度高达200fps
## 4.6传入网络进行计算
 
 - 手写0-9的数字除了3识别不了其余均能识别
 - 检测速度高达60fps
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201205173339222.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NoYWthbGFrYVBIRA==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201205173343271.png)
# 5.代码实现
本文所有代码都已经上传至Github上[https://github.com/ShakalakaPHD/Hand_wrtten/tree/master](https://github.com/ShakalakaPHD/Hand_wrtten/tree/master)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201205173528842.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NoYWthbGFrYVBIRA==,size_16,color_FFFFFF,t_70)
## 5.1文件说明
 - dataset文件夹存放的是训练数据集
 - logs文件夹为训练结束后权重文件所在
 - real_img、real_img_resize、test_imgs为用来测试的图片文件夹
 - 下面的py文件为本文代码
 ## 5.2使用方法
 按照博主的环境配置自己的Python环境
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201205173902822.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NoYWthbGFrYVBIRA==,size_16,color_FFFFFF,t_70)
其中主要的包有：numpy、struct、matplotlib、OpenCV、Pytorch、torchvision、tqdm
## 5.3 训练模型
本文提供了训练好的模型，大家可以直接调用，已经上传至GitHub，如果不想训练的话，可以跳过训练这一步骤
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201206161200636.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NoYWthbGFrYVBIRA==,size_16,color_FFFFFF,t_70)
下面是训练的流程：

打开hand_wrtten_train.py文件，点击运行（博主使用的是PyCharm，大家根据自己喜好选择IDLE即可）
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201205174440507.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NoYWthbGFrYVBIRA==,size_16,color_FFFFFF,t_70)
值得注意的是，<font color=#0099ff size=5 face="黑体">数据集路径需要修改为自己的路径</font>，即这一段
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201205174606737.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NoYWthbGFrYVBIRA==,size_16,color_FFFFFF,t_70)
训练过程没报错会出现以下显示
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201205174801746.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NoYWthbGFrYVBIRA==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201205174815140.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NoYWthbGFrYVBIRA==,size_16,color_FFFFFF,t_70)
训练得到的权重会保存在logs文件夹下
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201205174902112.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NoYWthbGFrYVBIRA==,size_16,color_FFFFFF,t_70)
模型训练需要时间，此时等待训练结束即可（RTX2070上训练了1h左右）
## 5.4使用训练好的模型测试网络
测试采用图片进行测试，代码见main_pthoto.py文件，使用方法与上面训练代码一直，代开后运行即可
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201206155424345.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NoYWthbGFrYVBIRA==,size_16,color_FFFFFF,t_70)
同样值得注意的是，<font color=#0099ff size=5 face="黑体">main_pthoto.py文件中图片路径需要修改为自己的路径</font>，即这一段
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201206155638322.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NoYWthbGFrYVBIRA==,size_16,color_FFFFFF,t_70)
以及<font color=#0099ff size=5 face="黑体">predict.py文件中权重片路径需要修改为自己在5.3步中训练得到的.pth文件路径</font>，如图所示
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201206155937446.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NoYWthbGFrYVBIRA==,size_16,color_FFFFFF,t_70)
运行结果如下
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201206161024834.gif)
## 5.5调用摄像头实时检测
代码存在于main.py文件下，使用方法和5.4节图片检测一致，修改<font color=#0099ff size=5 face="黑体">predict.py文件中权重片路径需要修改为自己在5.3步中训练得到的.pth文件路径</font>，如图所示
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201206155937446.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NoYWthbGFrYVBIRA==,size_16,color_FFFFFF,t_70)
再运行main.py文件即可，可以看到载入网络模型后开始调用摄像头，并开始检测
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201206170712156.gif)


# 6.结束语
如果本文对你有帮助的话还请给个star哦，你的支持是我最大的动力！(づ｡◕ᴗᴗ◕｡)づ
