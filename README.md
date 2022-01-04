# DCGAN_pytorch(modified)
 
 env:
+  python >= 3.8.8
+  torchvision >= 0.9.1
+  pytorch >= 1.8.1

 Modified DCGAN implementation  of DCGAN. I use "shortcut" connection to enhance generator net, which can train faster than original DCGAN.

Before you start train your own net, you need to create your own dataset: just creat "image" folder in your current path, and put all your data in a subfolder.(if you already have a data, you can change the parser 'img_path' to load your own data)

Remember to change the 'img_channels' to 3 when you use RGB image. Grey value then change it to 1.

You can use command: "python dcgan.py" to train your own net.

---

 个人魔改了下DCGAN网络，把其中部分网络换成了residual connection，个人测试下，发现训练速度比原版网络快。

 环境要求不高，一般训练过网络的配置应该都是可以的，要是出现环境配置错误，在上面也列出了对应的主要包的版本。可以按照网上的教程进行配置

 代码中的注释也比较齐全，方便大家阅读。

 在训练自己的网络之前，你需要创建自己的数据集文件夹：在当前目录下创建‘image’文件夹，然后把所有训练图片放入‘image’文件下面的一个子文件夹就可以。
 
 当然也可以用经典的Fashion-MNIST数据集，在dcgan.py文件中注释掉原来获取数据的语句，取消下面的一段就可以了。
 本代码支持黑白照片和彩色图片，只需要在最前面的参数表中修改‘img_channels’即可完成。（黑白改为 1；彩色改为 3 ）

 注意，其中的dcgan.py才是真正的训练网络，下面的ResNet.py文件是自己实现的库，要两个都下载并放在同一目录下才能正确运行网络。
