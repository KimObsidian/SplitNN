## Split_dp说明文档

### split_mnist

- 该文件夹下对mnist数据集进行分割
- no_split.py是不使用split的网络结构
- distribute_data.py是对数据集进行分割
- splitnn_net.py是split的网络结构
- main.py是函数入口

### split_paddle

- 需要安装paddle,命令为pip install paddlepaddle==1.8.5

- 该文件夹下对movie_lens数据集进行分割
- process_data.py获得movie_lens数据集的数据，user_watch和user_search均为64维，user_other为32维，label是标签
- splitnn_net.py是split的网络结构
- distribute_data.py是对数据集进行分割，目前的分割方法是64，96
- split.py是函数入口


### split_paddle_padding

- 采用padding方法，对分割的64维，96维向量分别用0填充为2个160维的向量。
这种padding保持位置不变，比如（x1,x2,x3)划分为（x1,x2)和（x3)，填充后变为
  （x1,x2,0)和（0,x3）
