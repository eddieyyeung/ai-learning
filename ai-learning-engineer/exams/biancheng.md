### 题目31：实现矩形转换正方形代码

```python
import numpy as np
def convert_to_square(bbox):
    square_bbox = bbox.copy()
    if bbox.shape[0] == 0:
        return np.array([])
    h = bbox[:, 3] - bbox[:, 1]
    w = bbox[:, 2] - bbox[:, 0]
    max_side = np.maximum(h, w)
    #中心点-最大边长一半
    square_bbox[:, 0] = bbox[:, 0] + w * 0.5 - max_side * 0.5
    square_bbox[:, 1] = bbox[:, 1] + h * 0.5 - max_side * 0.5
    square_bbox[:, 2] = square_bbox[:, 0] + max_side
    square_bbox[:, 3] = square_bbox[:, 1] + max_side

    return square_bbox

if __name__ == '__main__':
    box=np.array([[16,36,65,94],[54,62,79,82]])
    print(convert_to_square(box))
```

### 题目32：按图所示完成Q-Learning回报矩阵的更新，其中超参数 alpha=1, gamma=0.8(10分)
![1762532204535](image/biancheng/1762532204535.png)

```python
import numpy as np
r_array = np.array([
    [-1, -1, -1, -1, 0, -1],
    [-1, -1, -1, 0, -1, 100],
    [-1, -1, -1, 0, -1, -1],
    [-1, 0, 0, -1, 0, -1],
    [0, -1, -1, 0, -1, 100],
    [-1, 0, -1, -1, 0, 100]], dtype=np.float32)
q_array = np.zeros(36).reshape(6,6)
alpha, gamma = 1, 0.8
for epoch in range(20):
    for i in range(6):
        for j in range(6):
            if r_array[i][j]>=0:
                q_array[i][j]=q_array[i][j]+alpha*(r_array[i][j]+gamma*max(q_array[j])-q_array[i][j])
            else:
                continue
print((q_array/np.max(q_array)*100).round())
```

### 题目33：使用TensorFlow1.x框架完成输入输出x、y和参数w、b的图构建，并且使用会话输出参数w的值(10分)

```python
import tensorflow as tf
print(tf.__version__)
#图当中是看不到真实数据的，只有数据的形状和类型
x = tf.placeholder(dtype=tf.float32,shape=[None,784])
y = tf.placeholder(dtype=tf.float32,shape=[None,10])
w = tf.Variable(tf.random_normal(shape=[784,10],stddev=0.01,dtype=tf.float32))
b = tf.Variable(tf.zeros(shape=[10],dtype=tf.float32))

print(x)
print(w)

# 初始化图中全局变量的操作.
init = tf.global_variables_initializer()
#会话当中可以看到真实数据
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(w))
```

### 题目34：完成YOLOv5模型中的Focus模块代码(10分)

```python
import torch
from torch import nn

class Focus(nn.Module):
    def __init__(self, c1, c2, k=1):
        super(Focus, self).__init__()
        self.conv = nn.Conv2d(c1 * 4, c2, k, 1)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))

if __name__ == '__main__':
    data=torch.randn(1,64,224,224)
    focus=Focus(64,64)
    out=focus(data)
    print(out.shape)
```

### 题目31：实现图中的网络代码

![1762526805893](image/biancheng/1762526805893.png)

```python
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=3, stride=1),  # 10*10*3
            # nn.BatchNorm2d(10),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # 5*5*10
            nn.Conv2d(10, 16, kernel_size=3, stride=1),  # 3*3*16
            # nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),  # 1*1*32
            # nn.BatchNorm2d(32),
            nn.PReLU()
        )
        self.conv4_1 = nn.Conv2d(32, 2, kernel_size=1, stride=1)
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1, stride=1)
        self.conv4_3 = nn.Conv2d(32, 10, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.pre_layer(x)
        cls = torch.softmax(self.conv4_1(x), 1)
        bbox = self.conv4_2(x)
        landmark = self.conv4_3(x)
        return cls, bbox, landmark

if __name__ == '__main__':
    data = torch.randn(10, 3, 12, 12)
    net = Net()
    out = net(data)
    print(out[0].shape)
    print(out[1].shape)
    print(out[2].shape)
```

### 题目32：使用pytorch创建Embedding对象，并一次性输出顺序和倒序的值

```python
import torch
from torch import nn
a = nn.Embedding(4,5)
print(a.weight)
idx = torch.tensor([[0,1,2,3],[3,2,1,0]])
print(a(idx))
```

### 题目31： 使用训练好的U²net模型将下面图像中左图的人物分割出来，将图像背景（非人物部分）替换成透明背景（如右图）

![1762527412735](image/biancheng/1762527412735.png)

```python
import numpy as np
from PIL import Image

def crop(img_file, mask_file):
    img_array = np.array(Image.open(img_file))
    mask = np.array(Image.open(mask_file))

    #从mask中随便找一个通道，cat到原图的RGB通道后面，转成RGBA通道模式
    res = np.concatenate((img_array, mask[:, :, [0]]), -1)
    img = Image.fromarray(res.astype('uint8'), mode='RGBA')
    return img

if __name__ == "__main__":
    img_file = "1.jpg"
    mask_root = "2.png"
    res = crop(img_file,mask_root)
    print(res.mode)
    res.show()
    res.save("./{}.png".format("result"))
```

### 题目34：实现Centerloss主体代码

```python
import torch
import torch.nn as nn
class Centerloss(nn.Module):
    def __init__(self,lambdas,feature_num=2,class_num=10):
        super(Centerloss,self).__init__()
        self.lambdas=lambdas
        self.center = nn.Parameter(torch.randn(class_num, feature_num), requires_grad=True)
        # self.center = torch.randn(class_num, feature_num, requires_grad=True)
    def forward(self, feature,label):
        center_exp = self.center.index_select(dim=0, index=label.long())
        count = torch.histc(label, bins=int(max(label).item() + 1), min=0, max=int(max(label).item()))
        count_exp = count.index_select(dim=0, index=label.long())
        loss = self.lambdas*2*torch.mean(torch.div(torch.sum(torch.pow(feature - center_exp, 2),dim=1),count_exp))
        return loss
if __name__ == '__main__':
    data = torch.tensor([[3, 4], [5, 6], [7, 8], [9, 8], [6, 5]], dtype=torch.float32)
    label = torch.tensor([0, 0, 1, 0, 1], dtype=torch.float32)
    center_loss=Centerloss(2,2,5)
    print(center_loss(data,label))
    print(list(center_loss.parameters()))
```

### 题目32： 完成Unet主干模型的代码

```python
import torch
import torch.nn.functional as F
class CNNLayer(torch.nn.Module):
    def __init__(self, C_in, C_out):
        super(CNNLayer, self).__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Conv2d(C_in, C_out, 3, 1, 1),
            torch.nn.BatchNorm2d(C_out),
            torch.nn.Dropout(0.3),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(C_out, C_out, 3, 1, 1),
            torch.nn.BatchNorm2d(C_out),
            torch.nn.Dropout(0.4),
            torch.nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


class DownSampling(torch.nn.Module):
    def __init__(self, C):
        super(DownSampling, self).__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Conv2d(C, C, 3, 2, 1),
            torch.nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


class UpSampling(torch.nn.Module):
    def __init__(self, C):
        super(UpSampling, self).__init__()
        self.C = torch.nn.Conv2d(C, C // 2, 1, 1)#逐点卷积

    def forward(self, x, r):
        up = F.interpolate(x, scale_factor=2, mode='nearest')#临近插值
        x = self.C(up)
        return torch.cat((x, r), 1)


class MainNet(torch.nn.Module):
    def __init__(self):
        super(MainNet, self).__init__()
        self.C1 = CNNLayer(3, 64)
        self.D1 = DownSampling(64)#128
        self.C2 = CNNLayer(64, 128)
        self.D2 = DownSampling(128)#64
        self.C3 = CNNLayer(128, 256)
        self.D3 = DownSampling(256)#32
        self.C4 = CNNLayer(256, 512)
        self.D4 = DownSampling(512)#16
        self.C5 = CNNLayer(512, 1024)
        self.U1 = UpSampling(1024)#32
        self.C6 = CNNLayer(1024, 512)
        self.U2 = UpSampling(512)#64
        self.C7 = CNNLayer(512, 256)
        self.U3 = UpSampling(256)#128
        self.C8 = CNNLayer(256, 128)
        self.U4 = UpSampling(128)#256
        self.C9 = CNNLayer(128, 64)
        self.pre = torch.nn.Conv2d(64, 3, 3, 1, 1)#转换通道
        self.Th = torch.nn.Sigmoid()

    def forward(self, x):
        R1 = self.C1(x)
        R2 = self.C2(self.D1(R1))#128
        R3 = self.C3(self.D2(R2))#64
        R4 = self.C4(self.D3(R3))#32
        Y1 = self.C5(self.D4(R4))#16
        O1 = self.C6(self.U1(Y1, R4))#32
        O2 = self.C7(self.U2(O1, R3))#64
        O3 = self.C8(self.U3(O2, R2))#128
        O4 = self.C9(self.U4(O3, R1))#256
        return self.Th(self.pre(O4))#[3,256,256]

if __name__ == '__main__':
    a = torch.randn(2, 3, 416, 416).cuda()
    net = MainNet().cuda()
    print(net(a).shape)
```

### 题目32： 实现IOU代码

```python
import numpy as np


def iou(box, boxes, isMin = False):
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    w = np.maximum(0, xx2 - xx1)
    h = np.maximum(0, yy2 - yy1)

    inter = w * h
    if isMin:
        ovr = np.true_divide(inter, np.minimum(box_area, area))
    else:
        ovr = np.true_divide(inter, (box_area + area - inter))

    return ovr

if __name__ == '__main__':
    a = np.array([1,1,11,11])
    bs = np.array([[1,1,10,10],[11,11,20,20]])
    print(iou(a,bs))
```

### 题目33： 按图所示，使用python将左侧的R回报矩阵更新为为右侧的Q矩阵，其中超参数 alpha=1, gamma=0.8

![1762532363767](image/biancheng/1762532363767.png)

```python
import numpy as np
r_array = np.array([
    [-1, -1, -1, -1, 0, -1],
    [-1, -1, -1, 0, -1, 100],
    [-1, -1, -1, 0, -1, -1],
    [-1, 0, 0, -1, 0, -1],
    [0, -1, -1, 0, -1, 100],
    [-1, 0, -1, -1, 0, 100]], dtype=np.float32)
q_array = np.zeros(36).reshape(6,6)
alpha, gamma = 1,0.8
for epoch in range(20):
    for i in range(6):
        for j in range(6):
            if r_array[i][j]>=0:
                q_array[i][j]=q_array[i][j]+alpha*(r_array[i][j]+gamma*max(q_array[j])-q_array[i][j])
            else:
                continue
print((q_array/np.max(q_array)*100).round())
```

### 题目34： 使用6层全连接实现AutoEncoder主网络代码

```python
from torch import nn
import torch
class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.aenet=nn.Sequential(
            nn.Linear(784,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid()
        )
    def forward(self,x):
        return self.aenet(x)

if __name__ == '__main__':
    data=torch.randn(10,784)
    ae=AutoEncoder()
    out=ae(data)
    print(out.shape)
```

### 题目33： 根据torchtext的GloVe词向量包里的"china"和"beijing"的关系计算出"japan"对应的词

```python
import torch
import torchtext
gv = torchtext.vocab.GloVe(name="6B",dim=50)#一个词用50个长度的向量来表示

def get_wv(word):
    return gv.vectors[gv.stoi[word]]

def sim_10(word,n = 10):
    aLL_dists = [(gv.itos[i],torch.dist(word,w)) for i,w in enumerate(gv.vectors)]
    return sorted(aLL_dists,key=lambda t: t[1])[:n]

def answer(w1,w2,w3):
    print("{0}：{1}=={2}：{3}".format(w1,w2,w3,"x"))
    w4 = get_wv(w3)-get_wv(w1)+get_wv(w2)
    print(sim_10(w4))
    return sim_10(w4)[0][0]#拿出10组中的第一组的第一个值，也就是距离最小的词
print("x="+answer("china","beijing","japan"))
```

### 题目34： 使用三层全连接分别完成原始GAN的D网络和G网络的代码

```python
from torch import nn
class D_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.dnet=nn.Sequential(
            nn.Linear(784,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,1),
            nn.Sigmoid()
        )
    def forward(self,x):
        return self.dnet(x)

class G_Net(nn.Module):
    def __init__(self):
        super(G_Net, self).__init__()
        self.gnet=nn.Sequential(
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Linear(512,784)
        )
    def forward(self,x):
        return self.gnet(x)
```

### 题目32： 实现ArcFace Loss主体代码
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ArcNet(nn.Module):
    def __init__(self,feature_dim=2,cls_dim=10):
        super().__init__()
        #生成一个隔离带向量，训练这个向量和原来的特征向量尽量分开，达到增加角度的目的
        self.W=nn.Parameter(torch.randn(feature_dim,cls_dim).cuda(),requires_grad=True)
    def forward(self, feature,m=0.5,s=64):
        #对特征维度进行标准化
        x = F.normalize(feature,dim=1)#shape=【100，2】
        w = F.normalize(self.W, dim=0)#shape=【2，10】

        # s = torch.sqrt(torch.sum(torch.pow(x, 2))) * torch.sqrt(torch.sum(torch.pow(w, 2)))
        cosa = torch.matmul(x, w)/s

        a=torch.acos(cosa)#反三角函数得出的是弧度，而非角度，1弧度=1*180/3.14=57角度
        # 这里对e的指数cos(a+m)再乘回来，让指数函数的输出更大，
        # 从而使得arcsoftmax输出更小，即log_arcsoftmax输出更小，则-log_arcsoftmax更大。
        arcsoftmax = torch.exp(
            s * torch.cos(a + m)) / (torch.sum(torch.exp(s * cosa), dim=1, keepdim=True) - torch.exp(
            s * cosa) + torch.exp(s * torch.cos(a + m)))

        return arcsoftmax

if __name__ == '__main__':

    arc=ArcNet(feature_dim=2,cls_dim=10)
    feature=torch.randn(100,2).cuda()
    out=arc(feature)
    print(feature.shape)
    print(out.shape)
```

### 题目31： 实现NMS代码
```python
import numpy as np
def iou(box, boxes, isMin = False):
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    w = np.maximum(0, xx2 - xx1)
    h = np.maximum(0, yy2 - yy1)

    inter = w * h
    if isMin:
        ovr = np.true_divide(inter, np.minimum(box_area, area))
    else:
        ovr = np.true_divide(inter, (box_area + area - inter))

    return ovr


def nms(boxes, thresh=0.3, isMin = False):

    if boxes.shape[0] == 0:
        return np.array([])

    _boxes = boxes[(-boxes[:, 4]).argsort()]
    r_boxes = []

    while _boxes.shape[0] > 1:
        a_box = _boxes[0]
        b_boxes = _boxes[1:]

        r_boxes.append(a_box)

        # print(iou(a_box, b_boxes))

        index = np.where(iou(a_box, b_boxes,isMin) < thresh)
        _boxes = b_boxes[index]

    if _boxes.shape[0] > 0:
        r_boxes.append(_boxes[0])

    return np.stack(r_boxes)

if __name__ == '__main__':

    bs = np.array([[1, 1, 10, 10, 0.98], [1, 1, 9, 9, 0.8], [9, 8, 13, 20, 0.7], [6, 11, 18, 17, 0.85]])
    print((-bs[:,4]).argsort())
    print(nms(bs))
```

### 题目31： 完成对音频数据0_0_0_0_1_1_1_1.wav使用μ律压缩方法进行编码和解码的代码
```python
import torchaudio
import matplotlib.pyplot as plt
filename = r"0_0_0_0_1_1_1_1.wav"

#从语音文件读出音频图和采样率
waveform, sample_rate = torchaudio.load(filename)

#先查看当前的音频最大最小值，因为编码重构音频需要将输入的原音频数据规范到[-1,1]之间
print("Min of waveform:{}\nMax of waveform:{}\nMean of waveform:{}".format(waveform.min(),waveform.max(),waveform.mean()))

#如果音频数据本身就是在[-1,1]之间，就不需要进行缩放了
def normalize(tensor):
    # 减去均值，然后缩放到[-1,1]
    tensor_minusmean = tensor - tensor.mean()
    return tensor_minusmean/tensor_minusmean.abs().max()

# waveform = normalize(waveform)


#先对音频数据进行编码，输入范围是[-1,1]，输出范围是[0,255]
transformed = torchaudio.transforms.MuLawEncoding()(waveform)
print("Shape of transformed waveform: {}".format(transformed.size()))
plt.figure()
plt.plot(transformed[0, :].numpy())
plt.show()

#然后对编码后的音频数据解码重构，输入范围是[0,255]，输出范围和原音频范围一致
reconstructed = torchaudio.transforms.MuLawDecoding()(transformed)
print("Shape of recovered waveform: {}".format(reconstructed.size()))
plt.figure()
plt.plot(reconstructed[0, :].numpy())
plt.show()

#最后将重构的音频数据和原始音频数据进行比较，查看误差率
err = ((waveform - reconstructed).abs() / waveform.abs()).median()
print("Median relative difference between original and MuLaw reconstucted signals: {:.2%}".format(err))
```

### 题目33： 完成YOLOv3模型中的残差模块和卷积集模块代码

```python
import torch
#定义卷积层
class ConvolutionalLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(ConvolutionalLayer, self).__init__()

        self.sub_module = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.sub_module(x)

#定义残差结构
class ResidualLayer(torch.nn.Module):

    def __init__(self, in_channels):
        super(ResidualLayer, self).__init__()

        self.sub_module = torch.nn.Sequential(
            ConvolutionalLayer(in_channels, in_channels // 2, 1, 1, 0),
            ConvolutionalLayer(in_channels // 2, in_channels, 3, 1, 1),
        )

    def forward(self, x):
        return x + self.sub_module(x)

#定义卷积块
class ConvolutionalSet(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvolutionalSet, self).__init__()

        self.sub_module = torch.nn.Sequential(
            ConvolutionalLayer(in_channels, out_channels, 1, 1, 0),
            ConvolutionalLayer(out_channels, in_channels, 3, 1, 1),

            ConvolutionalLayer(in_channels, out_channels, 1, 1, 0),
            ConvolutionalLayer(out_channels, in_channels, 3, 1, 1),

            ConvolutionalLayer(in_channels, out_channels, 1, 1, 0),
        )

    def forward(self, x):
        return self.sub_module(x)
```

### 题目31： 完成YOLOv3模型中的dataset采样代码

```python
from torch.utils.data import Dataset,DataLoader
import torchvision
import numpy as np
import os
from PIL import Image
import math

class cfg:
    IMG_HEIGHT = 416
    IMG_WIDTH = 416
    CLASS_NUM = 10

    "anchor box是对coco数据集聚类获得"
    ANCHORS_GROUP_KMEANS = {
        52: [[10, 13], [16, 30], [33, 23]],
        26: [[30, 61], [62, 45], [59, 119]],
        13: [[116, 90], [156, 198], [373, 326]]}

    ANCHORS_GROUP = {
        13: [[360, 360], [360, 180], [180, 360]],
        26: [[180, 180], [180, 90], [90, 180]],
        52: [[90, 90], [90, 45], [45, 90]]}

    ANCHORS_GROUP_AREA = {
        13: [x * y for x, y in ANCHORS_GROUP[13]],
        26: [x * y for x, y in ANCHORS_GROUP[26]],
        52: [x * y for x, y in ANCHORS_GROUP[52]],
    }


transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
    mean = [0.485, 0.456, 0.406],
    std = [0.229, 0.224, 0.225])
])

def one_hot(cls_num, i):
    b = np.zeros(cls_num)
    b[i] = 1.
    return b

class MyDataset(Dataset):

    def __init__(self,label_path,image_dir):
        self.label_apth=label_path
        self.image_dir=image_dir
        with open(self.label_apth) as f:
            self.dataset = f.readlines()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        labels = {}
        line = self.dataset[index]
        strs = line.split()
        _img_data = Image.open(os.path.join(self.image_dir, strs[0]))
        img_data = transforms(_img_data)
        _boxes = np.array(float(x) for x in strs[1:])
        _boxes = np.array(list(map(float, strs[1:])))
        boxes = np.split(_boxes, len(_boxes) // 5)

        for feature_size, anchors in cfg.ANCHORS_GROUP.items():
            labels[feature_size] = np.zeros(shape=(feature_size, feature_size, 3, 5 + cfg.CLASS_NUM))
            for box in boxes:
                cls, cx, cy, w, h = box
                cx_offset, cx_index = math.modf(cx * feature_size / cfg.IMG_WIDTH)
                cy_offset, cy_index = math.modf(cy * feature_size / cfg.IMG_WIDTH)
                for i, anchor in enumerate(anchors):
                    anchor_area = cfg.ANCHORS_GROUP_AREA[feature_size][i]
                    p_w, p_h = w / anchor[0], h / anchor[1]
                    # tw=np.log(w / anchor[0])
                    p_area = w * h
                    iou = min(p_area, anchor_area) / max(p_area, anchor_area)
                    labels[feature_size][int(cy_index), int(cx_index), i] = np.array(
                        [iou, cx_offset, cy_offset, np.log(p_w), np.log(p_h), *one_hot(cfg.CLASS_NUM, int(cls))])#10,i
        return labels[13], labels[26], labels[52], img_data
if __name__ == '__main__':
    x=one_hot(10,2)
    print(x)
    LABEL_FILE_PATH = "data/person_label.txt"
    IMG_BASE_DIR = "data/"

    data = MyDataset(LABEL_FILE_PATH,IMG_BASE_DIR)
    dataloader = DataLoader(data,2,shuffle=True)
    for target_13, target_26, target_52, img_data in dataloader:
        print(target_13.shape)
        print(target_26.shape)
        print(target_52.shape)
        print(img_data.shape)
```

### 题目33： 完成U²net中掩码分割图和原图合并成透明背景图的代码
```python
import numpy as np
from PIL import Image

def crop(img_file, mask_file):
    img_array = np.array(Image.open(img_file))
    mask = np.array(Image.open(mask_file))

    #从mask中随便找一个通道，cat到原图的RGB通道后面，转成RGBA通道模式
    res = np.concatenate((img_array, mask[:, :, [0]]), -1)
    img = Image.fromarray(res.astype('uint8'), mode='RGBA')
    return img

if __name__ == "__main__":
    img_file = "1.jpg"
    mask_root = "2.png"
    res = crop(img_file,mask_root)
    print(res.mode)
    res.show()
    res.save("./{}.png".format("result"))
```

### 题目31： 完成Unet模型中对图像转正方形并居中处理的代码
```python
import PIL.Image as pimg
"按照缩放的边长对图片等比例缩放，并转成正方形居中"
def scale_img(img,scale_side):
    # "获得图片宽高"
    w1, h1 = img.size
    # print(w1,h1)
    # "根据最大边长缩放,图像只会被缩小，不会变大"
    # "当被缩放的图片宽和高都小于缩放尺寸的时候，图像不变"
    img.thumbnail((scale_side, scale_side))
    # "获得缩放后的宽高"
    w2, h2 = img.size
    # print(w2,h2)
    # "获得缩放后的比例"
    s1 = w1 / w2
    s2 = h1 / h2
    s = (s1 + s2) / 2
    # "新建一张scale_side*scale_side的空白黑色背景图片"
    bg_img = pimg.new("RGB", (scale_side, scale_side), (0, 0, 0))
    # "根据缩放后的宽高粘贴图像到背景图上"
    if w2 == scale_side:
        bg_img.paste(img, (0, int((scale_side - h2) / 2)))
    elif h2 == scale_side:
        bg_img.paste(img, (int((scale_side - w2) / 2), 0))
    # "原图比缩放后的图要小的时候"
    else:
        bg_img.paste(img, (int((scale_side - w2) / 2), (int((scale_side - h2) / 2))))
    return bg_img

if __name__ == '__main__':
    image="img.png"
    new_img=scale_img(pimg.open(image),416)
    new_img.show()
```

### 题目34： 完成YOLOv3模型中的上采样层和下采样层代码
```python
import torch
import torch.nn.functional as F
#定义上采样层，邻近插值
class UpsampleLayer(torch.nn.Module):
    def __init__(self):
        super(UpsampleLayer, self).__init__()

    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode='nearest')

#定义卷积层
class ConvolutionalLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(ConvolutionalLayer, self).__init__()

        self.sub_module = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.sub_module(x)

#定义下采样层
class DownsamplingLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsamplingLayer, self).__init__()

        self.sub_module = torch.nn.Sequential(
            ConvolutionalLayer(in_channels, out_channels, 3, 2, 1)
        )

    def forward(self, x):
        return self.sub_module(x)
```