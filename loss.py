import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16
import gc

class AdversialLoss(nn.Module):
    def __init__(self, cartoon_labels, fake_labels):
        super(AdversialLoss, self).__init__()
        self.cartoon_labels = cartoon_labels
        self.fake_labels = fake_labels
        self.base_loss = nn.BCEWithLogitsLoss()

    def forward(self, cartoon, generated_f, edge_f):
        #print(cartoon.shape, self.cartoon_labels.shape)
        D_cartoon_loss = self.base_loss(cartoon, self.cartoon_labels)
        D_generated_fake_loss = self.base_loss(generated_f, self.fake_labels)
        D_edge_fake_loss = self.base_loss(edge_f, self.fake_labels)

        # TODO Log maybe?
        return D_cartoon_loss + D_generated_fake_loss + D_edge_fake_loss
        
# #内容损失
# class ContentLoss(nn.Module):
#     def __init__(self, omega=10, device=None):
#         super(ContentLoss, self).__init__()

#         self.base_loss = nn.L1Loss()
#         self.omega = omega
#         self.device = device if device is not None else torch.device('cpu')

#         perception = list(vgg16(pretrained=True).features)[:25]
#         self.perception = nn.Sequential(*perception).eval()

#         for param in self.perception.parameters():
#             param.requires_grad = False

#         gc.collect()

#     def forward(self, x1, x2):
#         x1 = self.perception(x1)
#         x2 = self.perception(x2)
        
#         return self.omega * self.base_loss(x1, x2)

    
       
class ContentLoss(nn.Module):
    def __init__(self, omega=10, device=None):
        super(ContentLoss, self).__init__()
        
        # 初始化设备
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.omega = omega
        self.base_loss = nn.L1Loss().to(self.device)
        
        # 加载VGG16并截取前25层
        perception = vgg16(pretrained=True).features[:25]
        self.perception = perception.to(self.device).eval()  # 确保在目标设备上
        
        # 冻结参数
        for param in self.perception.parameters():
            param.requires_grad = False

        gc.collect()

    def forward(self, x1, x2):
        # 确保输入在相同设备上
        x1, x2 = x1.to(self.device), x2.to(self.device)
        
        # 特征提取前标准化（VGG要求）
        x1 = self.normalize(x1)
        x2 = self.normalize(x2)

        feat1 = self.perception(x1)
        feat2 = self.perception(x2)
        return self.omega * self.base_loss(feat1, feat2)
    
    @staticmethod
    def normalize(x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)  # 单通道转为三通道
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        return (x - mean) / std

