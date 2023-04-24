import torch 
from torch import nn
from collections import OrderedDict
from .MTLU_Package.MTLU import MTLU_AF
import math


class MTLU(nn.Module):
	def __init__(self,BinNum=20,BinWidth=0.1, FeatMapNum=64):
		super(MTLU, self).__init__()
		self.mtluweight = nn.Parameter(torch.zeros(FeatMapNum, BinNum))
		self.mtlubias   = nn.Parameter(torch.zeros(FeatMapNum, BinNum))
		HalfBinNum = int(BinNum/2)
		self.mtluweight.data[:,HalfBinNum:]=1
		self.MTLUpara = nn.Parameter(torch.zeros(2))
		self.MTLUpara.data[0] = BinNum
		self.MTLUpara.data[1] = BinWidth
	
	
	def forward(self, x):
		return MTLU_AF.apply(x, self.mtluweight, self.mtlubias, self.MTLUpara)

class MTSiLU(nn.Module):
    def __init__(self,BinNum=20,BinWidth=0.1, FeatMapNum=64):
        super(MTSiLU, self).__init__()
        self.mtluweight = nn.Parameter(torch.zeros(FeatMapNum, BinNum))
        self.mtlubias   = nn.Parameter(torch.zeros(FeatMapNum, BinNum))
        HalfBinNum = int(BinNum/2)
        self.mtluweight.data[:,HalfBinNum:]=1
        self.MTLUpara = nn.Parameter(torch.zeros(2))
        self.MTLUpara.data[0] = BinNum
        self.MTLUpara.data[1] = BinWidth
        self.sigmoid = nn.Sigmoid()
	
	
    def forward(self, x):
        input_ = x
        tmp = MTLU_AF.apply(x, self.mtluweight, self.mtlubias, self.MTLUpara)
        return tmp*self.sigmoid(input_)

class CPN_mc(nn.Module):
    def __init__(self, BinNum=20, BinWidth=0.1, FeatMapNum=64):
        super(CPN_mc, self).__init__()
        self.FeatMapNum = FeatMapNum
        self.mtlu_y = nn.Parameter(torch.zeros(FeatMapNum, BinNum))
        self.BinWidth = BinWidth
        self.BinNum = BinNum
        self.HalfBinNum = int(BinNum/2)
        self.mtlu_x = torch.tensor([x.data for x in torch.arange(-0.9, 1.1, self.BinWidth)])
        self.mtlu_y.data[:,self.HalfBinNum:] = self.mtlu_x[self.HalfBinNum:]
        self.MTLUpara = nn.Parameter(torch.zeros(2))
        self.MTLUpara.data[0] = BinNum
        self.MTLUpara.data[1] = BinWidth
        self.tmp1 = self.mtlu_y.data[:,:-1]
        self.tmp2 = torch.zeros(self.FeatMapNum,1)
        self.mtlu_y_ = torch.cat((self.tmp2,self.tmp1),1).cuda()
        self.index = torch.tensor([i for i in range(-self.HalfBinNum+1, self.HalfBinNum+1)]).cuda()

    def forward(self, x):
        self.mtluweight = (self.mtlu_y - self.mtlu_y_)/self.BinWidth
        self.mtlubias = self.mtlu_y - (self.mtlu_y - self.mtlu_y_) * self.index #
        return MTLU_AF.apply(x, self.mtluweight, self.mtlubias, self.MTLUpara)

class CPN_nl(nn.Module):
    def __init__(self, FeatMapNum=64,BinNum=4):
        super(CPN_nl,self).__init__()
        self.BinNum = BinNum
        self.FeatMapNum = FeatMapNum
        self.coef = nn.Parameter(torch.zeros(FeatMapNum, 3*BinNum))
        HalfBinNum = int(BinNum/2)
        self.coef.data[:,:HalfBinNum] = 1
        self.silu = nn.SiLU()
 
    def forward(self, x):
        a = self.coef[:, :self.BinNum].reshape(1,self.FeatMapNum,self.BinNum)
        b = self.coef[:, self.BinNum : 2 * self.BinNum].reshape(1,self.FeatMapNum,self.BinNum)
        c = self.coef[:, 2 * self.BinNum : 3 * self.BinNum].reshape(1,self.FeatMapNum,self.BinNum)
        x_perm = x.permute(2, 3, 0, 1).unsqueeze(-1)
        output = a * x_perm + b * self.silu(x_perm) + c
        result = torch.max(output, dim=-1)[0].permute(2, 3, 0, 1) 
        return result

class CPN_nl_1(nn.Module):
    def __init__(self, n_feats = 64):
        super(CPN_nl_1, self).__init__()
        self.mtlu = MTLU(FeatMapNum=n_feats)
        self.mtsilu = MTSiLU(FeatMapNum=n_feats)

    def forward(self, x):
        mtlu_x = self.mtlu(x)
        mtsilu_x = self.mtsilu(x)
        return mtlu_x + mtsilu_x

def sigma(x):
    return 1/(1 + math.exp(-x))
class CPN_nl_2(nn.Module):
    def __init__(self, FeatMapNum=64):
        super(CPN_nl_2,self).__init__()
        self.y1 = nn.Parameter(torch.zeros(1,FeatMapNum,1,1))
        self.y2 = nn.Parameter(torch.ones(1,FeatMapNum,1,1))
        self.silu = nn.SiLU()

    def forward(self, x):
        self.c = self.y1
        self.b = (self.y2 - 2 * self.y1)/(sigma(1)-sigma(-1))
        self.a = self.y1 - sigma(-1) * (self.y2 - 2 * self.y1)/(sigma(1)-sigma(-1))

        x = self.a * x + self.b * self.silu(x) + self.c
        return x

class xUnit(nn.Module):
    def __init__(self, num_features=64, kernel_size=7, batch_norm=True):
        super(xUnit, self).__init__()
        self.features = nn.Sequential(
            nn.BatchNorm2d(num_features=num_features) if batch_norm else Identity(),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=kernel_size, padding=(kernel_size // 2), groups=num_features),
            nn.BatchNorm2d(num_features=num_features) if batch_norm else Identity(),
            nn.Sigmoid()
        )

    def forward(self, x):
        a = self.features(x)     
        r = x * a
        return r
class Identity(nn.Module):
    def __init__(self,):
        super(Identity, self).__init__()
    def forward(self, x):
        return x


class DyReLUA(nn.Module):
    def __init__(self,
                 channels,
                 reduction=4,
                 k=2):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        self.k = k

        self.coef = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, 2 * k, 1),
            nn.Sigmoid()
        )

        self.register_buffer('lambdas', torch.Tensor([1.] * k + [0.5] * k).float())
        self.register_buffer('bias', torch.Tensor([1.] + [0.] * (2 * k - 1)).float())

    def forward(self, x):
        coef = self.coef(x)
        coef = 2 * coef - 1
        coef = coef.view(-1, 2 * self.k) * self.lambdas + self.bias

        x_perm = x.permute(1, 2, 3, 0).unsqueeze(-1) 
        output = x_perm * coef[:, :self.k] + coef[:, self.k:]
        result = torch.max(output, dim=-1)[0].permute(3, 0, 1, 2)
        return result
