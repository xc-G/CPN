import torch
import torch.nn as nn
from torch.autograd import Variable
import mtlu_cuda
import timeit

class MTLU_AF(torch.autograd.Function):
	@staticmethod
	def forward(self, x, weight, bias, paras):


		outputs = mtlu_cuda.forward(x, weight, bias, paras)
		y = outputs[0]
		indexmat = outputs[1]
		#print(indexmat)
		self.save_for_backward(x, weight, bias, paras, indexmat)
		return y

	@staticmethod
	def backward(self, grad_output):
		#starter = timeit.default_timer()
		x, weight, bias, paras, indexmat = self.saved_tensors
		grad_paras = None
		outputs =  mtlu_cuda.backward(x, weight, grad_output.data, indexmat, paras)			
		grad_input, grad_weight, grad_bias = outputs
		return grad_input, grad_weight, grad_bias, grad_paras

	# mblu_paras[0] feat_num  
	# mblu_paras[1] bin_width 
	# mblu_paras[2] bin_num
	# mblu_paras[3] count 
	# mblu_paras[4] feat_size

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


class MTLU_6(nn.Module):
    def __init__(self,BinNum=122,BinWidth=0.1, FeatMapNum=64):
        
        super(MTLU_6, self).__init__()
        self.mtluweight = nn.Parameter(torch.zeros(FeatMapNum, BinNum))
        self.mtlubias   = nn.Parameter(torch.zeros(FeatMapNum, BinNum))
        HalfBinNum = int(BinNum/2)
        self.mtluweight.data[:,HalfBinNum:-1]=1
        self.mtlubias.data[:,-1]=6
        self.MTLUpara = nn.Parameter(torch.zeros(2))
        self.MTLUpara.data[0] = BinNum
        self.MTLUpara.data[1] = BinWidth


    def forward(self, x):
        return MTLU_AF.apply(x, self.mtluweight, self.mtlubias, self.MTLUpara)


class MTLU_continuous(nn.Module):
    def __init__(self, BinNum=20, BinWidth=0.1, FeatMapNum=64):
        super(MTLU_continuous, self).__init__()
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

class MTLU_6_continuous(nn.Module):
    def __init__(self, BinNum=122, BinWidth=0.1, FeatMapNum=64):
        super(MTLU_6_continuous, self).__init__()
        self.FeatMapNum = FeatMapNum
        self.mtlu_y = nn.Parameter(torch.zeros(FeatMapNum, BinNum))
        self.BinWidth = BinWidth
        self.BinNum = BinNum
        self.HalfBinNum = BinNum // 2
        self.mtlu_x = torch.tensor([x.data for x in torch.arange(-6.0, 6.2, self.BinWidth)])
        self.mtlu_y.data[:,self.HalfBinNum:-1] = self.mtlu_x[self.HalfBinNum:-1]
        self.mtlu_y.data[:,-1] = 6.0
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