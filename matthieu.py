#############################################################################
# Import                                                                    #
#############################################################################
import os
import random
import PIL.Image as Image
from tqdm import tqdm

import numpy as np
import scipy.io

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn.functional as F

class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

#############################################################################
# Hyperparameters                                                           #
#############################################################################   
opt = DotDict()

opt.dataset = 'celebA'
opt.dataPath = './data'

# Input space
opt.nc = 3                    # number of input channels
opt.sizeX = 64                # size of the image
opt.sizeS = 64                # size of random noise S vectors
opt.sizeZ = 512               # size of random noise Z vectors

# Convolution settings
opt.nf = 64                   # base number of filter in G and D

# Hardward settings
opt.workers = 4               # workers data for preprocessing
opt.cuda = True               # use CUDA
opt.gpu = 0                   # GPU id

# Optimisation scheme
opt.batchSize = 128           # minibatch size
opt.nIteration = 1000001      # number of training iterations
opt.lrG = 2e-4                # learning rate for G
opt.lrD = 5e-5                # learning rate for D
opt.recW = 0.5
opt.swap1W = 1
opt.swap2W = 1
opt.classW = 0
opt.klzW = .1

# Save/Load networks
opt.checkpointDir = '.'       # checkpoints directory
opt.load = 0                  # if > 0, load given checkpoint
opt.checkpointFreq = 5        # frequency of checkpoints (in number of epochs)

#############################################################################
# Loading Weights                                                           #
#############################################################################
opt.netEnc = ''
opt.netDec = ''
opt.netD = ''
if opt.load > 0:
    opt.netEnc = '%s/netEnc_%d.pth' % (opt.checkpointDir, opt.load)
    opt.netDec = '%s/netDec_%d.pth' % (opt.checkpointDir, opt.load)
    opt.netD = '%s/netD_%d.pth' % (opt.checkpointDir, opt.load)

#############################################################################
# RandomSeed                                                                #
#############################################################################
opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

#############################################################################
# CUDA                                                                      #
#############################################################################
cudnn.benchmark = True
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
if opt.cuda:
    torch.cuda.set_device(opt.gpu)

#############################################################################
# Dataloader                                                                #
#############################################################################
if opt.dataset == 'celebA':
    opt.nClass = 10000
elif opt.dataset == '3Dchairs':
    opt.nClass = 1300


class PairCelebADataset(torch.utils.data.Dataset):
    def __init__(self, dataPath, labelFile, transform=transforms.ToTensor()):
        super(PairCelebADataset, self).__init__()
        self.dataPath = dataPath
        with open(labelFile, 'r') as f:
            lines = np.array([p.split() for p in f.readlines()])
        self.files = lines[:,0]
        self.labels = lines[:,1].astype(int)        
        self.transform = transform
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        label = self.labels[idx]
        file1 = self.files[idx]
        file2 = np.random.choice(self.files[self.labels == label])
        img1 = self.transform(Image.open(os.path.join(self.dataPath, file1)))
        img2 = self.transform(Image.open(os.path.join(self.dataPath, file2)))
        return img1, img2, torch.LongTensor(1).fill_(int(label))
    
class Pair3DchairsDataset(torch.utils.data.Dataset):
    def __init__(self, dataPath, transform=transforms.ToTensor()):
        super(Pair3DchairsDataset, self).__init__()
        self.dataPath = dataPath
        self.folders = np.array(os.listdir(dataPath))
        self.transform = transform
    def __len__(self):
        return len(self.folders)
    def __getitem__(self, idx):
        idA, idB = np.random.choice(os.listdir(os.path.join(self.dataPath, self.folders[idx])),2)
        label = idx
        imgA = Image.open(os.path.join(self.dataPath, self.folders[idx], idA))
        imgB = Image.open(os.path.join(self.dataPath, self.folders[idx], idB))
        imgA = self.transform(imgA)
        imgB = self.transform(imgB)
        return imgA, imgB, torch.LongTensor(1).fill_(int(label))

#############################################################################
# Datasets                                                                  #
#############################################################################
if opt.dataset == 'celebA':
    dataset = PairCelebADataset(os.path.join(opt.dataPath, "celebA/aligned"),
                                os.path.join(opt.dataPath, "celebA/identity_celebA_train.txt"),
                                transforms.Compose([transforms.CenterCrop(128),
                                                    transforms.Resize(opt.sizeX),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))
    testset = PairCelebADataset(os.path.join(opt.dataPath, "celebA/aligned"),
                                os.path.join(opt.dataPath, "celebA/identity_celebA_val.txt"),
                                transforms.Compose([transforms.CenterCrop(128),
                                                    transforms.Resize(opt.sizeX),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))
elif opt.dataset == '3Dchairs':
    dataset = Pair3DchairsDataset(os.path.join(opt.dataPath, "rendered_chairs/train"),
                                  transforms.Compose([transforms.CenterCrop(300),
                                                      transforms.Resize(opt.sizeX),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                  ]))
    testset = Pair3DchairsDataset(os.path.join(opt.dataPath, "rendered_chairs/val"),
                                  transforms.Compose([transforms.CenterCrop(300),
                                                      transforms.Resize(opt.sizeX),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                  ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))

#############################################################################
# weights init                                                              #
#############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        if m.weight:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

#############################################################################
# Modules                                                                   #
#############################################################################
class _encoder(nn.Module):
    def __init__(self, nc, zSize, sSize, nf, xSize):
        super(_encoder, self).__init__()
        self.mods = nn.Sequential(nn.Conv2d(nc, nf, 3, 1, 1),
                                  nn.BatchNorm2d(nf, 0.1, affine=False),
                                  nn.ReLU(),
                                  nn.Conv2d(nf, nf, 3, 1, 1),
                                  nn.BatchNorm2d(nf, 0.1, affine=False),
                                  nn.ReLU(),
                                  nn.Conv2d(nf, 2*nf, 2, 2),
                                  nn.BatchNorm2d(2*nf, 0.1, affine=False),
                                  nn.ReLU(),
                                  nn.Conv2d(2*nf, 2*nf, 3, 1, 1),
                                  nn.BatchNorm2d(2*nf, 0.1, affine=False),
                                  nn.ReLU(),
                                  nn.Conv2d(2*nf, 4*nf, 2, 2),
                                  nn.BatchNorm2d(4*nf, 0.1, affine=False),
                                  nn.ReLU(),
                                  nn.Conv2d(4*nf, 4*nf, 3, 1, 1),
                                  nn.BatchNorm2d(4*nf, 0.1, affine=False),
                                  nn.ReLU(),
                                  nn.Conv2d(4*nf, 8*nf, 2, 2),
                                  nn.BatchNorm2d(8*nf, 0.1, affine=False),
                                  nn.ReLU(),
                                  nn.Conv2d(8*nf, 8*nf, 3, 1, 1),
                                  nn.BatchNorm2d(8*nf, 0.1, affine=False),
                                  nn.ReLU(),)
        npix = nf * 8 * (xSize//8) * (xSize//8)
        self.modsZ = nn.Linear(npix, zSize*2)
        self.modsS = nn.Linear(npix, sSize)
    def forward(self, x):
        x = self.mods(x)
        x = x.view(x.size(0), -1)
        z = self.modsZ(x)
        s = self.modsS(x)
        z = z.view(z.size(0), 2, -1)
        return z, s

class _decoder(nn.Module):
    def __init__(self, nc, zSize, sSize, nf, xSize):
        super(_decoder, self).__init__()
        npix = nf * 8 * 4 * 4
        self.modsZ = nn.Linear(zSize, npix)
        self.modsS = nn.Linear(sSize, npix)
        self.mods = nn.Sequential(nn.Conv2d(nf*8, nf*8, 3, 1, 1),
                                  nn.BatchNorm2d(nf*8, 0.1, affine=False),
                                  nn.ReLU(),
                                  nn.ConvTranspose2d(nf*8, nf*4, 2, 2),
                                  nn.BatchNorm2d(nf*4, 0.1, affine=False),
                                  nn.ReLU(),
                                  nn.Conv2d(nf*4, nf*4, 3, 1, 1),
                                  nn.BatchNorm2d(nf*4, 0.1, affine=False),
                                  nn.ReLU(),
                                  nn.ConvTranspose2d(nf*4, nf*2, 2, 2),
                                  nn.BatchNorm2d(nf*2, 0.1, affine=False),
                                  nn.ReLU(),
                                  nn.Conv2d(nf*2, nf*2, 3, 1, 1),
                                  nn.BatchNorm2d(nf*2, 0.1, affine=False),
                                  nn.ReLU(),
                                  nn.ConvTranspose2d(nf*2, nf, 2, 2),
                                  nn.BatchNorm2d(nf, 0.1, affine=False),
                                  nn.ReLU(),
                                  nn.Conv2d(nf, nf, 3, 1, 1),
                                  nn.BatchNorm2d(nf, 0.1, affine=False),
                                  nn.ReLU(),
                                  nn.Conv2d(nf, nc, 3, 1, 1))
    def forward(self, z, s):
        z = self.modsZ(z)
        s = self.modsS(s)
        x = z + s
        x = x.view(x.size(0), -1, 4, 4)
        x = self.mods(x)
        return F.tanh(x)

class _discriminator(nn.Module):
    def __init__(self, nc, nf, xSize, nClass):
        super(_discriminator, self).__init__()
        self.xSize = xSize
        self.embeddings = nn.ModuleList([nn.Embedding(nClass, nf*1*(xSize//8)*(xSize//8)),
                                         nn.Embedding(nClass, nf*2*(xSize//8)*(xSize//8)),
                                         nn.Embedding(nClass, nf*4*(xSize//8)*(xSize//8))])
        self.mods = nn.ModuleList([nn.Sequential(nn.Conv2d(nc, nf, 3, 1, 1),
                                                 nn.LeakyReLU(.2),
                                                 nn.Conv2d(nf, nf, 2, 2),),
                                   nn.Sequential(nn.BatchNorm2d(nf),
                                                 nn.LeakyReLU(.2),
                                                 nn.Conv2d(nf, nf*2, 2, 2),
                                                 nn.BatchNorm2d(nf*2),
                                                 nn.LeakyReLU(.2),
                                                 nn.Conv2d(nf*2, nf*2, 3, 1, 1),
                                                 nn.BatchNorm2d(nf*2),
                                                 nn.LeakyReLU(.2),),
                                   nn.Sequential(nn.BatchNorm2d(nf*2),
                                                 nn.LeakyReLU(.2),
                                                 nn.Conv2d(nf*2, nf*4, 2, 2),
                                                 nn.BatchNorm2d(nf*4),
                                                 nn.LeakyReLU(.2),
                                                 nn.Conv2d(nf*4, nf*4, 3, 1, 1),),
                                   nn.Sequential(nn.BatchNorm2d(nf*4),
                                                 nn.LeakyReLU(.2),
                                                 nn.Conv2d(nf*4, nf*4, 3, 1, 1),
                                                 nn.BatchNorm2d(nf*4),
                                                 nn.LeakyReLU(.2),),
                                   nn.Linear(nf*4*(xSize//8)*(xSize//8), 1),
                                  ]) 
    def forward(self, x, sid):
        x = self.mods[0](x)
        s0 = self.embeddings[0](sid)
        s0 = s0.view(x.size(0), x.size(1), self.xSize//8, self.xSize//8)
        s0 = nn.functional.upsample(s0, scale_factor=4, mode='nearest')
        x = self.mods[1](x + s0)
        s1 = self.embeddings[1](sid)
        s1 = s1.view(x.size(0), x.size(1), self.xSize//8, self.xSize//8)
        s1 = nn.functional.upsample(s1, scale_factor=2, mode='nearest')
        x = self.mods[2](x + s1)
        s2 = self.embeddings[2](sid)
        s2 = s2.view(x.size(0), x.size(1), self.xSize//8, self.xSize//8)
        x = self.mods[3](x + s2)
        x = x.view(x.size(0), -1)
        x = F.dropout(x)
        x = self.mods[4](x)
        return x

#############################################################################
# Modules - DC                                                              #
#############################################################################
class _dcencoder(nn.Module):
    def __init__(self, nc, zSize, sSize, nf, xSize):
        super(_dcencoder, self).__init__()
        self.mods = nn.Sequential(nn.Conv2d(nc, nf, 4, 2, 1),
                                  nn.BatchNorm2d(nf, 0.1, affine=False),
                                  nn.ReLU(),
                                  nn.Conv2d(nf, 2*nf, 4, 2, 1),
                                  nn.BatchNorm2d(2*nf, 0.1, affine=False),
                                  nn.ReLU(),
                                  nn.Conv2d(2*nf, 4*nf, 4, 2, 1),
                                  nn.BatchNorm2d(4*nf, 0.1, affine=False),
                                  nn.ReLU(),
                                  nn.Conv2d(4*nf, 8*nf, 4, 2, 1),
                                  nn.BatchNorm2d(8*nf, 0.1, affine=False),
                                  nn.ReLU(),)
        npix = nf * 8 * (xSize//16) * (xSize//16)
        self.modsZ = nn.Linear(npix, zSize*2)
        self.modsS = nn.Linear(npix, sSize)
    def forward(self, x):
        x = self.mods(x)
        x = x.view(x.size(0), -1)
        z = self.modsZ(x)
        s = self.modsS(x)
        z = z.view(z.size(0), 2, -1)
        return z, s

class _dcdecoder(nn.Module):
    def __init__(self, nc, zSize, sSize, nf, xSize):
        super(_dcdecoder, self).__init__()
        npix = nf * 8 * (xSize//16) * (xSize//16)
        self.modsZ = nn.Linear(zSize, npix)
        self.modsS = nn.Linear(sSize, npix)
        self.mods = nn.Sequential(nn.BatchNorm2d(nf*8, 0.1, affine=False),
                                  nn.ReLU(),
                                  nn.ConvTranspose2d(nf*8, nf*4, 4, 2, 1),
                                  nn.BatchNorm2d(nf*4, 0.1, affine=False),
                                  nn.ReLU(),
                                  nn.ConvTranspose2d(nf*4, nf*2, 4, 2, 1),
                                  nn.BatchNorm2d(nf*2, 0.1, affine=False),
                                  nn.ReLU(),
                                  nn.ConvTranspose2d(nf*2, nf, 4, 2, 1),
                                  nn.BatchNorm2d(nf, 0.1, affine=False),
                                  nn.ReLU(),
                                  nn.ConvTranspose2d(nf, nc, 4, 2, 1))
    def forward(self, z, s):
        z = self.modsZ(z)
        s = self.modsS(s)
        x = z + s
        x = x.view(x.size(0), -1, 4, 4)
        x = self.mods(x)
        return F.tanh(x)            

class _dcdiscriminator(nn.Module):
    def __init__(self, nc, nf, xSize, nClass):
        super(_dcdiscriminator, self).__init__()
        self.xSize = xSize
        self.embeddings = nn.ModuleList([nn.Embedding(nClass, nf*1*(xSize//8)*(xSize//8)),
                                         nn.Embedding(nClass, nf*1*(xSize//8)*(xSize//8)),
                                         nn.Embedding(nClass, nf*2*(xSize//8)*(xSize//8)),
                                         nn.Embedding(nClass, nf*4*(xSize//8)*(xSize//8))])
        self.mods = nn.ModuleList([nn.Sequential(nn.Conv2d(nc, nf, 3, 1, 1),
                                                 nn.LeakyReLU(.2)),
                                   nn.Sequential(nn.BatchNorm2d(nf),
                                                 nn.LeakyReLU(.2),
                                                 nn.Conv2d(nf, nf, 4, 2, 1)),
                                   nn.Sequential(nn.BatchNorm2d(nf),
                                                 nn.LeakyReLU(.2),
                                                 nn.Conv2d(nf, nf*2, 4, 2, 1)),
                                   nn.Sequential(nn.BatchNorm2d(nf*2),
                                                 nn.LeakyReLU(.2),
                                                 nn.Conv2d(nf*2, nf*4, 4, 2, 1)),
                                   nn.Sequential(nn.BatchNorm2d(nf*4),
                                                 nn.LeakyReLU(.2),
                                                 nn.Conv2d(nf*4, nf*8, 4, 2, 1)),
                                   nn.Linear(nf*8*(xSize//16)*(xSize//16), 1),
                                  ]) 
    def forward(self, x, sid):
        x = self.mods[0](x)
        s0 = self.embeddings[0](sid)
        s0 = s0.view(x.size(0), x.size(1), self.xSize//8, self.xSize//8)
        s0 = nn.functional.upsample(s0, scale_factor=8, mode='nearest')
        x = self.mods[1](x + s0)
        s1 = self.embeddings[1](sid)
        s1 = s1.view(x.size(0), x.size(1), self.xSize//8, self.xSize//8)
        s1 = nn.functional.upsample(s1, scale_factor=4, mode='nearest')
        x = self.mods[2](x + s1)
        s2 = self.embeddings[2](sid)
        s2 = s2.view(x.size(0), x.size(1), self.xSize//8, self.xSize//8)
        s2 = nn.functional.upsample(s2, scale_factor=2, mode='nearest')
        x = self.mods[3](x + s2)
        s3 = self.embeddings[3](sid)
        s3 = s3.view(x.size(0), x.size(1), self.xSize//8, self.xSize//8)
        x = self.mods[4](x + s3)
        x = x.view(x.size(0), -1)
        x = F.dropout(x)
        x = self.mods[5](x)
        return x
    
def UnitGaussianKLDLoss(z):
    return (.5 * (- z[:,1] + (z[:,1]).exp() + (z[:,0]*z[:,0]) - 1)).mean()

lossD = nn.BCEWithLogitsLoss()
lossL = nn.MSELoss()
lossKL = UnitGaussianKLDLoss

netEnc = _dcencoder(opt.nc, opt.sizeZ, opt.sizeS, opt.nf, opt.sizeX)
netDec = _dcdecoder(opt.nc, opt.sizeZ, opt.sizeS, opt.nf, opt.sizeX)
netD = _dcdiscriminator(opt.nc, opt.nf, opt.sizeX, opt.nClass)

#############################################################################
# Placeholders                                                              #
#############################################################################
x1 = torch.FloatTensor()
x2 = torch.FloatTensor()
x3 = torch.FloatTensor()
ids = torch.LongTensor()
eps = torch.FloatTensor()
zero = torch.FloatTensor(1,1).fill_(0)
labelPos = torch.FloatTensor(1,1).fill_(.9)
labelNeg = torch.FloatTensor(1,1).fill_(.1)

#############################################################################
# Test data                                                                 #
#############################################################################
batch_test = 5
views = 5
steps = 5

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_test, shuffle=True, num_workers=int(opt.workers), drop_last=True)
x_test, _, _ = next(iter(testloader))

z_test = torch.FloatTensor(1, views, steps, opt.sizeZ)
z_test[:,:,0].normal_()
z_test[:,:,-1].normal_()
zstep_test = (z_test[:,:,-1] - z_test[:,:,0]) / steps
for i in range(1, steps-1):
    z_test[:,:,i] = z_test[:,:,i-1] + zstep_test
z_test = z_test.repeat(batch_test,1,1,1).view(-1, opt.sizeZ)

#############################################################################
# To Cuda                                                                   #
#############################################################################
if opt.cuda:
    netEnc.cuda()
    netDec.cuda()
    netD.cuda()
    x1 = x1.cuda()
    x2 = x2.cuda()
    x3 = x3.cuda()
    ids = ids.cuda()
    eps = eps.cuda()
    zero = zero.cuda()
    labelPos = labelPos.cuda()
    labelNeg = labelNeg.cuda()
    z_test = z_test.cuda()
    x_test = x_test.cuda()

#############################################################################
# Optimizer                                                                 #
#############################################################################
optimizerEnc = optim.Adam(netEnc.parameters(), lr=opt.lrG, betas=(0.5, 0.999))
optimizerDec = optim.Adam(netDec.parameters(), lr=opt.lrG, betas=(0.5, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(0.5, 0.999))

#############################################################################
# Train                                                                     #
#############################################################################
print("Start Training")
iteration = opt.load * len(dataloader)
epoch = opt.load

while iteration <= opt.nIteration:
    log_dNeg = []
    log_dPos = []
    log_rec = []
    log_swap = []
    log_kl = []
    for x1_cpu, x2_cpu, ids_cpu in tqdm(dataloader):
        netEnc.train()
        netDec.train()
        netD.train()
        x1.resize_(x1_cpu.size(0),x1_cpu.size(1),x1_cpu.size(2),x1_cpu.size(3)).copy_(x1_cpu)
        x2.resize_(x2_cpu.size(0),x2_cpu.size(1),x2_cpu.size(2),x2_cpu.size(3)).copy_(x2_cpu)
        x3.resize_(x2_cpu.size(0),x2_cpu.size(1),x2_cpu.size(2),x2_cpu.size(3))
        x3[:-1].copy_(x2_cpu[1:])
        x3[-1].copy_(x2_cpu[0])
        ids.resize_(ids_cpu.size(0), ids_cpu.size(1))
        ids[:-1].copy_(ids_cpu[1:])
        ids[-1].copy_(ids_cpu[0])
        pz1, s1 = netEnc(Variable(x1))
        pz2, s2 = netEnc(Variable(x2))
        pz3, s3 = netEnc(Variable(x3))
        eps.resize_as_(pz1.data[:,0]).normal_()
        z1 = pz1[:,0] + (pz1[:,1]*.5).exp() * Variable(eps)
        z2 = pz2[:,0] + (pz2[:,1]*.5).exp() * Variable(eps)
        z3 = pz3[:,0] + (pz3[:,1]*.5).exp() * Variable(eps)
        y1 = netDec(z1, s1)
        y2 = netDec(z1, s2)
        y3 = netDec(z1, s3)
        ye = netDec(Variable(eps), s3)
        err_rec = lossL(y1, Variable(x1))
        err_swap = lossL(y2, Variable(x1))
        err_kl = lossKL(pz1)
        d3 = netD(y3, Variable(ids))
        de = netD(ye, Variable(ids))
        (err_rec * opt.recW +
         err_swap * opt.swap1W +
         lossD(d3, Variable(labelNeg.expand_as(d3))) * opt.swap2W + 
         lossD(de, Variable(labelPos.expand_as(de))) * opt.swap2W +
         err_kl * opt.klzW).backward()
        netD.zero_grad()
        d3 = netD(y3.detach(), Variable(ids))
        de = netD(ye.detach(), Variable(ids))
        (lossD(d3, Variable(labelPos.expand_as(d3))) * opt.swap2W + 
         lossD(de, Variable(labelNeg.expand_as(de))) * opt.swap2W).backward()
        optimizerEnc.step()
        optimizerDec.step()
        optimizerD.step()
        netEnc.zero_grad()
        netDec.zero_grad()
        netD.zero_grad()
        log_dNeg.append(de.data.mean())
        log_dPos.append(d3.data.mean())
        log_rec.append(err_rec.data.mean())
        log_swap.append(err_swap.data.mean())
        log_kl.append(err_kl.data.mean())
        iteration += 1
    epoch = epoch+1
    print(epoch,
         np.array(log_dNeg).mean(),
         np.array(log_dPos).mean(),
         np.array(log_rec).mean(),
         np.array(log_swap).mean(),
         np.array(log_kl).mean())
    if epoch% opt.checkpointFreq == 0:
        netEnc.eval()
        netDec.eval()
        pz_test, s_test = netEnc(Variable(x_test, volatile=True))
        s_test = s_test.unsqueeze(1).repeat(1,steps*views,1).view(-1,opt.sizeS)
        y_test = netDec(Variable(z_test, volatile=True), s_test)
        vutils.save_image(y_test.data, "interpolate_%d.png" % (epoch+1), nrow=views*steps, normalize=True, range=(-1,1))
        torch.save(netEnc.state_dict(), '%s/netEnc_%d.pth' % (opt.checkpointDir, epoch))
        torch.save(netDec.state_dict(), '%s/netDec_%d.pth' % (opt.checkpointDir, epoch))
        #torch.save(netD.state_dict(), '%s/netD_%d.pth' % (opt.checkpointDir, epoch))
        


