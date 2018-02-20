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

opt.dataset = '3Dchairs'        # [ celebA | 102flowers | 3Dchairs ]
opt.dataPath = './data'


# Input space
opt.nc = 3                    # number of input channels
opt.sizeX = 64                # size of the image
opt.sizeZ = 64                # size of random noise vectors

# Convolution settings
opt.nf = 64                   # base number of filter in G and D
opt.nLayers = 4               # number of conv layers in G and D

# Hardward settings
opt.workers = 4               # workers data for preprocessing
opt.cuda = True               # use CUDA
opt.gpu = 0                   # GPU id

# Optimisation scheme
opt.batchSize = 128           # minibatch size
opt.nIteration = 1000001      # number of training iterations
opt.lrG = 5e-5                # learning rate for G
opt.lrD = 5e-5                # learning rate for D

# Save/Load networks
opt.checkpointDir = '.'       # checkpoints directory
opt.load = 0                  # if > 0, load given checkpoint
opt.checkpointFreq = 1000     # frequency of checkpoints (in number of epochs)

#############################################################################
# Loading Weights                                                           #
#############################################################################
opt.netG = ''
opt.netD = ''
opt.netE = ''
if opt.load > 0:
    opt.netG = '%s/netG_%d.pth' % (opt.checkpointDir, opt.load)
    opt.netD = '%s/netD_%d.pth' % (opt.checkpointDir, opt.load)
    opt.netE = '%s/netE_%d.pth' % (opt.checkpointDir, opt.load)

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

class Pair102flowersDataset(torch.utils.data.Dataset):
    def __init__(self, dataPath, labelFile, nc, transform=transforms.ToTensor()):
        super(Pair102flowersDataset, self).__init__()
        self.dataPath = dataPath
        self.files = np.sort(os.listdir(dataPath))
        self.labels = scipy.io.loadmat(labelFile)['labels'][0]
        self.transform = transform
        self.nc = nc
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        label = self.labels[idx]
        fileA = self.files[idx]
        fileB = np.random.choice(self.files[self.labels == label])
        imgA = Image.open(os.path.join(self.dataPath, fileA))
        imgB = Image.open(os.path.join(self.dataPath, fileB))
        imgA = self.transform(imgA)
        imgB = self.transform(imgB)
        if imgA.size(0) == 1:
            imgA = imgA.repeat(self.nc,1,1)
        if imgB.size(0) == 1:
            imgB = imgB.repeat(self.nc,1,1)
        return imgA[:self.nc], imgB[:self.nc], torch.LongTensor(1).fill_(int(label))

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
    
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

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
class _dcEncoder(nn.Module):
    def __init__(self, nIn=3, nOut=1024, nf=64, nLayer=4, sizeX=64):
        super(_dcEncoder, self).__init__()
        self.mods = nn.Sequential()
        sizeX = sizeX //2
        self.mods.add_module("Conv0_%dx%dx%d" % (nf, sizeX, sizeX), nn.Conv2d(nIn, nf, 4, 2, 1, bias=False))
        self.mods.add_module("BN0", nn.BatchNorm2d(nf))
        self.mods.add_module("ReLU0", nn.ReLU(True))
        for i in range(1,nLayer):
            sizeX = sizeX //2
            self.mods.add_module("Conv%d_%dx%dx%d" % (i, nf*2, sizeX, sizeX), nn.Conv2d(nf, nf*2, 4, 2, 1, bias=False))
            self.mods.add_module("BN%d"% i, nn.BatchNorm2d(nf*2))
            self.mods.add_module("ReLU%d" % i, nn.ReLU(True))
            nf = nf * 2
        self.mods.add_module("FC_%dx1x1" % nOut, nn.Conv2d(nf, nOut, sizeX, bias=False))
        weights_init(self.mods)
    def forward(self, x):
        return self.mods(x)

class _dcDecoder(nn.Module):
    def __init__(self, nIn=1024, nOut=3, nf=512, nLayer=4, sizeX=64):
        super(_dcDecoder, self).__init__()
        sizeX = sizeX // (2**nLayer)
        nf = nf * (2 ** (nLayer - 1))
        self.mods = nn.Sequential()
        self.mods.add_module("FC_%dx%dx%d" % (nf,sizeX,sizeX), nn.ConvTranspose2d(nIn, nf, sizeX, bias=False))
        self.mods.add_module("BN0", nn.BatchNorm2d(nf))
        self.mods.add_module("ReLU0", nn.ReLU(True))
        for i in range(1,nLayer):
            sizeX = sizeX * 2
            self.mods.add_module("ConvTr%d_%dx%dx%d" % (i, nf//2, sizeX, sizeX), nn.ConvTranspose2d(nf, nf//2, 4, 2, 1, bias=False))
            self.mods.add_module("BN%d"% i, nn.BatchNorm2d(nf//2))
            self.mods.add_module("ReLU%d" % i, nn.ReLU(True))
            nf = nf // 2
        self.mods.add_module("ConvTrO_%dx%dx%d" % (nf, sizeX, sizeX), nn.ConvTranspose2d(nf, nOut, 4, 2, 1, bias=False))
        weights_init(self.mods)
    def forward(self, x):
        return self.mods(x)

class _dcDiscriminator(nn.Module):
    def __init__(self, nIn=3, nOut=1024, nf=64, nLayer=4, sizeX=64):
        super(_dcDiscriminator, self).__init__()
        self.mods = nn.Sequential()
        sizeX = sizeX //2
        self.mods.add_module("Conv0_%dx%dx%d" % (nf, sizeX, sizeX), nn.Conv2d(nIn, nf, 4, 2, 1, bias=False))
        self.mods.add_module("LReLU0", nn.LeakyReLU(0.2))
        for i in range(1,nLayer):
            sizeX = sizeX //2
            self.mods.add_module("Conv%d_%dx%dx%d" % (i, nf*2, sizeX, sizeX), nn.Conv2d(nf, nf*2, 4, 2, 1, bias=False))
            self.mods.add_module("BN%d"% i, nn.BatchNorm2d(nf*2))
            self.mods.add_module("LReLU%d" % i, nn.LeakyReLU(0.2))
            nf = nf * 2
        self.mods.add_module("FC_%dx1x1" % nOut, nn.Conv2d(nf, nOut, sizeX, bias=False))
        weights_init(self.mods)
    def forward(self, x):
        return self.mods(x)

netG = _dcDecoder(nIn=opt.sizeZ*2, nOut=opt.nc, nf=opt.nf, nLayer=opt.nLayers, sizeX=opt.sizeX)
netD = _dcDiscriminator(nIn=opt.nc*2, nOut=1, nf=opt.nf, nLayer=opt.nLayers, sizeX=opt.sizeX)
netE = _dcEncoder(nIn=opt.nc, nOut=opt.sizeZ, nf=opt.nf, nLayer=opt.nLayers, sizeX=opt.sizeX)

if  opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
if opt.netE != '':
    netE.load_state_dict(torch.load(opt.netE))

print(netG)
print(netD)
print(netE)

discriminationLoss = nn.BCEWithLogitsLoss()

#############################################################################
# Placeholders                                                              #
#############################################################################
x1_real = torch.FloatTensor(opt.batchSize, opt.nc, opt.sizeX, opt.sizeX)
x2_real = torch.FloatTensor(opt.batchSize, opt.nc, opt.sizeX, opt.sizeX)
zView1 = torch.FloatTensor(opt.batchSize, opt.sizeZ, 1, 1).normal_()
zView2 = torch.FloatTensor(opt.batchSize, opt.sizeZ, 1, 1).normal_()
labelPos = torch.FloatTensor(opt.batchSize)
labelNeg = torch.FloatTensor(opt.batchSize)

#############################################################################
# Test data                                                                 #
#############################################################################
batch_test = 16
nViews = 10

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_test, shuffle=True, num_workers=int(opt.workers))

x1_test, x2_test, _ = next(iter(testloader))
zView_test = torch.FloatTensor(nViews, 1, opt.sizeZ, 1, 1).normal_().repeat(1, batch_test, 1, 1, 1)

#############################################################################
# To Cuda                                                                   #
#############################################################################
if opt.cuda:
    print("Convert to Cuda")
    torch.cuda.set_device(opt.gpu)
    netG.cuda()
    netD.cuda()
    netE.cuda()
    discriminationLoss.cuda()
    x1_real = x1_real.cuda()
    x2_real = x2_real.cuda()
    zView1 = zView1.cuda()
    zView2 = zView2.cuda()
    labelPos = labelPos.cuda()
    labelNeg = labelNeg.cuda()
    zView_test = zView_test.cuda()
    x1_test = x1_test.cuda()
    x2_test = x2_test.cuda()

#############################################################################
# Optimizer                                                                 #
#############################################################################
optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(0.5, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(0.5, 0.999))
optimizerE = optim.Adam(netE.parameters(), lr=opt.lrG, betas=(0.5, 0.999))

#############################################################################
# Train                                                                     #
#############################################################################
print("Start Training")
iteration = opt.load * len(dataloader)
epoch = opt.load

while iteration <= opt.nIteration:
    log_dNeg = []
    log_dPos = []
    for x1_cpu, x2_cpu, _ in tqdm(dataloader):
        #######################
        # Init iteration      #
        #######################
        netG.train()
        netD.train()
        netE.train()
        x1_real.resize_(x1_cpu.size(0), x1_cpu.size(1), x1_cpu.size(2), x1_cpu.size(3)).copy_(x1_cpu)
        x2_real.resize_(x2_cpu.size(0), x2_cpu.size(1), x2_cpu.size(2), x2_cpu.size(3)).copy_(x2_cpu)
        zView1.resize_(x1_cpu.size(0), opt.sizeZ, 1, 1).normal_()
        zView2.resize_(x1_cpu.size(0), opt.sizeZ, 1, 1).normal_()
        labelPos.resize_(x1_cpu.size(0), 1, 1, 1).fill_(.9)
        labelNeg.resize_(x1_cpu.size(0), 1, 1, 1).fill_(.1)
        #######################
        # Train               #
        #######################
        # Generation Objective
        netG.zero_grad()
        netE.zero_grad()
        zContent = netE(Variable(x1_real))
        if (iteration % 3) == 0:
            x1_generated = Variable(x1_real)
            x2_generated = F.tanh(netG(torch.cat((zContent, Variable(zView2)),1)))
        elif (iteration % 3) == 1:
            x1_generated = F.tanh(netG(torch.cat((zContent, Variable(zView1)),1)))
            x2_generated = F.tanh(netG(torch.cat((zContent, Variable(zView2)),1)))
        elif (iteration % 3) == 2:
            x1_generated = F.tanh(netG(torch.cat((zContent, Variable(zView1)),1)))
            x2_generated = Variable(x1_real)
        dGen = netD(torch.cat((x1_generated, x2_generated),1))
        generationObjective = discriminationLoss(dGen, Variable(labelPos))
        generationObjective.backward()
        # Discriminator objective
        netD.zero_grad()
        dPos = netD(Variable(torch.cat((x1_real,x2_real),1)))
        dNeg = netD(torch.cat((x1_generated.detach(), x2_generated.detach()),1))
        discriminationObjective = discriminationLoss(dNeg, Variable(labelNeg)) + discriminationLoss(dPos, Variable(labelPos))
        discriminationObjective.backward()
        # Update weights
        optimizerG.step()
        optimizerE.step()
        optimizerD.step()
        # Logs
        dPos = dPos.detach()
        dNeg = dNeg.detach()
        dPos.volatile = True
        dNeg.volatile = True
        log_dPos.append(F.sigmoid(dPos).data.mean())
        log_dNeg.append(F.sigmoid(dNeg).data.mean())
        iteration += 1
    epoch = epoch+1
    print(epoch,
          np.array(log_dPos).mean(),
          np.array(log_dNeg).mean(),
          )
    with open('logs.dat', 'ab') as f:
        np.savetxt(f, np.vstack((np.array(log_dPos),
                                 np.array(log_dNeg),
                                 )).T)
    if epoch % opt.checkpointFreq == 0:
        netG.eval()
        netE.eval()
        zContent_test = netE(Variable(x1_test,volatile=True))
        out = torch.FloatTensor(x1_test.size(0), nViews+3, opt.nc, opt.sizeX, opt.sizeX).cuda().zero_()
        out[:,0] = x1_test
        out[:,1] = x2_test
        for t in range(nViews):
            out[:,t+3] = F.tanh(netG(torch.cat((zContent_test, Variable(zView_test[t], volatile=True)),1))).data
        vutils.save_image(out.view(-1,opt.nc,opt.sizeX,opt.sizeX), 'out_%d.png' % (epoch+1), nrow=(nViews+3), normalize=True)
        #torch.save(netG.state_dict(), '%s/netG_%d.pth' % (opt.checkpointDir, epoch))
        #torch.save(netD.state_dict(), '%s/netD_%d.pth' % (opt.checkpointDir, epoch))
        #torch.save(netE.state_dict(), '%s/netE_%d.pth' % (opt.checkpointDir, epoch))

