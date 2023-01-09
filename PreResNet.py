import torch
from scipy.stats import geom
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

class NestedDropout(nn.Module):
        def __init__(self, p,group):
                super(NestedDropout, self).__init__()
                self.GD=torch.distributions.geometric.Geometric(torch.tensor([p]))
                self.group=group

                feat_dim=512
                dis_weight=[]
                for i in range(1,int(feat_dim/self.group)+1):
                    dis_weight=dis_weight+[geom.pmf(i, p)]*self.group
                    
                dis_weight=dis_weight+[1-geom.cdf(int(feat_dim/self.group), p)]*(feat_dim%self.group)
                self.dis_weight=torch.FloatTensor(dis_weight).view(1,-1).cuda(1)

        def forward(self, X,):
                if self.training:
                        instances = self.GD.sample([X.shape[0],])*self.group
                        instances = torch.clamp(instances, min=self.group, max=X.shape[1]).long()
                        #scale= X.shape[1]/instances.cuda(X.get_device())
                        mask = torch.zeros(X.shape).cuda(X.get_device())
                        for i,instance in enumerate(instances):
                            mask[i,:instance]=1
                        #return X *mask * scale
                        return X *mask
                else:
                        #instances = torch.ones([X.shape[0],]).long()*self.group
                        #instances = self.GD.sample([X.shape[0],])*self.group
                        #instances = torch.clamp(instances, min=self.group, max=X.shape[1]).long()
                        #scale= X.shape[1]/instances.cuda(X.get_device())
                        #mask = torch.zeros(X.shape).cuda(X.get_device())
                        #for i,instance in enumerate(instances):
                        #    mask[i,:instance]=1
                        #return X *mask * scale
                        #return X*mask
                        #print(X[0].tolist())
                        return X *self.dis_weight
                #return X


def conv3x3(in_planes, out_planes, stride=1):
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, in_planes, planes, stride=1):
                super(BasicBlock, self).__init__()
                self.conv1 = conv3x3(in_planes, planes, stride)
                self.bn1 = nn.BatchNorm2d(planes)
                self.conv2 = conv3x3(planes, planes)
                self.bn2 = nn.BatchNorm2d(planes)

                self.shortcut = nn.Sequential()
                if stride != 1 or in_planes != self.expansion*planes:
                        self.shortcut = nn.Sequential(
                                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                                nn.BatchNorm2d(self.expansion*planes)
                        )

        def forward(self, x):
                out = F.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                out += self.shortcut(x)
                out = F.relu(out)
                return out


class PreActBlock(nn.Module):
        '''Pre-activation version of the BasicBlock.'''
        expansion = 1

        def __init__(self, in_planes, planes, stride=1):
                super(PreActBlock, self).__init__()
                self.bn1 = nn.BatchNorm2d(in_planes)
                self.conv1 = conv3x3(in_planes, planes, stride)
                self.bn2 = nn.BatchNorm2d(planes)
                self.conv2 = conv3x3(planes, planes)

                self.shortcut = nn.Sequential()
                if stride != 1 or in_planes != self.expansion*planes:
                        self.shortcut = nn.Sequential(
                                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
                        )

        def forward(self, x):
                out = F.relu(self.bn1(x))
                shortcut = self.shortcut(out)
                out = self.conv1(out)
                out = self.conv2(F.relu(self.bn2(out)))
                out += shortcut
                return out


class Bottleneck(nn.Module):
        expansion = 4

        def __init__(self, in_planes, planes, stride=1):
                super(Bottleneck, self).__init__()
                self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
                self.bn1 = nn.BatchNorm2d(planes)
                self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
                self.bn2 = nn.BatchNorm2d(planes)
                self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
                self.bn3 = nn.BatchNorm2d(self.expansion*planes)

                self.shortcut = nn.Sequential()
                if stride != 1 or in_planes != self.expansion*planes:
                        self.shortcut = nn.Sequential(
                                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                                nn.BatchNorm2d(self.expansion*planes)
                        )

        def forward(self, x):
                out = F.relu(self.bn1(self.conv1(x)))
                out = F.relu(self.bn2(self.conv2(out)))
                out = self.bn3(self.conv3(out))
                out += self.shortcut(x)
                out = F.relu(out)
                return out


class PreActBottleneck(nn.Module):
        '''Pre-activation version of the original Bottleneck module.'''
        expansion = 4

        def __init__(self, in_planes, planes, stride=1):
                super(PreActBottleneck, self).__init__()
                self.bn1 = nn.BatchNorm2d(in_planes)
                self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
                self.bn2 = nn.BatchNorm2d(planes)
                self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
                self.bn3 = nn.BatchNorm2d(planes)
                self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

                self.shortcut = nn.Sequential()
                if stride != 1 or in_planes != self.expansion*planes:
                        self.shortcut = nn.Sequential(
                                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
                        )

        def forward(self, x):
                out = F.relu(self.bn1(x))
                shortcut = self.shortcut(out)
                out = self.conv1(out)
                out = self.conv2(F.relu(self.bn2(out)))
                out = self.conv3(F.relu(self.bn3(out)))
                out += shortcut
                return out


class ResNet(nn.Module):
        def __init__(self, block, num_blocks,p,group ,num_classes=10):
                super(ResNet, self).__init__()
                self.in_planes = 64

                self.conv1 = conv3x3(3,64)
                self.bn1 = nn.BatchNorm2d(64)
                self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
                self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
                self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
                self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
                #self.nested_dropout=NestedDropout(p=p,group=group)
                self.linear = nn.Linear(512*block.expansion, num_classes,bias=False)
                #self.linear_z = nn.Linear(512*block.expansion, 128)

        def _make_layer(self, block, planes, num_blocks, stride):
                strides = [stride] + [1]*(num_blocks-1)
                layers = []
                for stride in strides:
                        layers.append(block(self.in_planes, planes, stride))
                        self.in_planes = planes * block.expansion
                return nn.Sequential(*layers)

        def forward(self, x, lin=0, lout=5,feat=False):
                out = x
                if lin < 1 and lout > -1:
                        out = self.conv1(out)
                        out = self.bn1(out)
                        out = F.relu(out)
                if lin < 2 and lout > 0:
                        out = self.layer1(out)
                if lin < 3 and lout > 1:
                        out = self.layer2(out)
                if lin < 4 and lout > 2:
                        out = self.layer3(out)
                if lin < 5 and lout > 3:
                        out = self.layer4(out)
                if lout > 4:
                        out = F.avg_pool2d(out, 4)
                        out = out.view(out.size(0), -1)
                        if feat:
                                f=out
                                #f=self.linear_z(f)
                                out = self.linear(out)
                                return f,out
                        else:
                                #out = self.nested_dropout(out)
                                out = self.linear(out)
                                return out


def ResNet18(num_classes=10,p=0.33,group=50):
        return ResNet(PreActBlock, [2,2,2,2],p=p,group=group, num_classes=num_classes)

def ResNet34(num_classes=10):
        return ResNet(BasicBlock, [3,4,6,3], num_classes=num_classes)

def ResNet50(num_classes=10):
        return ResNet(Bottleneck, [3,4,6,3], num_classes=num_classes)

def ResNet101(num_classes=10):
        return ResNet(Bottleneck, [3,4,23,3], num_classes=num_classes)

def ResNet152(num_classes=10):
        return ResNet(Bottleneck, [3,8,36,3], num_classes=num_classes)


if __name__=='__main__':

        print([(i,geom.pmf(i, 0.33)) for i in range(15)])
        exit()
        p=0.33
        GD=torch.distributions.geometric.Geometric(torch.tensor([p]))
        instances = GD.sample([100,])
        print(instances)
        exit()

        net = ResNet18()
        #y = net(Variable(torch.randn(1,3,32,32)))
        for k,v in net.named_parameters():
                print(k,v.dim())
        for buffer_train in net.buffers():
                print(buffer_train)
