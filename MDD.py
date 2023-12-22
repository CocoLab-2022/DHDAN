import torch.nn as nn
import backbone as backbone
from modules.grl import WarmStartGradientReverseLayer
import torch.nn.functional as F
import torch
import numpy as np
import random
import resnet_model
from vit_modeling import VisionTransformer, CONFIGS

from torch.nn import init

import torch
import torch.nn as nn
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
class MDDNet(nn.Module):
    def __init__(self, base_net='ResNet50', use_bottleneck=True, bottleneck_dim=1024, width=1024, class_num=31):
        super(MDDNet, self).__init__()
        ## set base network
        # self.base_network = backbone.network_dict[base_net]()

        config_vit = CONFIGS['R50-ViT-B_16']
        config_vit.n_classes = 100
        config_vit.n_skip = 3
        config_vit.patches.grid = (int(224 / 16), int(224 / 16))
        self.use_bottleneck = use_bottleneck
        self.feature_extractor = VisionTransformer(config_vit, img_size=[224, 224], num_classes=config_vit.n_classes)
        self.feature_extractor.load_from(weights=np.load(config_vit.pretrained_path))


        self.base_network = resnet_model.resnet50(pretrained=True,model_root='resnet50.pth')
        self.proj0 = nn.Conv2d(2048, 8192, kernel_size=1, stride=1)
        self.proj1 = nn.Conv2d(2048, 8192, kernel_size=1, stride=1)
        self.proj2 = nn.Conv2d(2048, 8192, kernel_size=1, stride=1)
        self.proj3 = nn.Conv2d(2048, 8192, kernel_size=1, stride=1)

        self.softmax = nn.LogSoftmax(dim=1)
        # self.avgpool = nn.AvgPool2d(kernel_size=14)
        self.avgpool = nn.AvgPool2d(kernel_size=7)
        self.maxpool = nn.MaxPool2d(kernel_size=7)

        self.shortcut = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(2048)
        )




        self.use_bottleneck = use_bottleneck
        self.grl_layer = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=1000., auto_step=True)
        self.bottleneck_layer_list = [nn.Linear(8192*7, bottleneck_dim), nn.BatchNorm1d(bottleneck_dim), nn.ReLU(), nn.Dropout(0.5)]
        self.bottleneck_layer = nn.Sequential(*self.bottleneck_layer_list)


        self.classifier_layer_list = [nn.Linear(bottleneck_dim, width), nn.ReLU(), nn.Dropout(0.5),
                                        nn.Linear(width, class_num)]
        self.classifier_layer = nn.Sequential(*self.classifier_layer_list)
        self.classifier_layer_2_list = [nn.Linear(bottleneck_dim, width), nn.ReLU(), nn.Dropout(0.5),
                                        nn.Linear(width, class_num)]
        self.classifier_layer_2 = nn.Sequential(*self.classifier_layer_2_list)
        # self.softmax1 = nn.Softmax(dim=1)

        ## initialization
        self.bottleneck_layer[0].weight.data.normal_(0, 0.005)
        self.bottleneck_layer[0].bias.data.fill_(0.1)

        # self.atten_list  = [nn.MultiheadAttention(embed_dim=8192, num_heads=8, dropout=0.1)]
        # self.atten = nn.Sequential(*self.atten_list)
        for dep in range(2):
            self.classifier_layer_2[dep * 3].weight.data.normal_(0, 0.01)
            self.classifier_layer_2[dep * 3].bias.data.fill_(0.0)
            self.classifier_layer[dep * 3].weight.data.normal_(0, 0.01)
            self.classifier_layer[dep * 3].bias.data.fill_(0.0)


        ## collect parameters
        self.parameter_list = [{"params":self.base_network.parameters(), "lr":0.1},
                        {"params":self.feature_extractor.parameters(), "lr":0.01},
                        {"params":self.proj0.parameters(), "lr":0.1},
                        {"params":self.proj1.parameters(), "lr":0.1},
                        {"params":self.proj2.parameters(), "lr":0.1},
                        {"params":self.shortcut.parameters(), "lr":0.1},
                            {"params":self.bottleneck_layer.parameters(), "lr":1},
                               # {"params": self.bottleneck_layer1.parameters(), "lr": 1},
                        {"params":self.classifier_layer.parameters(), "lr":1},
                               {"params":self.classifier_layer_2.parameters(), "lr":1}]

    def forward(self, inputs):
        # features = self.base_network(inputs)
        feature3,feature4_0, feature4_1, feature4_2 = self.base_network(inputs)
        feature3 = self.shortcut(feature3)
        # print("feature3.shape",feature3.shape)  #48,1024,14,14
        _,feature4_3,feature4_4,feature4_5,feature4_6 = self.feature_extractor(inputs)

        feature4_0 = feature4_0 + feature3
        feature4_1 = feature4_1 + feature4_0
        feature4_2 = feature4_2 + feature4_1


        feature4_0 = self.proj0(feature4_0)
        feature4_1 = self.proj1(feature4_1)
        feature4_2 = self.proj2(feature4_2)
        # feature4_3 = self.proj2(feature4_3)
        # print("feature4_0.shape",feature4_0.shape) #48,8192,7,7

        inter1 = feature4_0 * feature4_1
        inter2 = feature4_0 * feature4_2
        inter3 = feature4_1 * feature4_2

        # inter4 = feature4_2 * feature4_3

        #default
        inter1 = self.avgpool(inter1).view(inputs.size(0), -1)
        inter2 = self.avgpool(inter2).view(inputs.size(0), -1)
        inter3 = self.avgpool(inter3).view(inputs.size(0), -1)
        # inter4 = self.avgpool(inter4).view(inputs.size(0), -1)


        result1 = torch.nn.functional.normalize(torch.sign(inter1) * torch.sqrt(torch.abs(inter1) + 1e-10))
        result2 = torch.nn.functional.normalize(torch.sign(inter2) * torch.sqrt(torch.abs(inter2) + 1e-10))
        result3 = torch.nn.functional.normalize(torch.sign(inter3) * torch.sqrt(torch.abs(inter3) + 1e-10))
        # feature4_3 = torch.nn.functional.normalize(torch.sign(feature4_3) * torch.sqrt(torch.abs(feature4_3) + 1e-10))
        # feature4_4 = torch.nn.functional.normalize(torch.sign(feature4_4) * torch.sqrt(torch.abs(feature4_4) + 1e-10))
        # feature4_5 = torch.nn.functional.normalize(torch.sign(feature4_5) * torch.sqrt(torch.abs(feature4_5) + 1e-10))
        # result4 = torch.nn.functional.normalize(torch.sign(inter4) * torch.sqrt(torch.abs(inter3) + 1e-10))

        # print("result1.shape",result1.shape)

        # features = torch.stack([result1, result2, result3,feature4_3,feature4_4], dim=0)
        # features,_ = self.atten(features, features, features)
        # features = features.view(inputs.size(0), -1)

        features = torch.cat((result1, result2, result3,feature4_3,feature4_4,feature4_5,feature4_6), 1)
        if self.use_bottleneck:
            features = self.bottleneck_layer(features)
            # features1 = self.bottleneck_layer1(features)
            # features = torch.cat((features0, features1), 1)
        features_adv = self.grl_layer(features)
        outputs_adv = self.classifier_layer_2(features_adv)
        outputs = self.classifier_layer(features)
        softmax_outputs = self.softmax(outputs)

        return features, outputs, softmax_outputs, outputs_adv

class MDD(object):
    def __init__(self, base_net='ResNet50', width=1024, class_num=31, use_bottleneck=True, use_gpu=True, srcweight=3):
        self.c_net = MDDNet(base_net, use_bottleneck, width, width, class_num)
        self.use_gpu = use_gpu
        self.is_train = False
        self.iter_num = 0
        self.class_num = class_num
        self.focal_loss = FocalLoss(gamma=0.5)
        if self.use_gpu:
            self.c_net = self.c_net.cuda()
        self.srcweight = srcweight

    def get_loss(self, inputs, labels_source):
        class_criterion = nn.CrossEntropyLoss()
        features, outputs, softmax_outputs, outputs_adv = self.c_net(inputs)
        # classifier_loss = class_criterion(outputs.narrow(0, 0, labels_source.size(0)), labels_source)
        classifier_loss = self.focal_loss(outputs.narrow(0, 0, labels_source.size(0)), labels_source)

        outputs_source = softmax_outputs.narrow(0, 0, labels_source.size(0))

        con_loss = self.contrastive_loss(outputs_source, labels_source)

        target_adv = outputs.max(1)[1]
        target_adv_src = target_adv.narrow(0, 0, labels_source.size(0))
        target_adv_tgt = target_adv.narrow(0, labels_source.size(0), inputs.size(0) - labels_source.size(0))

        classifier_loss_adv_src = class_criterion(outputs_adv.narrow(0, 0, labels_source.size(0)), target_adv_src)

        logloss_tgt = torch.log(torch.clamp(1 - F.softmax(outputs_adv.narrow(0, labels_source.size(0), inputs.size(0) - labels_source.size(0)), dim = 1), min=1e-15)) #add small value to avoid the log value expansion

        classifier_loss_adv_tgt = F.nll_loss(logloss_tgt, target_adv_tgt)

        transfer_loss = self.srcweight * classifier_loss_adv_src + classifier_loss_adv_tgt

        outputs_target = outputs.narrow(0, labels_source.size(0), inputs.size(0) - labels_source.size(0))
        #en_loss = entropy(outputs_target)
        self.iter_num += 1
        # total_loss = classifier_loss + transfer_loss +  0.001 * con_loss #+ 0.1*en_loss
        total_loss = classifier_loss + transfer_loss #+ 0.1*en_loss
        # total_loss = classifier_loss + transfer_loss  #+ 0.1*en_loss
        #print(classifier_loss.data, transfer_loss.data, en_loss.data)
        # return [total_loss, classifier_loss, transfer_loss, classifier_loss_adv_src, classifier_loss_adv_tgt]
        return [total_loss, classifier_loss, transfer_loss, classifier_loss_adv_src, con_loss]


    def get_features(self,inputs, labels_source):
        # base_features = self.c_net.base_network(inputs)
        base_features, outputs, _, outputs_adv = self.c_net(inputs)

        # base_features = self.c_net.recon_layer(features)
        # base_features= features
        source_features = base_features.narrow(0, 0, labels_source.size(0))
        target_features = base_features.narrow(0, labels_source.size(0), inputs.size(0) - labels_source.size(0))
        return source_features, target_features

    def predict(self, inputs):
        feature, _, softmax_outputs,_= self.c_net(inputs)
        return softmax_outputs, feature

    def get_parameter_list(self):
        return self.c_net.parameter_list

    def state_dict(self):
        return self.c_net.state_dict()

    def set_train(self, mode):
        self.c_net.train(mode)
        self.is_train = mode

    def eval(self):
        self.c_net.eval()