import torch
import torch.nn as nn
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import numpy as np
import torch.nn.functional as F
from datasetsclassmutil import LOSO_DATASET
from modelclassTriV0 import AUwGCN
from torch.utils.tensorboard import SummaryWriter
from utils.train_utilsclassTriV0 import configure_optimizers
from utils.loss_func import _probability_loss, MultiCEFocalLoss_New,feat_loss_func,aMultiCEFocalLoss_New
from functools import partial
import argparse
import yaml
# fix random seed
def same_seeds(seed):
    torch.manual_seed(seed)  # fix random seed for CPU
    if torch.cuda.is_available():  # fix random seed for GPU
        torch.cuda.manual_seed(seed)  # set for current GPU
        torch.cuda.manual_seed_all(seed)  # set for all GPUs
    np.random.seed(seed)  # fix random seed for random number generation
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True  # Set True when GPU available
    torch.backends.cudnn.deterministic = True  # fix architecture

# for reproduction, same as orig. paper setting
same_seeds(1)

# keep track of statistics
class AverageMeter(object):
    def __init__(self):
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.sum += val
        self.count += n
    def avg(self):
        return self.sum/self.count
def norm(data):
    l2 = torch.norm(data, p = 2, dim = -1, keepdim = True)
    return torch.div(data, l2)

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, proto_vectors, embeded_feature):
        # 归一化原型向量
        proto_vectors = norm(proto_vectors)  # [C, 1, F]
        embeded_feature = norm(embeded_feature)  # [B, T, F]

        # 初始化损失
        loss_contra = 0

        # 计算每个嵌入特征和原型向量之间的相似度矩阵
        sim_matrix = torch.matmul(embeded_feature.unsqueeze(2), proto_vectors.permute(1, 2, 0)) / 0.1  # [B, T, C]
        sim_matrix = torch.exp(sim_matrix).reshape(sim_matrix.size(0), sim_matrix.size(1), -1).mean(dim=1)  # [B, C]

        # 计算对比损失
        for i in range(proto_vectors.size(0)):
            pos_sim = sim_matrix[:, i]  # [B]
            neg_sim = torch.cat([sim_matrix[:, j] for j in range(proto_vectors.size(0)) if j != i],
                                dim=0)  # [B * (C-1)]

            loss_contra += -torch.log(pos_sim / (pos_sim + neg_sim.sum(dim=0)))  # 计算对比损失

        loss_contra = loss_contra / proto_vectors.size(0)  # 对类别数进行平均
        loss_contra = loss_contra / embeded_feature.size(0)  # 对批次大小进行平均

        return loss_contra.mean()
def calprobility( b,micro_start_score,reg,apex):
    reg=F.softmax(reg,dim=1)
    seqleng=micro_start_score.shape[1]
    probabilities = torch.zeros(b, seqleng, device=device)
    for frame in range(seqleng):
        scores = []


        for i in range(7):
            if frame - i >= 0:  # 确保索引不越界
                score = micro_start_score[:, frame - i] *reg[:, i, frame]*apex[:,frame]
                scores.append(score)
            else:
                break

        # 将所有得分进行 Softmax
        if scores:  # 如果 scores 列表非空
            scores_tensor = torch.stack(scores, dim=1)

            # Calculate mean and standard deviation along the same dimension
            mean = scores_tensor.mean(dim=1, keepdim=True)
            if frame==0:
                std = torch.ones(mean.shape[0],1).to(device)
            else:
                std = scores_tensor.std(dim=1, keepdim=True)

            # Standardize the scores (subtract mean and divide by std)
            scores_normalized = (scores_tensor - mean) / (std + 1e-8)
            softmax_scores = F.softmax(scores_normalized, dim=1)

            for i in range(len(softmax_scores[0])):
                if frame - i >= 0:
                    probabilities[:, frame - i] += softmax_scores[:, i]/7
                else:
                    break
    return probabilities
def calprobilityend( b,micro_end_score,reg,apex):
    reg=F.softmax(reg,dim=1)
    seqleng = micro_end_score.shape[1]
    probabilities = torch.zeros(b, seqleng, device=device)
    for frame in range(seqleng):
        scores = []
        for i in range(7):
            if frame + i <seqleng:  # 确保索引不越界
                score = micro_end_score[:, frame + i] *reg[:, i, frame]*apex[:,frame]
                scores.append(score)
            else:
                break

        # 将所有得分进行 Softmax
        if scores:  # 如果 scores 列表非空

            scores_tensor = torch.stack(scores, dim=1)

            # Calculate mean and standard deviation along the same dimension
            mean = scores_tensor.mean(dim=1, keepdim=True)
            if frame == seqleng-1:
                std = torch.ones(mean.shape[0], 1).to(device)
            else:
                std = scores_tensor.std(dim=1, keepdim=True)

            # Standardize the scores (subtract mean and divide by std)
            scores_normalized = (scores_tensor - mean) / (std + 1e-8)
            softmax_scores = F.softmax(scores_normalized, dim=1)
            for i in range(len(softmax_scores[0])):
                if frame + i <seqleng:
                    probabilities[:,  frame + i] += softmax_scores[:, i]/7
                else:
                    break
    return probabilities
def train(opt, data_loader, model, optimizer, epoch, device, writer):
    model.train()
    loss_am = AverageMeter()
    
    # define loss function for binary classification
    bi_loss_apex = partial(_probability_loss, gamma=opt["abfcm_apex_gamma"], 
                           alpha=opt["abfcm_apex_alpha"], 
                           lb_smooth=opt["abfcm_label_smooth"])
    
    bi_loss_action = partial(_probability_loss, 
                             gamma=opt["abfcm_action_gamma"], 
                             alpha=opt["abfcm_action_alpha"], 
                             lb_smooth=opt["abfcm_label_smooth"])
    
    # define loss function for 3-cls classification
    _tmp_alpha = opt["abfcm_start_end_alpha"]
    cls_loss_func = MultiCEFocalLoss_New(
            class_num=3,
            alpha=torch.tensor(
                [_tmp_alpha / 2, _tmp_alpha / 2, 1 - _tmp_alpha],
                dtype=torch.float32),
            gamma=opt["abfcm_start_end_gama"],
            # lb_smooth=0.06,
        )
    class_loss_func = aMultiCEFocalLoss_New(
        class_num=5,
        alpha=torch.tensor(
            [1 - 0.98, 0.98 / 5, 0.98 / 5, 0.98 / 5,0.98 / 5],
            dtype=torch.float32),
        gamma=opt["abfcm_start_end_gama"],
        # lb_smooth=0.06,
    )
    
    for batch_idx, (feature, micro_apex_score, macro_apex_score,
                    micro_action_score, macro_action_score,
                    micro_start_end_label, macro_start_end_label
                    ,micro_class_score,macro_class_score) in enumerate(data_loader):
        
        # forward pass
        b, t, n, c = feature.shape
        feature = feature.to(device)

        micro_apex_score = micro_apex_score.to(device)
        macro_apex_score = macro_apex_score.to(device)
        micro_action_score = micro_action_score.to(device)
        macro_action_score = macro_action_score.to(device)
        micro_start_end_label = micro_start_end_label.to(device)
        macro_start_end_label = macro_start_end_label.to(device)
        micro_class_score=micro_class_score.to(device)
        macro_class_score = macro_class_score.to(device)

        STEP = int(opt["RECEPTIVE_FILED"] // 2)

        cls,reg,output_probability = model(feature)
        output_probability=output_probability.to(device)
        # emdfeatture=output_probability['embeded_feature']
        # output_probability = output_probability[:, :, STEP:-STEP]
        # cls=cls[:, :, STEP:-STEP]
        # reg=reg[:, :, STEP:-STEP]

        output_micro_apex = output_probability[:, 6, :].to(device)
        output_macro_apex = output_probability[:, 7, :].to(device)
        output_micro_action = output_probability[:, 8, :].to(device)
        output_macro_action = output_probability[:, 9, :].to(device)

        output_micro_start_end = output_probability[:, 0: 0 + 3, :].to(device)
        output_macro_start_end = output_probability[:, 3: 3 + 3, :].to(device)
        output_micro_class=cls[:, 0:5, :].to(device)
        output_macro_class = cls[:, 5:10, :].to(device)
        output_micros_reg=reg[:,0:7,:].to(device)
        output_macros_reg = reg[:, 7:14, :].to(device)
        output_microe_reg = reg[:, 14:21, :].to(device)
        output_macroe_reg = reg[:, 21:28, :].to(device)
        # calculate loss: binary classification loss
        loss_micro_apex = bi_loss_apex(output_micro_apex,
                                            micro_apex_score)
        
        loss_macro_apex = bi_loss_apex(output_macro_apex,
                                            macro_apex_score)
        loss_micro_action = bi_loss_action(output_micro_action,
                                              micro_action_score)
        loss_macro_action = bi_loss_action(output_macro_action,
                                              macro_action_score)
        micro_start_score=output_micro_start_end[:,0,:].to(device)
        #micro_start_score[micro_start_score == 0]=1
        macro_start_score = output_macro_start_end[:,0,:].to(device)
        #macro_start_score[macro_start_score == 0] = 1
        micro_end_score = output_micro_start_end[:,1,:].to(device)
        #micro_end_score[micro_end_score == 1] = 1
        macro_end_score = output_macro_start_end[:,1,:].to(device)
        #macro_end_score[macro_end_score == 1] = 1
        array_score_micro_apex = torch.sigmoid(
            output_micro_apex).detach().to(device)
        array_score_macro_apex = torch.sigmoid(
            output_macro_apex).detach().to(device)
        output_micro_start=calprobility(b,micro_start_score,output_micros_reg,array_score_micro_apex )
        output_macro_start=calprobility(b,macro_start_score,output_macros_reg,array_score_macro_apex)
        output_micro_end=calprobilityend(b,micro_end_score,output_microe_reg,array_score_micro_apex)
        output_macro_end=calprobilityend(b,macro_end_score,output_macroe_reg,array_score_macro_apex)

        #将得到的综合考虑的Fs与gt计算
        one_tensor = torch.tensor(1, device=device)
        zero_tensor = torch.tensor(0, device=device)
        milabel_start = micro_start_end_label.clone()
        milabel_start = torch.where(milabel_start == 0, one_tensor, zero_tensor)

        milabel_end = micro_start_end_label.clone()
        milabel_end = torch.where(milabel_end == 1, one_tensor, zero_tensor)

        # malabel_start and malabel_end
        malabel_start = macro_start_end_label.clone()
        malabel_start = torch.where(malabel_start == 0, one_tensor, zero_tensor)

        malabel_end = macro_start_end_label.clone()
        malabel_end = torch.where(malabel_end == 1, one_tensor, zero_tensor)
        #########
        loss_micro_start = bi_loss_apex(output_micro_start,
                                        milabel_start)
        loss_macro_start = bi_loss_apex(output_macro_start,
                                        malabel_start)
        loss_micro_end = bi_loss_apex(output_micro_end,
                                      milabel_end)
        loss_macro_end = bi_loss_apex(output_macro_end,
                                      malabel_end)
        # ##clloss
        # proto_vectors=model.memory.proto_vectors
        #
        # # 实例化损失函数
        # criterion = ContrastiveLoss(margin=1.0)
        #
        # # 计算损失
        # clloss = criterion(proto_vectors, emdfeatture)
        # calculate loss: 3-cls loss
        
        loss_micro_start_end = cls_loss_func(
            output_micro_start_end.permute(0, 2, 1).contiguous(),
            micro_start_end_label)
        loss_macro_start_end = cls_loss_func(
            output_macro_start_end.permute(0, 2, 1).contiguous(),
            macro_start_end_label)
        loss_micro_class = class_loss_func(
            output_micro_class.permute(0, 2, 1).contiguous(),
            micro_class_score)
        loss_macro_class = class_loss_func(
            output_macro_class.permute(0, 2, 1).contiguous(),
            macro_class_score)
        
        # aggregate loss
        loss = (1.8 * loss_micro_apex
                + 1.0 * loss_micro_start_end
                + 0.1 * loss_micro_action
                +1*loss_micro_class
                +loss_micro_start
                +loss_micro_end
                + 0.1 * (
                    1.0 * loss_macro_apex
                    + 1.0 * loss_macro_start_end
                    + 0.1 * loss_macro_action
                    + 0.1 * loss_macro_class
                    + 1*loss_macro_start
                    + 1*loss_macro_end

                ))
        
        # update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        # update losses
        loss_am.update(loss.detach())
        writer.add_scalar("Loss/train", loss, epoch)
    results = "[Epoch {0:03d}]\tLoss {1:.5f}(train)\n".format(
            epoch, loss_am.avg())
    print(results)
    
    state = {'epoch': epoch + 1,
             'state_dict': model.state_dict()}
    
    ckpt_dir = opt["model_save_root"]
    
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    
    weight_file = os.path.join(
                    ckpt_dir, 
                    "checkpoint_epoch_" + str(epoch).zfill(3) + ".pth.tar")
    
    # save state_dict every x epochs to save memory
    if (epoch + 1) % opt['save_intervals'] == 0:
        torch.save(state, weight_file)

            
if __name__ == '__main__':
    from pprint import pprint
    import opts
    
    args = opts.parse_args()
    
    # prep output folder
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        
    
    
    # load config & params.
    with open("./config.yaml", encoding="UTF-8") as f:
        yaml_config = yaml.safe_load(f)
        if args.dataset is not None:
            dataset = args.dataset
        else:
            dataset = yaml_config['dataset']
        opt = yaml_config[dataset]
        opt['dataset'] = dataset
    subject = args.subject
    
    # update opt. according to args.
    opt['output_dir_name'] = os.path.join(args.output, subject)
    opt['model_save_root'] = os.path.join(opt['output_dir_name'], 'models')
    
    # tensorboard writer
    writer_dir = os.path.join(opt['output_dir_name'], 'logs')
    if not os.path.exists(writer_dir):
        os.makedirs(writer_dir)
    tb_writer = SummaryWriter(writer_dir)
    
    
    # save the current config
    with open(os.path.join(writer_dir, 'config.txt'), 'w') as fid:
        pprint(opt, stream=fid)
        fid.flush()
        pprint(vars(args), stream=fid)
        fid.flush()
        
    # prep model
    device = opt['device'] if torch.cuda.is_available() else 'cpu'
    #-----fusemodel
    # num_features_stream1 = 2  # x, y 坐标
    # num_features_stream2 = 2
    # hidden_channels = 64
    # num_classes = 3
    #
    # edge_index = torch.from_numpy(np.load("/code/CodeTest/wzl/AUW-GCN/assets/cas(me)^2.npy"))
    # edge_index.to(device)
    # model = AUwGCN(opt, num_features_stream1, num_features_stream2, hidden_channels, edge_index)
    model = AUwGCN(opt)
    #-----fusemodel
    model = model.to(device)
    
    
    # define dataset and dataloader
    train_dataset = LOSO_DATASET(opt,args.input, "train", subject)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=opt['batch_size'],
                                               shuffle=True,
                                               num_workers=opt['num_workers'])
    # model.memory.init(args, model, train_loader,opt)
    # # define optimizer and scheduler
    optimizer = configure_optimizers(model, opt["abfcm_training_lr"],
                                     opt["abfcm_weight_decay"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, opt['abfcm_lr_scheduler'])


    for epoch in range(opt['epochs']):
        train(opt, train_loader, model, optimizer, epoch, device, tb_writer)
        scheduler.step()
    
    tb_writer.close()
    print("Finish training!")