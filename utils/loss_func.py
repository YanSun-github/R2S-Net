import torch
from torch.autograd import Variable

class MultiCEFocalLoss_New(torch.nn.Module):
    def __init__(self, class_num, gamma=2, alpha=None, lb_smooth=0,
                 reduction='mean'):
        super(MultiCEFocalLoss_New, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.class_num = class_num

    def forward(self, predict, target ):
        pt = torch.softmax(predict, dim=-1).view(-1, self.class_num)
        class_mask = torch.nn.functional.one_hot(
            target, self.class_num).view(-1, self.class_num)

        # # 创建掩蔽 (mask) 张量，忽略超过长度的帧
        # frame_mask = torch.arange(pt.size(0)).to(predict.device) < length
        # pt = pt * frame_mask.unsqueeze(1).float()
        # class_mask = class_mask * frame_mask.unsqueeze(1).float()

        ids = target.view(-1, 1)
        alpha = self.alpha.clone().to(ids.device)
        alpha = alpha[ids.view(-1)].view(-1, 1)
        alpha = alpha.to(predict.device)

        positive_class_mask_indices = torch.nonzero(
            class_mask[:, 2] == 0).squeeze()
        negative_class_mask_indices = torch.nonzero(
            class_mask[:, 2] == 1).squeeze()
        positive_pt = pt[positive_class_mask_indices]
        negative_pt = pt[negative_class_mask_indices]
        positive_class_mask = class_mask[positive_class_mask_indices]
        negative_class_mask = class_mask[negative_class_mask_indices]
        positive_alpha = alpha[positive_class_mask_indices]
        negative_alpha = alpha[negative_class_mask_indices]

        # p_num = torch.sum(class_mask[:, :-1]).item()
        # n_num = torch.sum(class_mask[:, -1]).item()
        # if torch.sum(class_mask[:, -1]) == class_mask.shape[0]:
        #     return 0
        # negative_alpha = 1 / math.log2(n_num / p_num)
        # positive_alpha = 1 - negative_alpha

        positive_probs = (positive_pt * positive_class_mask).sum(-1).view(-1, 1)
        positive_log_p = positive_probs.log()
        positive_loss = -positive_alpha * torch.pow(
            (1 - positive_probs), self.gamma) * positive_log_p

        negative_probs = (negative_pt * negative_class_mask).sum(-1).view(-1, 1)

        positive_probs = torch.clamp(positive_probs, min=1e-8, max=1 - 1e-8)
        negative_probs = torch.clamp(negative_probs, min=1e-8, max=1 - 1e-8)

        negative_log_p = negative_probs.log()
        negative_loss = -negative_alpha * torch.pow(
            torch.clamp(1 - self.lb_smooth - negative_probs, min=0),
            self.gamma) * negative_log_p
        if torch.isnan(positive_probs).any() or torch.isnan(negative_probs).any():
            print("NaN detected in probabilities")
        if torch.isinf(positive_probs).any() or torch.isinf(negative_probs).any():
            print("Inf detected in probabilities")
        if torch.isnan(positive_loss).any() or torch.isnan(negative_loss).any():
            print("NaN detected in loss calculation")
        if torch.isinf(positive_loss).any() or torch.isinf(negative_loss).any():
            print("Inf detected in loss calculation")
        loss = torch.cat((positive_loss, negative_loss))

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

class aMultiCEFocalLoss_New(torch.nn.Module):
    def __init__(self, class_num, gamma=2, alpha=None, lb_smooth=0,
                 reduction='mean'):
        super(aMultiCEFocalLoss_New, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.class_num = class_num

    def forward(self, predict, target):
        pt = torch.softmax(predict, dim=-1).view(-1, self.class_num)
        class_mask = torch.nn.functional.one_hot(
            target, self.class_num).view(-1, self.class_num)
        ids = target.view(-1, 1)
        alpha = self.alpha.clone().to(ids.device)
        alpha = alpha[ids.view(-1)].view(-1, 1)
        alpha = alpha.to(predict.device)

        positive_class_mask_indices = torch.nonzero(
            class_mask[:, 0] == 0).squeeze()
        negative_class_mask_indices = torch.nonzero(
            class_mask[:, 0] == 1).squeeze()
        positive_pt = pt[positive_class_mask_indices]#少数类的预测
        negative_pt = pt[negative_class_mask_indices]
        positive_class_mask = class_mask[positive_class_mask_indices]#少数类的gt
        negative_class_mask = class_mask[negative_class_mask_indices]
        positive_alpha = alpha[positive_class_mask_indices]
        negative_alpha = alpha[negative_class_mask_indices]

        # p_num = torch.sum(class_mask[:, :-1]).item()
        # n_num = torch.sum(class_mask[:, -1]).item()
        # if torch.sum(class_mask[:, -1]) == class_mask.shape[0]:
        #     return 0
        # negative_alpha = 1 / math.log2(n_num / p_num)
        # positive_alpha = 1 - negative_alpha

        positive_probs = (positive_pt * positive_class_mask).sum(-1).view(-1, 1)
        positive_probs = torch.clamp(positive_probs, min=1e-8, max=1 - 1e-8)
        positive_log_p = positive_probs.log()

        positive_loss = -positive_alpha * torch.pow(
            (1 - positive_probs), self.gamma) * positive_log_p

        negative_probs = (negative_pt * negative_class_mask).sum(-1).view(-1, 1)


        negative_probs = torch.clamp(negative_probs, min=1e-8, max=1 - 1e-8)

        negative_log_p = negative_probs.log()
        negative_loss = -negative_alpha * torch.pow(
            torch.clamp(1 - self.lb_smooth - negative_probs, min=0),
            self.gamma) * negative_log_p
        if torch.isnan(positive_probs).any() or torch.isnan(negative_probs).any():
            print("NaN detected in probabilities")
        if torch.isinf(positive_probs).any() or torch.isinf(negative_probs).any():
            print("Inf detected in probabilities")
        if torch.isnan(positive_loss).any() or torch.isnan(negative_loss).any():
            print("NaN detected in loss calculation")
        if torch.isinf(positive_loss).any() or torch.isinf(negative_loss).any():
            print("Inf detected in loss calculation")
        loss = torch.cat((positive_loss, negative_loss))

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss
def _focal_loss(output, label, gamma, alpha, lb_smooth):
    output = output.contiguous().view(-1)
    label = label.reshape(-1)
    mask_class = (label > 0).float()

    # p_num = torch.sum(label > 0).item()
    # n_num = torch.sum(label == 0).item()
    # if p_num == 0:
    #     return 0
    # c_0 = 1 / math.log2(n_num / p_num)
    # c_1 = 1 - c_0

    c_1 = alpha
    c_0 = 1 - c_1
    loss = ((c_1 * torch.abs(label - output)**gamma * mask_class
            * torch.log(output + 0.00001))
            + (c_0 * torch.abs(label + lb_smooth - output)**gamma
            * (1.0 - mask_class)
            * torch.log(1.0 - output + 0.00001)))
    loss = -torch.mean(loss)
    return loss

def ce_loss(output, label, gamma, alpha, lb_smooth):
    output = output.contiguous().view(-1)
    label = label.reshape(-1)
    mask_class = (label > 0).float()

    # p_num = torch.sum(label > 0).item()
    # n_num = torch.sum(label == 0).item()
    # if p_num == 0:
    #     return 0
    # c_0 = 1 / math.log2(n_num / p_num)
    # c_1 = 1 - c_0


    loss = (( 30*(label  )  * mask_class
            * torch.log(output + 0.00001))
            + ( 1
            * (1.0 - mask_class)
            * torch.log(1.0 - output + 0.00001)))
    loss = -torch.mean(loss)
    return loss
def _probability_loss(output, score, gamma, alpha, lb_smooth):
    output = torch.sigmoid(output)
    loss = _focal_loss(output, score, gamma, alpha, lb_smooth)
    return loss
def _probability_lossce(output, score, gamma, alpha, lb_smooth):
    output = torch.sigmoid(output)
    loss = ce_loss(output, score, gamma, alpha, lb_smooth)
    return loss
def norm(data):
    l2 = torch.norm(data, p = 2, dim = -1, keepdim = True)
    return torch.div(data, l2)
def feat_loss_func(  embeded_feature,proto,args, act_seed, bkg_seed, vid_label):
    loss_contra = 0
    proto_vectors = norm(proto.to(
        args.device))  # [C,N,F]
    for b in range(act_seed.shape[0]):
        # >> extract pseudo-action/background features
        gt_class = torch.nonzero(vid_label[b]).squeeze(1)
        act_feat_lst = []
        for c in gt_class:
            act_feat_lst.append(utils.extract_region_feat(act_seed[b, :, c], embeded_feature[b, :, :]))
        bkg_feat = utils.extract_region_feat(bkg_seed[b].squeeze(-1), embeded_feature[b, :, :])

        # >> caculate similarity matrix
        if len(bkg_feat) == 0:
            continue
        bkg_feat = norm(torch.cat(bkg_feat, 0))  # [t_b,F]
        b_sim_matrix = torch.matmul(bkg_feat.unsqueeze(0).expand(args.num_class, -1, -1),
                                    torch.transpose(proto_vectors, 1, 2)) / 0.1  # [C,t_b,N]
        b_sim_matrix = torch.exp(b_sim_matrix).reshape(b_sim_matrix.shape[0], -1).mean(dim=-1)  # [C]
        for idx, act_feat in enumerate(act_feat_lst):
            if act_feat is not None:
                if len(act_feat) == 0:
                    continue
                act_feat = norm(torch.cat(act_feat, 0))  # [t_a,F]
                a_sim_matrix = torch.matmul(act_feat.unsqueeze(0).expand(args.num_class, -1, -1),
                                            torch.transpose(proto_vectors, 1, 2)) / 0.1  # [C,t_a,N]
                a_sim_matrix = torch.exp(a_sim_matrix).reshape(a_sim_matrix.shape[0], -1).mean(
                    dim=-1)  # [C]

                # >> caculate contrastive loss
                c = gt_class[idx]
                loss_contra_act = - torch.log(a_sim_matrix[c] / a_sim_matrix.sum())
                loss_contra_bkg = - torch.log(a_sim_matrix[c] /
                                              (a_sim_matrix[c] + b_sim_matrix[c]))
                loss_contra += (0.5 * loss_contra_act + 0.5 * loss_contra_bkg)

        loss_contra = loss_contra / gt_class.shape[0]
    loss_contra = loss_contra / act_seed.shape[0]

    return loss_contra