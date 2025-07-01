import torch

def fully_supervised_triplet_loss(out1, out2, labels1, labels2, anchor_prompts, pos_prompts, anchor_view_id, sample_view_id):
    if ssl_prediction:
        fea1, anchor_preds = out1
        fea2, pos_preds = out2
    else:
        fea1 = out1
        fea2 = out2
    anchor_idxes = []
    pos_idxes = []
    neg_idxes = []
    for i, label1 in enumerate(labels1):
        for j, label2 in enumerate(labels2):
            if label1 == label2:
                for k, neg_label2 in enumerate(labels2):
                    if j == k:
                        continue
                    anchor_idxes.append(i)
                    pos_idxes.append(j)
                    neg_idxes.append(k)
    if len(anchor_idxes) == 0:
        return -1
    anchor_fea = fea1[anchor_idxes]
    pos_fea = fea2[pos_idxes]
    neg_fea = fea2[neg_idxes]
    pos_dis = torch.sqrt((anchor_fea - pos_fea).pow(2).sum(-1))
    neg_dis = torch.sqrt((anchor_fea - neg_fea).pow(2).sum(-1))
    M = 1.0
    loss = torch.nn.functional.relu(pos_dis - neg_dis + M).mean()
    
    idxes1 = []
    idxes2 = []
    for i, label1 in enumerate(labels1):
        for j, label2 in enumerate(labels2):
            if label1 == label2:
                idxes1.append(i)
                idxes2.append(j)
    
    anchor_loss1 = (anchor_preds[:, anchor_view_id[0]] - anchor_prompts).abs().sum(-1).mean()
    anchor_loss2 = (anchor_preds[idxes1, sample_view_id[0]] - pos_prompts[idxes2]).abs().sum(-1).mean()
    pos_loss1 = (pos_preds[idxes2, anchor_view_id[0]] - anchor_prompts[idxes1]).abs().sum(-1).mean()
    pos_loss2 = (pos_preds[:, sample_view_id[0]] - pos_prompts).abs().sum(-1).mean()

    loss += (anchor_loss1 + anchor_loss2 + pos_loss1 + pos_loss2)
    
    return loss