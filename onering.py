mask = F.one_hot(labels, num_classes=logits.shape[-1])
outs_unk = logits[~mask.bool()].reshape(len(logits), -1)
label_unk = (torch.ones(len(logits), dtype=torch.long)
            * (logits.shape[-1]-2)).cuda()
#======== unknow classification loss

loss_unk = torch.nn.CrossEntropyLoss()(outs_unk, label_unk) * options['lambda']