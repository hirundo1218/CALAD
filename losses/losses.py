import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

EPS=1e-8

class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MaskedCrossEntropyLoss, self).__init__()
        
    def forward(self, input, target, mask, weight, reduction='mean'):
        if not (mask != 0).any():
            raise ValueError('Mask in MaskedCrossEntropyLoss is all zeros.')
        target = torch.masked_select(target, mask)
        b, c = input.size()
        n = target.size(0)
        input = torch.masked_select(input, mask.view(b, 1)).view(n, c)
        return F.cross_entropy(input, target, weight = weight, reduction = reduction)


def entropy(x, input_as_probabilities):
    if input_as_probabilities:
        x_ =  torch.clamp(x, min = EPS)
        b =  x_ * torch.log(x_)
    else:
        b = F.softmax(x, dim = 1) * F.log_softmax(x, dim = 1)

    if len(b.size()) == 2:
        return -b.sum(dim = 1).mean()
    elif len(b.size()) == 1:
        return - b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' %(len(b.size())))


class ClassificationLoss(nn.Module):
    def __init__(self, entropy_weight = 2.0, inconsistency_weight=0.0):
        super(ClassificationLoss, self).__init__()
        self.softmax = nn.Softmax(dim = 1)
        self.bce = nn.BCELoss()
        self.entropy_weight = entropy_weight 
        self.inconsistency_weight = inconsistency_weight

    def forward(self, anchors, nneighbors, fneighbors):
        def check_and_fix_nan(tensor, name):
            if torch.isnan(tensor).any():
                print(f"Warning: NaN detected in {name}. Replacing with zeros.")
                return torch.nan_to_num(tensor, nan=0.0)
            return tensor
        
        b, n = anchors.size()
        anchors_prob = self.softmax(anchors)
        positives_prob = self.softmax(nneighbors)
        negatives_prob = self.softmax(fneighbors)
        
        # Similarity in output space
        similarity = torch.bmm(anchors_prob.view(b, 1, n), positives_prob.view(b, n, 1)).squeeze()
        ones = torch.ones_like(similarity)
        consistency_loss = self.bce(similarity, ones)

        # DiSimilarity in output space
        negsimilarity = torch.bmm(anchors_prob.view(b, 1, n), negatives_prob.view(b, n, 1)).squeeze()
        zeros = torch.zeros_like(negsimilarity)
        inconsistency_loss = self.bce(negsimilarity, zeros)
        
        # Entropy loss
        entropy_loss = entropy(torch.mean(anchors_prob, 0), input_as_probabilities = True)

        # Total loss
        total_loss = consistency_loss - self.entropy_weight * entropy_loss + self.inconsistency_weight * inconsistency_loss

        return total_loss, consistency_loss, inconsistency_loss, entropy_loss


class PretextLoss(nn.Module):
    def __init__(self, bs, temperature, initial_margin=1.0, adjust_factor=0.1):
        super(PretextLoss, self).__init__()
        self.temperature = temperature
        self.bs = bs
        self.margin = initial_margin
        self.adjust_factor = adjust_factor

    def forward(self, features, current_loss=None):
        features_org, features_pos, features_subseq = torch.split(features, self.bs, dim=0)

        anchor = F.normalize(features_org, dim=-1)
        positive = F.normalize(features_pos, dim=-1)
        negative = F.normalize(features_subseq, dim=-1)

        if current_loss is not None:
            self.margin = max(0.01, self.margin - self.adjust_factor * current_loss)

        positive_distance = torch.sum((anchor - positive) ** 2, dim=-1) / self.temperature
        negative_distance = torch.sum(torch.pow(anchor.unsqueeze(1) - negative, 2), dim=-1) / self.temperature
        hard_negative_distance = torch.min(negative_distance, dim=1)[0]
        loss = torch.clamp(self.margin + positive_distance - hard_negative_distance, min=0.0)
        loss = torch.mean(loss)

        return loss

    def cosine_similarity(self, x1, x2):
        dot_product = torch.sum(x1 * x2, dim=1)
        norm_product = torch.norm(x1, dim=1) * torch.norm(x2, dim=1)
        cosine_similarity = dot_product / norm_product
        return cosine_similarity

    def euclidan_dist(self, x1, x2):
        return torch.sqrt(((x1 - x2)**2).sum(dim=1))