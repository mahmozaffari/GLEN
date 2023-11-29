# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmf.common.registry import registry


@registry.register_loss("correct_pred")
class CorrectnessPredictionLoss(nn.Module):
    def __init__(self, **params):
        super().__init__()
        self.target_type = params["target_type"]
        self.t = params.get("acc_threshold", 0.5)

        assert self.target_type in ["threshold", "max_ind", "regress_bce", "regress_mse", "regress_l1"]

        if self.target_type == "regress_bce":
            self.loss_func = nn.BCELoss(reduction="mean")
        elif self.target_type == "regress_mse":
            self.loss_func = nn.MSELoss()
        elif self.target_type == "regress_l1":
            self.loss_func = nn.L1Loss()
        else:
            self.loss_func = nn.CrossEntropyLoss(weight=torch.Tensor([1.0, 1.0]))

    def _masked_unk_softmax(self, x, dim, mask_idx):
        """
        Copied from VQAAccuracy.
        """
        x1 = F.softmax(x, dim=dim)
        x1[:, mask_idx] = 0
        x1_sum = torch.sum(x1, dim=1, keepdim=True)
        y = x1 / x1_sum
        return y

    def forward(self, sample_list, model_output):
        """
        Compute binary correctness prediction loss.
        Requires:
            scores --> model logits over answers
            targets --> ground truth accuracies for each answer
            confidences --> confidence of correctness prediction (binary from 2-class softmax)
        """
        logits = model_output["scores"]
        # for three branch movie+mcan model
        if logits.dim() == 3:
            logits = logits[:, 0]

        targets = sample_list["targets"]

        normalized_logits = self._masked_unk_softmax(logits, 1, 0)
        pred_inds = torch.argmax(normalized_logits, dim=1)

        if self.target_type == "max_ind":
            tgt_inds = torch.argmax(targets, dim=1)
            correctness = (pred_inds == tgt_inds).to(dtype=torch.long)
        else:
            one_hots = targets.new_zeros(*targets.size())
            one_hots.scatter_(1, pred_inds.view(-1, 1), 1)
            tgt_scores = torch.sum(one_hots * targets, dim=-1)
            if "regress" in self.target_type:
                tgt_scores = tgt_scores.unsqueeze(1)
                correctness = torch.cat([1. - tgt_scores, tgt_scores], dim=-1)
            else:
                correctness = (tgt_scores >= self.t).to(dtype=torch.long)

        confidences = model_output["confidences"]  # normalized confidences, B x 2 if not regression
        
        if self.target_type == "regress_bce":
            return self.loss_func(confidences, correctness) * correctness.size(1)
        else:
            return self.loss_func(confidences, correctness)
        
# Added: GFL loss
@registry.register_loss("gfl_logit_bce")
class GFLLogitBinaryCrossEntropy(nn.Module):
    """Returns Binary Cross Entropy for logits.

    Attention:
        `Key`: logit_bce
    """
    
    def __init__(self,**params):
        super().__init__()
        self.gfl_lambda = params.get("lambda", 1)

    def forward(self, sample_list, model_output):
        """Calculates and returns the re-weighted binary cross entropy for logits

        Args:
            sample_list (SampleList): SampleList containing `targets` attribute.
            model_output (Dict): Model output containing `scores` attribute.

        Returns:
            torch.FloatTensor: Float value for loss.

        """
        scores = model_output["scores"]
        targets = sample_list["targets"]
        
        loss = F.binary_cross_entropy_with_logits(scores, targets, reduction="none")
        loss = torch.sum(loss,dim=1)
        mx = torch.max(loss)
        
        # re-weighting for GFL
        L = (loss - mx)/self.gfl_lambda
        Z = torch.exp(L)/torch.sum(torch.exp(L)) #nom/denom
        
        return torch.sum(Z*loss)
    

@registry.register_loss("gfl_triple_logit_bce")
class GFLTripleLogitBinaryCrossEntropy(nn.Module):
    """
    This is used for Three-branch fusion only. We predict scores and compute
    cross entropy loss for each of branches.
    """

    def __init__(self,**params):
        super().__init__()
        self.dro_lambda = params.get("lambda", 1)

    def forward(self, sample_list, model_output):
        """Calculates and returns the re-weighted binary cross entropy for logits
        Args:
            sample_list (SampleList): SampleList containing `targets` attribute.
            model_output (Dict): Model output containing `scores` attribute.
        Returns:
            torch.FloatTensor: Float value for loss.
        """
        scores = model_output["scores"]
        targets = sample_list["targets"]

        if scores.dim() == 3:
            loss1 = F.binary_cross_entropy_with_logits(
                    scores[:, 0], targets, reduction="none"
                )
            loss1 = torch.sum(loss1,dim=1)
            mx = torch.max(loss1)
            L = (loss1 - mx)/self.dro_lambda
            Z = torch.exp(L)/torch.sum(torch.exp(L))
            loss1 = torch.sum(Z*loss1)

            loss2 = F.binary_cross_entropy_with_logits(
                    scores[:, 1], targets, reduction="none"
                )
            loss2 = torch.sum(loss2,dim=1)
            mx = torch.max(loss2)
            L = (loss2 - mx)/self.dro_lambda
            Z = torch.exp(L)/torch.sum(torch.exp(L))
            loss2 = torch.sum(Z*loss2)

            loss3 = F.binary_cross_entropy_with_logits(
                    scores[:, 2], targets, reduction="none"
                )
            loss3 = torch.sum(loss3,dim=1)
            mx = torch.max(loss3)
            L = (loss3 - mx)/self.dro_lambda
            Z = torch.exp(L)/torch.sum(torch.exp(L))
            loss3 = torch.sum(Z*loss3)
            
            loss = loss1 + loss2 + loss3
        else:
            loss = F.binary_cross_entropy_with_logits(scores, targets, reduction="none")
            loss = torch.sum(loss,dim=1)
            mx = torch.max(loss)
            L = (loss - mx)/self.dro_lambda
            Z = torch.exp(L)/torch.sum(torch.exp(L))
            loss = torch.sum(Z*loss)

        return loss

# End of GFL loss