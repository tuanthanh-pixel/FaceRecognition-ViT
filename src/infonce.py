import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):

    def __init__(self, temperature=0.07, use_weighted_negative=False,
        hard_negative_gamma=2.0):
        super().__init__()
        self.temperature = temperature
        self.use_weighted_negative = use_weighted_negative
        self.hard_negative_gamma = hard_negative_gamma

    def forward(self, embeddings, labels):

        embeddings = F.normalize(embeddings, dim=1)

        similarity = torch.matmul(
            embeddings,
            embeddings.t()
        ) / self.temperature

        batch_size = embeddings.size(0)

        labels = labels.unsqueeze(1)

        positive_mask = (labels == labels.T).float()

        identity = torch.eye(
            batch_size,
            device=embeddings.device
        )

        positive_mask = positive_mask - identity
        negative_mask = 1 - positive_mask - identity

        logits_mask = 1 - identity

        weights = torch.ones_like(similarity)

        if self.use_weighted_negative:

            negative_similarity = similarity.detach()

            negative_weight = torch.exp(
                self.hard_negative_gamma
                *
                negative_similarity
            )

            weights = (
                positive_mask
                +
                identity
                +
                negative_mask
                *
                negative_weight
            )

        exp_similarity = (
            torch.exp(similarity)
            *
            logits_mask
            *
            weights
        )

        log_prob = similarity - torch.log(
            exp_similarity.sum(dim=1, keepdim=True) + 1e-12
        )

        mean_log_prob = (
            positive_mask * log_prob
        ).sum(dim=1) / (
            positive_mask.sum(dim=1) + 1e-12
        )

        loss = -mean_log_prob.mean()

        return loss