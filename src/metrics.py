import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve


def calculate_eer(false_positive_rate, true_positive_rate, thresholds):
    false_negative_rate = 1.0 - true_positive_rate
    index = np.nanargmin(
        np.abs(false_positive_rate - false_negative_rate)
    )
    eer = (
        false_positive_rate[index] + false_negative_rate[index]
    ) / 2
    return float(eer), float(thresholds[index])


def calculate_verification_metrics(
    labels,
    distances,
    far_targets=(1e-2, 1e-3),
):
    labels = np.asarray(labels, dtype=np.int64)
    distances = np.asarray(distances, dtype=np.float64)
    scores = -distances
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        labels,
        scores,
    )
    auc = roc_auc_score(labels, scores)
    eer, eer_score_threshold = calculate_eer(
        false_positive_rate,
        true_positive_rate,
        thresholds,
    )
    eer_distance_threshold = -eer_score_threshold
    predictions = (distances <= eer_distance_threshold).astype(np.int64)

    metrics = {
        "roc_auc": float(auc),
        "eer": eer,
        "eer_distance_threshold": float(eer_distance_threshold),
        "verification_accuracy_at_eer": float(
            np.mean(predictions == labels)
        ),
    }

    for far_target in far_targets:
        valid_indices = np.flatnonzero(false_positive_rate <= far_target)
        if valid_indices.size == 0:
            tar = 0.0
            distance_threshold = 0.0
        else:
            best_index = valid_indices[
                np.argmax(true_positive_rate[valid_indices])
            ]
            tar = true_positive_rate[best_index]
            distance_threshold = -thresholds[best_index]

        target_name = f"{far_target:g}"
        metrics[f"tar_at_far_{target_name}"] = float(tar)
        metrics[f"threshold_at_far_{target_name}"] = float(
            distance_threshold
        )

    curve = {
        "false_positive_rate": false_positive_rate,
        "true_positive_rate": true_positive_rate,
    }
    return metrics, curve


def calculate_identification_metrics(
    embeddings,
    labels,
    gallery_indices,
    probe_indices,
    recall_ks=(1, 5),
):
    gallery_embeddings = embeddings[gallery_indices]
    gallery_labels = labels[gallery_indices]
    probe_embeddings = embeddings[probe_indices]
    probe_labels = labels[probe_indices]

    distances = torch.cdist(probe_embeddings, gallery_embeddings, p=2)
    ranked_indices = distances.argsort(dim=1)
    ranked_labels = gallery_labels[ranked_indices]
    relevant = ranked_labels == probe_labels.unsqueeze(1)

    metrics = {}
    for recall_k in recall_ks:
        effective_k = min(recall_k, gallery_embeddings.size(0))
        recalled = relevant[:, :effective_k].any(dim=1).float().mean()
        metrics[f"recall_at_{recall_k}"] = float(recalled.item())

    cumulative_relevant = relevant.cumsum(dim=1)
    ranks = torch.arange(
        1,
        relevant.size(1) + 1,
        device=relevant.device,
        dtype=torch.float32,
    )
    precision_at_rank = cumulative_relevant / ranks.unsqueeze(0)
    relevant_counts = relevant.sum(dim=1).clamp_min(1)
    average_precision = (
        precision_at_rank * relevant
    ).sum(dim=1) / relevant_counts
    metrics["mean_average_precision"] = float(
        average_precision.mean().item()
    )
    return metrics


# Kept for compatibility with older notebooks and scripts.
def calculate_roc_metrics(labels, distances):
    metrics, _ = calculate_verification_metrics(labels, distances)
    return {
        "roc_auc": metrics["roc_auc"],
        "eer": metrics["eer"],
        "eer_threshold": -metrics["eer_distance_threshold"],
    }
