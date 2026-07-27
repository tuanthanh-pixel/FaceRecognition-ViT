import argparse
import os
import random

import torch

from config import get_parser
from data import build_dataloaders
from metrics import (
    calculate_identification_metrics,
    calculate_verification_metrics,
)
from model import build_model
from visualization import (
    get_experiment_dir,
    plot_test_results,
    save_test_results,
)


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


@torch.no_grad()
def extract_embeddings(model, data_loader, device):
    model.eval()
    all_embeddings = []
    all_labels = []

    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        embeddings = model(images)
        all_embeddings.append(embeddings.cpu())
        all_labels.append(labels)

    return torch.cat(all_embeddings), torch.cat(all_labels)


def create_verification_pairs(embeddings, labels, number_of_pairs, seed):
    rng = random.Random(seed)
    label_to_indices = {}

    for index, label in enumerate(labels.tolist()):
        label_to_indices.setdefault(label, []).append(index)

    unique_labels = list(label_to_indices.keys())
    pair_labels = []
    pair_distances = []

    for pair_index in range(number_of_pairs):
        same_person = pair_index % 2 == 0

        if same_person:
            label = rng.choice(unique_labels)
            first_index, second_index = rng.sample(
                label_to_indices[label],
                2,
            )
            pair_label = 1
        else:
            first_label, second_label = rng.sample(unique_labels, 2)
            first_index = rng.choice(label_to_indices[first_label])
            second_index = rng.choice(label_to_indices[second_label])
            pair_label = 0

        distance = torch.linalg.vector_norm(
            embeddings[first_index] - embeddings[second_index],
            ord=2,
        ).item()
        pair_labels.append(pair_label)
        pair_distances.append(distance)

    return pair_labels, pair_distances


def create_gallery_probe_split(
    labels,
    gallery_images_per_identity,
    seed,
):
    rng = random.Random(seed)
    label_to_indices = {}

    for index, label in enumerate(labels.tolist()):
        label_to_indices.setdefault(label, []).append(index)

    gallery_indices = []
    probe_indices = []
    for label, indices in sorted(label_to_indices.items()):
        shuffled_indices = indices.copy()
        rng.shuffle(shuffled_indices)
        if len(shuffled_indices) <= gallery_images_per_identity:
            raise ValueError(
                f"Identity {label} khong du anh de tao gallery va probe."
            )
        gallery_indices.extend(
            shuffled_indices[:gallery_images_per_identity]
        )
        probe_indices.extend(
            shuffled_indices[gallery_images_per_identity:]
        )

    return (
        torch.tensor(gallery_indices, dtype=torch.long),
        torch.tensor(probe_indices, dtype=torch.long),
    )


def main():
    cli_cfg = get_parser()
    device = get_device()
    checkpoint_path = os.path.join(
        "checkpoints",
        f"{cli_cfg.experiment_name}_best.pth",
    )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model_cfg = argparse.Namespace(**checkpoint["config"])
    model_cfg.dataset_root = cli_cfg.dataset_root
    model_cfg.batch_size = cli_cfg.batch_size
    model_cfg.num_workers = cli_cfg.num_workers
    model_cfg.identities_per_batch = cli_cfg.identities_per_batch
    model_cfg.images_per_identity = cli_cfg.images_per_identity
    model_cfg.validation_identity_ratio = cli_cfg.validation_identity_ratio

    loaders, class_names, split_class_names = build_dataloaders(model_cfg)
    model = build_model(model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    embeddings, labels = extract_embeddings(
        model,
        loaders["test_images"],
        device,
    )
    pair_labels, pair_distances = create_verification_pairs(
        embeddings,
        labels,
        number_of_pairs=cli_cfg.verification_pairs,
        seed=model_cfg.seed,
    )
    verification_metrics, roc_curve = calculate_verification_metrics(
        pair_labels,
        pair_distances,
    )
    gallery_indices, probe_indices = create_gallery_probe_split(
        labels,
        cli_cfg.gallery_images_per_identity,
        model_cfg.seed,
    )
    identification_metrics = calculate_identification_metrics(
        embeddings,
        labels,
        gallery_indices,
        probe_indices,
    )

    positive_distances = [
        distance
        for label, distance in zip(pair_labels, pair_distances)
        if label == 1
    ]
    negative_distances = [
        distance
        for label, distance in zip(pair_labels, pair_distances)
        if label == 0
    ]

    results = {
        "checkpoint": checkpoint_path,
        "best_epoch": checkpoint["epoch"],
        "test_identities": len(set(labels.tolist())),
        "test_images": len(labels),
        "verification_pairs": cli_cfg.verification_pairs,
        "mean_positive_distance": (
            sum(positive_distances) / len(positive_distances)
        ),
        "mean_negative_distance": (
            sum(negative_distances) / len(negative_distances)
        ),
        "verification": verification_metrics,
        "gallery_images": len(gallery_indices),
        "probe_images": len(probe_indices),
        "gallery_images_per_identity": (
            cli_cfg.gallery_images_per_identity
        ),
        "identification": identification_metrics,
    }
    experiment_dir = get_experiment_dir(cli_cfg.experiment_name)
    embeddings_path = os.path.join(
        experiment_dir,
        "test_embeddings.pt",
    )
    torch.save(
        {
            "embeddings": embeddings,
            "labels": labels,
            "image_paths": loaders["test_images"].dataset.image_paths,
            "class_names": class_names,
            "test_class_names": split_class_names["test"],
            "gallery_indices": gallery_indices,
            "probe_indices": probe_indices,
            "eer_distance_threshold": verification_metrics[
                "eer_distance_threshold"
            ],
        },
        embeddings_path,
    )
    results["embeddings_path"] = embeddings_path
    results_path = save_test_results(results, cli_cfg.experiment_name)
    figure_path = plot_test_results(
        pair_labels,
        pair_distances,
        roc_curve,
        verification_metrics,
        identification_metrics,
        cli_cfg.experiment_name,
    )

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Best epoch: {checkpoint['epoch']}")
    print(f"Test identities: {len(set(labels.tolist()))}")
    print(f"Test images: {len(labels)}")
    print(
        "Mean positive distance: "
        f"{sum(positive_distances) / len(positive_distances):.4f}"
    )
    print(
        "Mean negative distance: "
        f"{sum(negative_distances) / len(negative_distances):.4f}"
    )
    print(f"ROC-AUC: {verification_metrics['roc_auc']:.4f}")
    print(f"EER: {verification_metrics['eer']:.2%}")
    print(
        "EER distance threshold: "
        f"{verification_metrics['eer_distance_threshold']:.4f}"
    )
    print(
        "Verification accuracy at EER: "
        f"{verification_metrics['verification_accuracy_at_eer']:.2%}"
    )
    print(
        "TAR@FAR=1%: "
        f"{verification_metrics['tar_at_far_0.01']:.2%}"
    )
    print(
        "TAR@FAR=0.1%: "
        f"{verification_metrics['tar_at_far_0.001']:.2%}"
    )
    print(f"Gallery images: {len(gallery_indices)}")
    print(f"Probe images: {len(probe_indices)}")
    print(
        f"Recall@1: {identification_metrics['recall_at_1']:.2%}"
    )
    print(
        f"Recall@5: {identification_metrics['recall_at_5']:.2%}"
    )
    print(
        "mAP: "
        f"{identification_metrics['mean_average_precision']:.2%}"
    )
    print(f"Saved test results to: {results_path}")
    print(f"Saved test figure to: {figure_path}")
    print(f"Saved test embeddings to: {embeddings_path}")


if __name__ == "__main__":
    main()
