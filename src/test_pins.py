import argparse
import os
import random

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import get_parser
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


# ======================================================
# Device
# ======================================================

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


# ======================================================
# Dataset Pins
# ======================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


class PinsDataset(Dataset):

    def __init__(self, root, image_size):

        self.image_paths = []
        self.labels = []
        self.class_names = []

        transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5] * 3,
                    std=[0.5] * 3,
                ),
            ]
        )

        self.transform = transform

        identity_names = sorted(
            folder
            for folder in os.listdir(root)
            if os.path.isdir(
                os.path.join(root, folder)
            )
        )

        for label, identity in enumerate(identity_names):

            self.class_names.append(identity)

            identity_dir = os.path.join(
                root,
                identity,
            )

            for image_name in sorted(
                os.listdir(identity_dir)
            ):

                extension = os.path.splitext(
                    image_name
                )[1].lower()

                if extension not in IMAGE_EXTENSIONS:
                    continue

                self.image_paths.append(
                    os.path.join(
                        identity_dir,
                        image_name,
                    )
                )

                self.labels.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        image = Image.open(
            self.image_paths[index]
        ).convert("RGB")

        image = self.transform(image)

        return image, self.labels[index]


def build_loader(cfg):

    dataset = PinsDataset(
        "/kaggle/input/datasets/hereisburak/pins-face-recognition/105_classes_pins_dataset",
        cfg.image_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    return loader, dataset
# ======================================================
# Load checkpoint
# ======================================================

checkpoint_path = (
    "/kaggle/input/datasets/tritechsic/"
    "checkpoint6/"
    "sic_facevit_vggface2_semi_hard_best.pth"
)


# ======================================================
# Extract embeddings
# ======================================================

@torch.no_grad()
def extract_embeddings(model, loader, device):

    model.eval()

    all_embeddings = []
    all_labels = []

    for images, labels in loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        embeddings = model(images)

        all_embeddings.append(
            embeddings.cpu()
        )

        all_labels.append(labels)

    return (
        torch.cat(all_embeddings),
        torch.cat(all_labels),
    )


# ======================================================
# Verification pairs
# ======================================================

def create_verification_pairs(
    embeddings,
    labels,
    number_of_pairs,
    seed,
):

    rng = random.Random(seed)

    label_to_indices = {}

    for index, label in enumerate(labels.tolist()):
        label_to_indices.setdefault(
            label,
            []
        ).append(index)

    unique_labels = list(
        label_to_indices.keys()
    )

    pair_labels = []
    pair_distances = []

    for pair_index in range(number_of_pairs):

        same_person = (
            pair_index % 2 == 0
        )

        if same_person:

            label = rng.choice(
                unique_labels
            )

            first_index, second_index = rng.sample(
                label_to_indices[label],
                2,
            )

            pair_label = 1

        else:

            first_label, second_label = rng.sample(
                unique_labels,
                2,
            )

            first_index = rng.choice(
                label_to_indices[first_label]
            )

            second_index = rng.choice(
                label_to_indices[second_label]
            )

            pair_label = 0

        distance = torch.linalg.vector_norm(
            embeddings[first_index]
            - embeddings[second_index],
            ord=2,
        ).item()

        pair_labels.append(pair_label)
        pair_distances.append(distance)

    return (
        pair_labels,
        pair_distances,
    )


# ======================================================
# Gallery / Probe
# ======================================================

def create_gallery_probe_split(
    labels,
    gallery_images_per_identity,
    seed,
):

    rng = random.Random(seed)

    label_to_indices = {}

    for index, label in enumerate(
        labels.tolist()
    ):
        label_to_indices.setdefault(
            label,
            []
        ).append(index)

    gallery_indices = []
    probe_indices = []

    for label, indices in sorted(
        label_to_indices.items()
    ):

        shuffled = indices.copy()
        rng.shuffle(shuffled)
        if len(shuffled) <= gallery_images_per_identity:
            continue
        gallery_indices.extend(
            shuffled[:gallery_images_per_identity]
        )

        probe_indices.extend(
            shuffled[gallery_images_per_identity:]
        )

    return (
        torch.tensor(
            gallery_indices,
            dtype=torch.long,
        ),
        torch.tensor(
            probe_indices,
            dtype=torch.long,
        ),
    )
# ======================================================
# Main
# ======================================================

def main():

    cli_cfg = get_parser()

    device = get_device()

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model_cfg = argparse.Namespace(
        **checkpoint["config"]
    )

    model_cfg.batch_size = cli_cfg.batch_size
    model_cfg.num_workers = cli_cfg.num_workers

    loader, dataset = build_loader(model_cfg)

    model = build_model(model_cfg).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        f"Loaded checkpoint: {checkpoint_path}"
    )

    embeddings, labels = extract_embeddings(
        model,
        loader,
        device,
    )

    pair_labels, pair_distances = (
        create_verification_pairs(
            embeddings,
            labels,
            number_of_pairs=cli_cfg.verification_pairs,
            seed=model_cfg.seed,
        )
    )

    verification_metrics, roc_curve = (
        calculate_verification_metrics(
            pair_labels,
            pair_distances,
        )
    )

    gallery_indices, probe_indices = (
        create_gallery_probe_split(
            labels,
            gallery_images_per_identity=cli_cfg.gallery_images_per_identity,
            seed=model_cfg.seed,
        )
    )

    identification_metrics = (
        calculate_identification_metrics(
            embeddings,
            labels,
            gallery_indices,
            probe_indices,
        )
    )
    # ======================================================
    # Statistics
    # ======================================================

    positive_distances = [
        distance
        for label, distance in zip(
            pair_labels,
            pair_distances,
        )
        if label == 1
    ]

    negative_distances = [
        distance
        for label, distance in zip(
            pair_labels,
            pair_distances,
        )
        if label == 0
    ]

    results = {
        "checkpoint": checkpoint_path,
        "best_epoch": checkpoint["epoch"],
        "test_identities": len(
            set(labels.tolist())
        ),
        "test_images": len(labels),
        "verification_pairs": 10000,
        "mean_positive_distance": (
            sum(positive_distances)
            / len(positive_distances)
        ),
        "mean_negative_distance": (
            sum(negative_distances)
            / len(negative_distances)
        ),
        "verification": verification_metrics,
        "gallery_images": len(
            gallery_indices
        ),
        "probe_images": len(
            probe_indices
        ),
        "gallery_images_per_identity": cli_cfg.gallery_images_per_identity,
        "identification": identification_metrics,
    }

    experiment_dir = get_experiment_dir(
        "pins_test"
    )

    embeddings_path = os.path.join(
        experiment_dir,
        "pins_embeddings.pt",
    )

    torch.save(
        {
            "embeddings": embeddings,
            "labels": labels,
            "image_paths": dataset.image_paths,
            "class_names": dataset.class_names,
            "gallery_indices": gallery_indices,
            "probe_indices": probe_indices,
            "eer_distance_threshold":
                verification_metrics[
                    "eer_distance_threshold"
                ],
        },
        embeddings_path,
    )

    results["embeddings_path"] = (
        embeddings_path
    )

    results_path = save_test_results(
        results,
        "pins_test",
    )

    figure_path = plot_test_results(
        pair_labels,
        pair_distances,
        roc_curve,
        verification_metrics,
        identification_metrics,
        "pins_test",
    )
    print(f"Best epoch: {checkpoint['epoch']}")
    print(
        f"Test identities: {len(set(labels.tolist()))}"
    )
    print(f"Test images: {len(labels)}")

    print(
        "Mean positive distance: "
        f"{results['mean_positive_distance']:.4f}"
    )

    print(
        "Mean negative distance: "
        f"{results['mean_negative_distance']:.4f}"
    )

    print(
        f"ROC-AUC: "
        f"{verification_metrics['roc_auc']:.4f}"
    )

    print(
        f"EER: "
        f"{verification_metrics['eer']:.2%}"
    )

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

    print(
        f"Gallery images: {len(gallery_indices)}"
    )

    print(
        f"Probe images: {len(probe_indices)}"
    )

    print(
        f"Recall@1: "
        f"{identification_metrics['recall_at_1']:.2%}"
    )

    print(
        f"Recall@5: "
        f"{identification_metrics['recall_at_5']:.2%}"
    )

    print(
        "mAP: "
        f"{identification_metrics['mean_average_precision']:.2%}"
    )

    print(
        f"Saved figure to: {figure_path}"
    )

    print(
        f"Saved results to: {results_path}"
    )

    print(
        f"Saved embeddings to: {embeddings_path}"
    )


if __name__ == "__main__":
    main()