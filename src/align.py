import torch
from PIL import Image
from torchvision import transforms


class FaceAligner:
    """Align a face image to a canonical crop using MTCNN landmarks.

    Falls back to the original image when no face is detected or when the
    detector cannot be loaded (e.g. no internet for the first weight download).
    """

    def __init__(self, image_size=224, margin=0, device="cpu"):
        self.image_size = image_size
        self.margin = margin
        self.device = device
        self._mtcnn = None
        self._load_error = None
        self._to_pil = transforms.ToPILImage()

    def _ensure_loaded(self):
        if self._mtcnn is None and self._load_error is None:
            try:
                from facenet_pytorch import MTCNN

                self._mtcnn = MTCNN(
                    image_size=self.image_size,
                    margin=self.margin,
                    min_face_size=20,
                    keep_all=False,
                    device=self.device,
                )
            except Exception as error:
                self._load_error = error
        return self._mtcnn

    @property
    def is_enabled(self):
        aligner = self._ensure_loaded()
        return aligner is not None

    def align(self, image):
        aligner = self._ensure_loaded()
        if aligner is None:
            return image

        aligned = aligner(image)
        if aligned is None:
            return image

        aligned_image = self._to_pil(aligned.squeeze(0))
        if aligned_image.size != (self.image_size, self.image_size):
            aligned_image = aligned_image.resize(
                (self.image_size, self.image_size),
                Image.BILINEAR,
            )
        return aligned_image


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")