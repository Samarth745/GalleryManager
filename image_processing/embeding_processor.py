"""
CLIP Embedding Module
Provides functionality to embed images and videos using CLIP model.
Supports batch processing, model management, and intelligent video frame selection.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import clip
import cv2
import numpy as np
import torch
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingModule:
    """
    A module for embedding images and videos using the CLIP model.
    Handles model loading, batch image embedding, and optimized video frame extraction.
    """

    def __init__(self, model_name: str = "ViT-B/32", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.preprocess = None
        logger.info(f"EmbeddingModule initialized. Device: {self.device}")

    def load_model(self):
        if self.model is not None:
            logger.warning("Model already loaded. Skipping...")
            return

        logger.info(f"Loading CLIP model: {self.model_name}")
        self.model, self.preprocess = clip.load(self.model_name, device=self.device)
        self.model.eval()

        if self.device.startswith("cuda"):
            self.model = self.model.half()
            logger.info("Model converted to float16 for faster GPU inference")

        logger.info("Model loaded successfully")

    def unload_model(self):
        if self.model is None:
            logger.warning("No model to unload")
            return

        logger.info("Unloading model...")
        del self.model
        del self.preprocess
        self.model = None
        self.preprocess = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model unloaded and memory freed")

    def _ensure_model_loaded(self):
        if self.model is None:
            self.load_model()

    def _normalize_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        return embeddings / embeddings.norm(dim=-1, keepdim=True)

    def _load_and_preprocess_image(self, image_path: str) -> Optional[torch.Tensor]:
        try:
            image = Image.open(image_path).convert("RGB")
            return self.preprocess(image).unsqueeze(0)
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None

    def embed_image(self, image_path: str) -> Optional[torch.Tensor]:
        self._ensure_model_loaded()

        image = self._load_and_preprocess_image(image_path)
        if image is None:
            return None

        image = image.to(self.device)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.device.startswith("cuda")):
            embedding = self.model.encode_image(image)
            embedding = self._normalize_embeddings(embedding)

        return embedding.cpu()

    def embed_images_batch(self, image_paths: List[str], batch_size: int = 16) -> List[torch.Tensor]:
        self._ensure_model_loaded()

        if not image_paths:
            return []

        max_workers = min(8, len(image_paths))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            images = list(executor.map(self._load_and_preprocess_image, image_paths))

        images = [img for img in images if img is not None]
        if not images:
            logger.warning("No valid images to embed")
            return []

        self.model.to(self.device)
        image_embeddings: List[torch.Tensor] = []
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.device.startswith("cuda")):
            for i in range(0, len(images), batch_size):
                batch = torch.cat(images[i : i + batch_size]).to(self.device)
                embeddings = self.model.encode_image(batch)
                embeddings = self._normalize_embeddings(embeddings).cpu()
                image_embeddings.extend([emb.unsqueeze(0) for emb in embeddings])

        return image_embeddings

    def _extract_keyframes_with_sampling(
        self, video_path: str, num_frames: int = 16, scene_change_threshold: float = 0.35
    ) -> List[np.ndarray]:
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                logger.warning(f"Video has 0 frames: {video_path}")
                cap.release()
                return []

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            shot_boundaries = self._detect_shot_boundaries(video_path, total_frames, fps, threshold=scene_change_threshold)

            if len(shot_boundaries) <= 2:
                frame_indices = sorted(set(np.linspace(0, total_frames - 1, min(num_frames, total_frames), dtype=int)))
            else:
                frame_indices = self._select_shot_representative_frames(shot_boundaries, num_frames, total_frames)

            selected = sorted(set(frame_indices))
            frames: List[np.ndarray] = []
            current_index = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            while current_index <= selected[-1]:
                ret, frame = cap.read()
                if not ret:
                    break
                if current_index in selected:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if len(frames) >= len(selected):
                        break
                current_index += 1

            cap.release()
            return frames
        except Exception as e:
            logger.error(f"Error extracting keyframes from {video_path}: {e}")
            return []

    def _detect_shot_boundaries(
        self, video_path: str, total_frames: int, fps: float, threshold: float = 0.35
    ) -> List[int]:
        boundaries: List[int] = [0]
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return boundaries

        desired_samples = 90
        sample_rate = max(1, min(int(round(total_frames / desired_samples)), int(round(fps * 0.5))))
        prev_hist = None

        for frame_idx in range(0, total_frames, sample_rate):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()

            if prev_hist is not None:
                diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
                if diff > threshold and frame_idx - boundaries[-1] > sample_rate * 2:
                    boundaries.append(frame_idx)

            prev_hist = hist

        boundaries.append(total_frames - 1)
        cap.release()
        return boundaries

    def _select_shot_representative_frames(
        self, boundaries: List[int], num_frames: int, total_frames: int
    ) -> List[int]:
        shot_ranges = list(zip(boundaries[:-1], boundaries[1:]))
        representative_indices = [(start + end) // 2 for start, end in shot_ranges]

        if len(representative_indices) >= num_frames:
            selected_shots = np.linspace(0, len(representative_indices) - 1, num_frames, dtype=int)
            return [representative_indices[i] for i in selected_shots]

        selected = representative_indices.copy()
        extra_needed = num_frames - len(selected)
        long_shots = sorted(shot_ranges, key=lambda shot: (shot[1] - shot[0]), reverse=True)

        for start, end in long_shots:
            if extra_needed <= 0:
                break
            if end - start > 4:
                quarter = start + (end - start) // 4
                three_quarter = start + 3 * (end - start) // 4
                for candidate in (quarter, three_quarter):
                    if candidate not in selected:
                        selected.append(candidate)
                        extra_needed -= 1
                        if extra_needed <= 0:
                            break

        if extra_needed > 0:
            fallback = list(sorted(set(np.linspace(0, total_frames - 1, num_frames * 2, dtype=int))))
            for candidate in fallback:
                if candidate not in selected:
                    selected.append(candidate)
                    extra_needed -= 1
                    if extra_needed <= 0:
                        break

        return sorted(set(selected))[:num_frames]

    def embed_video(self, video_path: str, num_frames: int = 16, return_frame_embeddings: bool = False) -> Optional[Dict[str, torch.Tensor]]:
        self._ensure_model_loaded()

        frames = self._extract_keyframes_with_sampling(video_path, num_frames)
        if not frames:
            logger.warning(f"No frames extracted from {video_path}")
            return None

        processed_frames = [self.preprocess(Image.fromarray(frame)).unsqueeze(0) for frame in frames]
        self.model.to(self.device)

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.device.startswith("cuda")):
            batch = torch.cat(processed_frames).to(self.device)
            embeddings = self.model.encode_image(batch)
            embeddings = self._normalize_embeddings(embeddings).cpu()

        video_embedding = self._normalize_embeddings(torch.mean(embeddings, dim=0, keepdim=True))

        result: Dict[str, torch.Tensor] = {
            "video_embedding": video_embedding,
            "num_frames_used": torch.tensor([embeddings.shape[0]], dtype=torch.int32),
        }
        if return_frame_embeddings:
            result["frame_embeddings"] = embeddings

        return result

    def _embed_video_task(self, video_path: str, num_frames: int) -> Optional[Dict[str, torch.Tensor]]:
        logger.info(f"Processing video: {video_path}")
        try:
            return self.embed_video(video_path, num_frames)
        except Exception as exc:
            logger.error(f"Video embedding failed for {video_path}: {exc}")
            return None

    def embed_videos_batch(self, video_paths: List[str], num_frames: int = 16) -> List[Dict[str, torch.Tensor]]:
        self._ensure_model_loaded()

        if not video_paths:
            return []

        max_workers = min(4, len(video_paths))
        if self.device.startswith("cuda"):
            # GPU inference is typically faster when run sequentially to avoid device contention.
            max_workers = 1

        logger.info(f"Embedding {len(video_paths)} videos with max_workers={max_workers}")
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for result in executor.map(lambda video_path: self._embed_video_task(video_path, num_frames), video_paths):
                if result is not None:
                    results.append(result)

        return results




