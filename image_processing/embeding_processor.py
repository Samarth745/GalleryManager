"""
CLIP Embedding Module
Provides functionality to embed images and videos using CLIP model.
Supports batch processing, model management, and intelligent video frame selection.
"""

from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import torch
import clip
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingModule:
    """
    A module for embedding images and videos using CLIP model.
    Handles model loading/unloading and batch processing for both images and videos.
    """
    
    def __init__(self, model_name: str = "ViT-B/32", device: Optional[str] = None):
        """
        Initialize the embedding module.
        
        Args:
            model_name: CLIP model to use (default: ViT-B/32)
            device: Device to use ('cuda' or 'cpu'). Auto-detects if None.
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.preprocess = None
        logger.info(f"EmbeddingModule initialized. Device: {self.device}")
    
    def load_model(self):
        """Load CLIP model and preprocessing function."""
        if self.model is not None:
            logger.warning("Model already loaded. Skipping...")
            return
        
        logger.info(f"Loading CLIP model: {self.model_name}")
        self.model, self.preprocess = clip.load(self.model_name, device=self.device)
        self.model.eval()
        logger.info("Model loaded successfully")
    
    def unload_model(self):
        """Unload model and free GPU memory."""
        if self.model is None:
            logger.warning("No model to unload")
            return
        
        logger.info("Unloading model...")
        del self.model
        del self.preprocess
        self.model = None
        self.preprocess = None
        torch.cuda.empty_cache()
        logger.info("Model unloaded and memory freed")
    
    def _ensure_model_loaded(self):
        """Ensure model is loaded before use."""
        if self.model is None:
            self.load_model()
    
    # ======================== IMAGE EMBEDDING ========================
    
    def _load_and_preprocess_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess a single image."""
        try:
            image = Image.open(image_path).convert('RGB')
            return self.preprocess(image).unsqueeze(0)
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None
    
    def embed_image(self, image_path: str) -> torch.Tensor:
        """
        Embed a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Normalized embedding tensor of shape (1, 512)
        """
        self._ensure_model_loaded()
        
        try:
            image = self._load_and_preprocess_image(image_path)
            if image is None:
                return None
            
            image = image.to(self.device)
            with torch.no_grad():
                embedding = self.model.encode_image(image)
                embedding /= embedding.norm(dim=-1, keepdim=True)
            
            return embedding.cpu()
        except Exception as e:
            logger.error(f"Error embedding image {image_path}: {e}")
            return None
    
    def embed_images_batch(self, image_paths: List[str], batch_size: int = 16) -> List[torch.Tensor]:
        """
        Embed multiple images in batches.
        
        Args:
            image_paths: List of image file paths
            batch_size: Batch size for processing
            
        Returns:
            List of normalized embedding tensors
        """
        self._ensure_model_loaded()
        
        image_embeddings = []
        
        # Load images in parallel
        with ThreadPoolExecutor(max_workers=8) as executor:
            images = list(executor.map(self._load_and_preprocess_image, image_paths))
        
        # Filter out failed loads
        images = [img for img in images if img is not None]
        
        if not images:
            logger.warning("No valid images to embed")
            return []
        
        # Process images in batches
        self.model.to(self.device)
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = torch.cat(images[i:i+batch_size]).to(self.device)
                embeddings = self.model.encode_image(batch)
                embeddings /= embeddings.norm(dim=-1, keepdim=True)
                embeddings = embeddings.cpu()
                image_embeddings.extend([emb.unsqueeze(0) for emb in embeddings])
        
        return image_embeddings
    
    # ======================== VIDEO EMBEDDING ========================
    
    def _extract_keyframes_with_sampling(self, video_path: str, num_frames: int = 16) -> List[np.ndarray]:
        """
        Extract keyframes using two strategies:
        1. Temporal stratified sampling (uniform distribution throughout video)
        2. Keyframe detection (scene changes detected via optical flow)
        
        Args:
            video_path: Path to video file
            num_frames: Target number of frames to extract
            
        Returns:
            List of frame arrays (RGB format)
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            if total_frames == 0:
                logger.warning(f"Video has 0 frames: {video_path}")
                cap.release()
                return []
            
            # Strategy: Combine temporal sampling + keyframe detection
            # 1. Temporal stratified sampling
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            
            # 2. Additional keyframe detection at scene changes
            keyframe_indices = self._detect_scene_changes(cap, total_frames, threshold=0.3)
            
            # Combine and deduplicate
            combined_indices = sorted(set(list(frame_indices) + keyframe_indices))[:num_frames]
            
            frames = []
            for frame_idx in combined_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
            
            cap.release()
            return frames
        
        except Exception as e:
            logger.error(f"Error extracting keyframes from {video_path}: {e}")
            return []
    
    def _detect_scene_changes(self, cap: cv2.VideoCapture, total_frames: int, threshold: float = 0.15) -> List[int]:
        """
        Detect scene changes using histogram comparison and optical flow.
        
        Args:
            cap: OpenCV VideoCapture object
            total_frames: Total number of frames in video
            threshold: Sensitivity threshold for scene change (0-1)
            
        Returns:
            List of frame indices with scene changes
        """
        keyframe_indices = []
        prev_frame = None
        
        # Sample every Nth frame for faster processing
        sample_rate = max(1, total_frames // 50)  # Check ~50 frames max
        
        for frame_idx in range(0, total_frames, sample_rate):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            if prev_frame is not None:
                # Compare histograms to detect scene changes
                hist_curr = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8],
                                         [0, 256, 0, 256, 0, 256])
                hist_prev = cv2.calcHist([prev_frame], [0, 1, 2], None, [8, 8, 8],
                                         [0, 256, 0, 256, 0, 256])
                
                # Normalize histograms
                hist_curr = cv2.normalize(hist_curr, hist_curr).flatten()
                hist_prev = cv2.normalize(hist_prev, hist_prev).flatten()
                
                # Compare using chi-square distance
                similarity = cv2.compareHist(hist_curr, hist_prev, cv2.HISTCMP_BHATTACHARYYA)
                
                if similarity > threshold:
                    keyframe_indices.append(frame_idx)
            
            prev_frame = frame.copy()
        
        return keyframe_indices
    
    def embed_video(self, video_path: str, num_frames: int = 16, return_frame_embeddings: bool = False) -> Dict:
        """
        Embed a video as a single vector by extracting and averaging keyframe embeddings.
        
        Strategy:
        - Extracts keyframes using temporal sampling + scene change detection
        - Embeds each keyframe using CLIP
        - Returns mean embedding for the video
        - Optionally returns per-frame embeddings for detailed analysis
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract and embed
            return_frame_embeddings: If True, also return individual frame embeddings
            
        Returns:
            Dictionary containing:
                - 'video_embedding': Mean embedding for the video
                - 'frame_embeddings': Individual frame embeddings (if return_frame_embeddings=True)
                - 'num_frames_used': Number of frames actually used
        """
        self._ensure_model_loaded()
        
        try:
            # Extract keyframes
            frames = self._extract_keyframes_with_sampling(video_path, num_frames)
            
            if not frames:
                logger.warning(f"No frames extracted from {video_path}")
                return None
            
            # Convert frames to CLIP preprocessed format
            processed_frames = []
            for frame in frames:
                pil_image = Image.fromarray(frame)
                processed = self.preprocess(pil_image).unsqueeze(0)
                processed_frames.append(processed)
            
            # Embed frames
            frame_embeddings = []
            self.model.to(self.device)
            
            with torch.no_grad():
                batch = torch.cat(processed_frames).to(self.device)
                embeddings = self.model.encode_image(batch)
                embeddings /= embeddings.norm(dim=-1, keepdim=True)
                frame_embeddings = embeddings.cpu()
            
            # Compute mean embedding for the video
            video_embedding = torch.mean(frame_embeddings, dim=0, keepdim=True)
            video_embedding /= video_embedding.norm()
            return video_embedding
        
        except Exception as e:
            logger.error(f"Error embedding video {video_path}: {e}")
            return None
    
    def embed_videos_batch(self, video_paths: List[str], num_frames: int = 16) -> List[Dict]:
        """
        Embed multiple videos.
        
        Args:
            video_paths: List of video file paths
            num_frames: Number of frames to extract per video
            
        Returns:
            List of result dictionaries from embed_video()
        """
        self._ensure_model_loaded()
        
        results = []
        for i, video_path in enumerate(video_paths):
            logger.info(f"Processing video {i+1}/{len(video_paths)}: {video_path}")
            result = self.embed_video(video_path, num_frames)
            if result is not None:
                results.append(result)
        
        return results
    
    # ======================== QUERY EMBEDDING ========================
    
    def encode_query(self, query: str) -> torch.Tensor:
        """
        Encode a text query using CLIP.
        
        Args:
            query: Text query to embed
            
        Returns:
            Normalized embedding tensor
        """
        self._ensure_model_loaded()
        
        try:
            text_inputs = clip.tokenize([query]).to(self.device)
            with torch.no_grad():
                text_embedding = self.model.encode_text(text_inputs)
                text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
            
            return text_embedding.cpu()
        except Exception as e:
            logger.error(f"Error encoding query '{query}': {e}")
            return None




