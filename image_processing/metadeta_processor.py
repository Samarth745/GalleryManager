#@title image_processor.py
import os
import hashlib
import datetime
import uuid
from PIL import Image, ExifTags
import config.config_app as config
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import clip

class ImageProcessor:
    def __init__(self):
        pass

    def _generate_unique_id(self, file_path):
        """
        Generate a unique ID for an image file based on its content hash.
        This ensures the same file always gets the same ID.
        """
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            print(f"Error generating unique ID for {file_path}: {e}")
            # Fallback to UUID if hashing fails
            return str(uuid.uuid4())

    # ===========================Images ===============================

    def extract_metadata_image(self, image_path):
        try:
            with Image.open(image_path) as img:
                metadata = {
                    'image_id': self._generate_unique_id(image_path),
                    'filename': os.path.basename(image_path),
                    'filepath': image_path,
                    'format': img.format,
                    'mode': img.mode,
                    'width': img.width,
                    'height': img.height,
                    'file_size_bytes': os.path.getsize(image_path),
                    'file_size_mb': os.path.getsize(image_path) / (1024 * 1024),
                    'date_taken': None
                }
                exif = img._getexif()
                if exif:
                    exif_data = {ExifTags.TAGS.get(tag, tag): value for tag, value in exif.items()}
                    date = exif_data.get('DateTimeOriginal') or exif_data.get('DateTimeDigitized') or exif_data.get('DateTime')
                    metadata['date_taken'] = date

                # Fallback: if no EXIF date, use file creation date
                if metadata['date_taken'] is None:
                    creation_timestamp = os.path.getctime(image_path)
                    creation_date = datetime.datetime.fromtimestamp(creation_timestamp)
                    metadata['date_taken'] = creation_date.strftime("%Y:%m:%d %H:%M:%S")

                return metadata
        except Exception as e:
            print(f"Error extracting metadata from {image_path}: {e}")
            return None

    def calculate_blurriness(self, image_path):
        """Optional: Calculate blurriness score. Can be called separately if needed."""
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Image not found or unable to read: {image_path}")

            # Compute Laplacian of the image
            laplacian = cv2.Laplacian(image, cv2.CV_64F)

            # Calculate the variance of the Laplacian
            variance = laplacian.var()
            return variance
        except Exception as e:
            print(f"Error calculating blurriness: {e}")
            return None
    
    def get_image_metadeta(self, image_path):
        return self.extract_metadata_image(image_path)

    def get_image_metadeta_batch(self, image_paths):
        # Increased workers for I/O bound operations
        with ThreadPoolExecutor(max_workers=24) as executor:
            results = list(filter(None, executor.map(self.get_image_metadeta, image_paths)))
        return results

    # ================= Video ============================================
    def extract_metadeta_video(self, video_path):
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Unable to open video: {video_path}")
            
            # Extract video properties
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Calculate duration in seconds
            duration = frame_count / fps if fps > 0 else 0
            
            # Get codec
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            
            cap.release()
            
            # Get file information
            creation_timestamp = os.path.getctime(video_path)
            creation_date = datetime.datetime.fromtimestamp(creation_timestamp)
            
            metadata = {
                'filename': os.path.basename(video_path),
                'filepath': video_path,
                'format': os.path.splitext(video_path)[1].lstrip('.'),
                'width': width,
                'height': height,
                'fps': fps,
                'frame_count': frame_count,
                'duration_seconds': duration,
                'codec': codec,
                'date_created': creation_date.strftime("%Y:%m:%d %H:%M:%S"),
                'file_size_mb': os.path.getsize(video_path) / (1024 * 1024)
            }
            
            return metadata
        except Exception as e:
            print(f"Error extracting metadata from video {video_path}: {e}")
            return None

    def get_video_metadeta(self, video_path):
        return self.extract_metadeta_video(video_path)

    def get_video_metadeta_batch(self, video_paths):
        # Increased workers for I/O bound operations
        with ThreadPoolExecutor(max_workers=24) as executor:
            results = list(filter(None, executor.map(self.get_video_metadeta, video_paths)))
        return results

    # ================= Helpers  ============================================
    def get_media_from_folder(self, folder_path):
        """
        Recursively scans a folder and all subfolders to return all images and videos as a dictionary.
        
        Args:
            folder_path (str): Path to the folder to scan
            
        Returns:
            dict: Dictionary with 'images' and 'videos' keys containing file paths
        """
        if not os.path.isdir(folder_path):
            raise ValueError(f"Invalid folder path: {folder_path}")
        
        # Define supported formats
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v'}
        
        images = []
        videos = []
        
        try:
            # Recursively walk through all subdirectories
            for root, dirs, files in os.walk(folder_path):
                for filename in files:
                    file_path = os.path.join(root, filename)
                    
                    # Check file extension
                    file_ext = os.path.splitext(filename)[1].lower()
                    
                    if file_ext in image_extensions:
                        images.append(file_path)
                    elif file_ext in video_extensions:
                        videos.append(file_path)
            
            # Sort the lists for consistency
            images.sort()
            videos.sort()
            
            result = {
                'images': images,
                'videos': videos,
                'total_images': len(images),
                'total_videos': len(videos)
            }
            
            return result
        except Exception as e:
            print(f"Error scanning folder {folder_path}: {e}")
            return {'images': [], 'videos': [], 'total_images': 0, 'total_videos': 0}

    