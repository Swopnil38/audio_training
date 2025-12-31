"""
Data preparation service for Whisper fine-tuning.
Converts corrected transcriptions from the database to HuggingFace Dataset format.
"""
import os
import re
import json
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import numpy as np
import librosa
import soundfile as sf
from datasets import Dataset, DatasetDict, Audio
from django.conf import settings

logger = logging.getLogger(__name__)

# Devanagari Unicode range
DEVANAGARI_RANGE = (0x0900, 0x097F)
DEVANAGARI_DIGITS = '०१२३४५६७८९'
ASCII_DIGITS = '0123456789'


class TextPreprocessor:
    """Preprocess text for Nepali (Devanagari) and English"""

    @staticmethod
    def is_devanagari(char: str) -> bool:
        """Check if character is Devanagari script"""
        if len(char) != 1:
            return False
        code = ord(char)
        return DEVANAGARI_RANGE[0] <= code <= DEVANAGARI_RANGE[1]

    @staticmethod
    def get_devanagari_ratio(text: str) -> float:
        """Calculate ratio of Devanagari characters in text"""
        if not text:
            return 0.0
        devanagari_count = sum(1 for c in text if TextPreprocessor.is_devanagari(c))
        alpha_count = sum(1 for c in text if c.isalpha())
        return devanagari_count / alpha_count if alpha_count > 0 else 0.0

    @staticmethod
    def detect_language(text: str) -> str:
        """Detect if text is Nepali, English, or mixed"""
        ratio = TextPreprocessor.get_devanagari_ratio(text)
        if ratio > 0.8:
            return 'nepali'
        elif ratio < 0.2:
            return 'english'
        else:
            return 'mixed'

    @staticmethod
    def normalize_devanagari_digits(text: str) -> str:
        """Convert ASCII digits to Devanagari digits in Nepali text"""
        trans_table = str.maketrans(ASCII_DIGITS, DEVANAGARI_DIGITS)
        return text.translate(trans_table)

    @staticmethod
    def is_romanized_nepali(text: str) -> bool:
        """
        Detect romanized Nepali (e.g., "namaste" instead of "नमस्ते").
        Returns True if text appears to be romanized Nepali.
        """
        # Common romanized Nepali patterns
        romanized_patterns = [
            r'\b(namaste|dhanyabad|kasto|cha|ho|haina|ramro|khabar)\b',
            r'\b(tapai|timi|ma|hami|uni|yo|tyo|ke|kasari)\b',
            r'\b(ghar|kam|kaam|paisa|desh|nepal)\b',
        ]
        text_lower = text.lower()
        for pattern in romanized_patterns:
            if re.search(pattern, text_lower):
                return True
        return False

    @staticmethod
    def clean_nepali(text: str) -> str:
        """Clean and normalize Nepali (Devanagari) text"""
        # Normalize Unicode (NFC = composed form)
        text = unicodedata.normalize('NFC', text)

        # Remove URLs and emails
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'\S+@\S+\.\S+', '', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Convert digits to Devanagari
        text = TextPreprocessor.normalize_devanagari_digits(text)

        return text

    @staticmethod
    def clean_english(text: str) -> str:
        """Clean and normalize English text"""
        # Lowercase
        text = text.lower()

        # Remove URLs and emails
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'\S+@\S+\.\S+', '', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    @staticmethod
    def preprocess(text: str, language: str = None) -> str:
        """Preprocess text based on detected or specified language"""
        if language is None:
            language = TextPreprocessor.detect_language(text)

        if language == 'nepali':
            return TextPreprocessor.clean_nepali(text)
        elif language == 'english':
            return TextPreprocessor.clean_english(text)
        else:  # mixed
            # Apply both normalizations carefully
            text = unicodedata.normalize('NFC', text)
            text = re.sub(r'http[s]?://\S+', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text


class AudioPreprocessor:
    """Preprocess audio files for Whisper"""

    TARGET_SR = 16000  # Whisper requires 16kHz

    @staticmethod
    def load_and_resample(audio_path: str) -> Tuple[np.ndarray, int]:
        """Load audio and resample to 16kHz mono"""
        audio, sr = librosa.load(audio_path, sr=AudioPreprocessor.TARGET_SR, mono=True)
        return audio, AudioPreprocessor.TARGET_SR

    @staticmethod
    def get_duration(audio_path: str) -> float:
        """Get audio duration in seconds"""
        return librosa.get_duration(path=audio_path)

    @staticmethod
    def normalize_audio(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
        """Normalize audio to target dB level"""
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 0:
            target_rms = 10 ** (target_db / 20)
            audio = audio * (target_rms / rms)
        return np.clip(audio, -1.0, 1.0)

    @staticmethod
    def is_valid_duration(audio_path: str, min_dur: float = 0.5, max_dur: float = 30.0) -> bool:
        """Check if audio duration is within valid range for Whisper training"""
        try:
            duration = AudioPreprocessor.get_duration(audio_path)
            return min_dur <= duration <= max_dur
        except Exception as e:
            logger.error(f"Error checking duration for {audio_path}: {e}")
            return False


class DataPrepService:
    """
    Prepare training data from corrected transcriptions in the database.
    Converts to HuggingFace Dataset format for Whisper training.
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.text_processor = TextPreprocessor()
        self.audio_processor = AudioPreprocessor()

    def fetch_corrections(self, min_samples: int = 100) -> List[Dict]:
        """
        Fetch approved corrected transcriptions from database.

        Returns:
            List of dicts with audio_path, text, start_time, end_time, language
        """
        from asr_app.models import Correction, TranscriptSegment

        # Fetch approved corrections with related data
        corrections = Correction.objects.filter(
            # Add your approval criteria here
            # For now, we'll use corrections that have non-empty corrected_text
        ).select_related('segment', 'segment__audio_file').order_by('created_at')

        if corrections.count() < min_samples:
            logger.warning(
                f"Only {corrections.count()} corrections available, "
                f"minimum {min_samples} recommended"
            )

        data = []
        for correction in corrections:
            segment = correction.segment
            audio_file = segment.audio_file

            # Get the corrected text
            text = correction.corrected_text

            if not text or not text.strip():
                continue

            # Skip if text is romanized Nepali
            if TextPreprocessor.is_romanized_nepali(text):
                logger.warning(f"Skipping romanized Nepali: {text[:50]}...")
                continue

            # Get audio file path
            audio_path = audio_file.file.path if hasattr(audio_file.file, 'path') else str(audio_file.file)

            data.append({
                'audio_path': audio_path,
                'text': text,
                'start_time': segment.start_time,
                'end_time': segment.end_time,
                'language': TextPreprocessor.detect_language(text),
                'audio_file_id': str(audio_file.id),
                'segment_id': str(segment.id),
            })

        logger.info(f"Fetched {len(data)} corrections from database")
        return data

    def extract_segment(self, audio_path: str, start: float, end: float, output_path: str) -> bool:
        """
        Extract audio segment and save to file.

        Args:
            audio_path: Path to source audio file
            start: Start time in seconds
            end: End time in seconds
            output_path: Path to save extracted segment

        Returns:
            True if successful, False otherwise
        """
        try:
            # Load segment from audio file
            duration = end - start
            audio, sr = librosa.load(
                audio_path,
                sr=16000,
                mono=True,
                offset=start,
                duration=duration
            )

            # Normalize audio
            audio = AudioPreprocessor.normalize_audio(audio)

            # Save to file
            sf.write(output_path, audio, sr)
            return True

        except Exception as e:
            logger.error(f"Failed to extract segment from {audio_path}: {e}")
            return False

    def prepare_dataset(
        self,
        min_samples: int = 100,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ) -> DatasetDict:
        """
        Prepare HuggingFace Dataset from corrections.

        Args:
            min_samples: Minimum number of samples required
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set

        Returns:
            DatasetDict with train/val/test splits
        """
        logger.info("Fetching corrections from database...")
        corrections = self.fetch_corrections(min_samples=min_samples)

        if len(corrections) < min_samples:
            raise ValueError(
                f"Not enough samples: {len(corrections)} < {min_samples}. "
                "Please add more corrected transcriptions."
            )

        logger.info(f"Processing {len(corrections)} corrections...")

        # Process each correction
        processed_data = []
        audio_dir = self.output_dir / 'audio'
        audio_dir.mkdir(exist_ok=True)

        for i, item in enumerate(corrections):
            # Extract segment audio
            segment_path = audio_dir / f"segment_{i:05d}.wav"

            success = self.extract_segment(
                item['audio_path'],
                item['start_time'],
                item['end_time'],
                str(segment_path)
            )

            if not success:
                logger.warning(f"Skipping segment {i} due to extraction failure")
                continue

            # Validate duration
            if not AudioPreprocessor.is_valid_duration(str(segment_path)):
                logger.warning(f"Skipping segment {i} due to invalid duration")
                segment_path.unlink()  # Remove invalid file
                continue

            # Preprocess text
            text = TextPreprocessor.preprocess(item['text'], item['language'])

            if not text.strip():
                logger.warning(f"Skipping segment {i} due to empty text after preprocessing")
                segment_path.unlink()
                continue

            processed_data.append({
                'audio': str(segment_path),
                'text': text,
                'language': item['language'],
            })

            # Log progress every 100 samples
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(corrections)} samples...")

        logger.info(f"Successfully processed {len(processed_data)} valid samples")

        if len(processed_data) < min_samples:
            raise ValueError(
                f"After processing, only {len(processed_data)} valid samples remain. "
                f"Need at least {min_samples}."
            )

        # Language distribution
        lang_counts = {}
        for item in processed_data:
            lang = item['language']
            lang_counts[lang] = lang_counts.get(lang, 0) + 1

        logger.info(f"Language distribution: {lang_counts}")

        # Create HuggingFace Dataset
        dataset = Dataset.from_list(processed_data)
        dataset = dataset.cast_column('audio', Audio(sampling_rate=16000))

        # Split dataset (stratified by language would be ideal, but simple random split for now)
        # First split: train vs (val + test)
        train_test = dataset.train_test_split(test_size=(val_ratio + test_ratio), seed=42)

        # Second split: val vs test
        val_test_ratio = test_ratio / (val_ratio + test_ratio)
        test_val = train_test['test'].train_test_split(test_size=val_test_ratio, seed=42)

        dataset_dict = DatasetDict({
            'train': train_test['train'],
            'validation': test_val['train'],
            'test': test_val['test'],
        })

        # Save to disk
        dataset_path = self.output_dir / 'dataset'
        dataset_dict.save_to_disk(str(dataset_path))
        logger.info(f"Dataset saved to: {dataset_path}")

        # Calculate total audio duration
        total_duration = sum(
            AudioPreprocessor.get_duration(item['audio'])
            for item in processed_data
        )

        # Save metadata
        metadata = {
            'total_samples': len(processed_data),
            'language_distribution': lang_counts,
            'splits': {
                'train': len(dataset_dict['train']),
                'validation': len(dataset_dict['validation']),
                'test': len(dataset_dict['test']),
            },
            'total_audio_hours': total_duration / 3600,
        }

        metadata_path = self.output_dir / 'metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"Metadata saved to: {metadata_path}")
        logger.info(f"Total audio: {metadata['total_audio_hours']:.2f} hours")

        return dataset_dict
