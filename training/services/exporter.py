"""
Export fine-tuned Whisper models to optimized formats.
- CTranslate2 (faster-whisper) - recommended for Python deployment
- GGML (whisper.cpp) - best CPU performance (optional)
"""
import os
import logging
import subprocess
from pathlib import Path
from typing import Optional

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

logger = logging.getLogger(__name__)


class ModelExporter:
    """Export Whisper models to optimized inference formats"""

    def __init__(self, model_path: str, output_dir: str):
        """
        Initialize exporter.

        Args:
            model_path: Path to trained model (merged, full model)
            output_dir: Directory to save exported models
        """
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_ctranslate2(
        self,
        quantization: str = "int8",
    ) -> str:
        """
        Export to CTranslate2 format for faster-whisper.

        Args:
            quantization: Quantization type
                - "int8": 8-bit integer quantization (recommended, ~4x smaller)
                - "int16": 16-bit integer quantization
                - "float16": 16-bit float (2x smaller)
                - "float32": No quantization (original size)

        Returns:
            Path to exported model

        Raises:
            ImportError: If ctranslate2 is not installed
            Exception: If conversion fails
        """
        output_path = self.output_dir / f"ctranslate2-{quantization}"

        logger.info(f"Exporting to CTranslate2 with {quantization} quantization...")

        try:
            from ctranslate2.converters import TransformersConverter

            # Convert model
            converter = TransformersConverter(str(self.model_path))
            converter.convert(
                str(output_path),
                quantization=quantization,
                force=True,
            )

            # Copy processor files (tokenizer, feature extractor)
            processor = WhisperProcessor.from_pretrained(self.model_path)
            processor.save_pretrained(str(output_path))

            # Calculate model size
            size_mb = self._get_directory_size(output_path)
            logger.info(f"CTranslate2 model saved to: {output_path}")
            logger.info(f"Model size: {size_mb:.1f} MB")

            return str(output_path)

        except ImportError:
            logger.error(
                "CTranslate2 not installed. Install with: pip install ctranslate2"
            )
            raise

        except Exception as e:
            logger.error(f"CTranslate2 export failed: {e}")
            raise

    def export_ggml(
        self,
        quantization: str = "q8_0",
    ) -> str:
        """
        Export to GGML format for whisper.cpp (CPU-optimized inference).

        NOTE: This requires whisper.cpp to be installed and available.
        Installation: https://github.com/ggerganov/whisper.cpp

        Args:
            quantization: GGML quantization type
                - "q8_0": 8-bit quantization
                - "q5_0": 5-bit quantization (smaller, slightly less accurate)
                - "q4_0": 4-bit quantization (smallest, less accurate)
                - "f16": float16
                - "f32": float32 (no quantization)

        Returns:
            Path to exported model

        Note:
            This is a placeholder implementation. Full GGML export requires
            whisper.cpp conversion tools. For now, we save in HuggingFace format
            with instructions for manual conversion.
        """
        output_path = self.output_dir / f"ggml-{quantization}"

        logger.warning(
            "GGML export requires whisper.cpp tools for full conversion. "
            "Saving in HuggingFace format for manual conversion."
        )

        try:
            # Load and save model in HuggingFace format
            model = WhisperForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,
            )

            hf_path = self.output_dir / "hf_model"
            hf_path.mkdir(parents=True, exist_ok=True)

            model.save_pretrained(str(hf_path))

            processor = WhisperProcessor.from_pretrained(self.model_path)
            processor.save_pretrained(str(hf_path))

            logger.info(f"HuggingFace model saved to: {hf_path}")
            logger.info(
                "\nTo convert to GGML format, use whisper.cpp tools:"
                "\n1. Clone whisper.cpp: git clone https://github.com/ggerganov/whisper.cpp"
                "\n2. Run conversion: python whisper.cpp/models/convert-h5-to-ggml.py {hf_path}"
                f"\n3. Quantize: ./whisper.cpp/quantize {hf_path}/ggml-model.bin {output_path}.bin {quantization}"
            )

            return str(hf_path)

        except Exception as e:
            logger.error(f"GGML export failed: {e}")
            raise

    def _get_directory_size(self, path: Path) -> int:
        """
        Get total size of directory in MB.

        Args:
            path: Directory path

        Returns:
            Size in megabytes
        """
        total = 0
        if path.is_dir():
            for f in path.rglob('*'):
                if f.is_file():
                    total += f.stat().st_size
        else:
            total = path.stat().st_size

        return total / (1024 * 1024)

    def get_model_size(self) -> int:
        """
        Get original model size in MB.

        Returns:
            Size in megabytes
        """
        return self._get_directory_size(self.model_path)

    def export_all(self, quantizations: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Export to all available formats.

        Args:
            quantizations: List of quantization types to export.
                If None, uses ["int8"] (recommended)

        Returns:
            Dict mapping format to exported path
        """
        if quantizations is None:
            quantizations = ["int8"]

        results = {}

        # Export to CTranslate2 for each quantization
        for quant in quantizations:
            try:
                path = self.export_ctranslate2(quantization=quant)
                results[f"ctranslate2-{quant}"] = path
            except Exception as e:
                logger.error(f"Failed to export CTranslate2 with {quant}: {e}")

        logger.info(f"Exported {len(results)} model variant(s)")

        return results


def test_faster_whisper_model(model_path: str, audio_file: str) -> Dict:
    """
    Test a faster-whisper (CTranslate2) model on an audio file.

    Args:
        model_path: Path to CTranslate2 model
        audio_file: Path to test audio file

    Returns:
        Dict with transcription and performance metrics

    Raises:
        ImportError: If faster-whisper is not installed
    """
    try:
        from faster_whisper import WhisperModel
        import time

        logger.info(f"Testing model: {model_path}")
        logger.info(f"Audio file: {audio_file}")

        # Load model
        model = WhisperModel(model_path, device="cpu", compute_type="int8")

        # Transcribe
        start_time = time.time()
        segments, info = model.transcribe(audio_file, language="ne")
        elapsed = time.time() - start_time

        # Collect segments
        transcription = " ".join([seg.text for seg in segments])

        # Get audio duration
        import librosa
        audio_duration = librosa.get_duration(path=audio_file)

        # Calculate RTF (Real-Time Factor)
        rtf = elapsed / audio_duration

        results = {
            'transcription': transcription,
            'audio_duration': audio_duration,
            'processing_time': elapsed,
            'rtf': rtf,
            'language': info.language,
            'language_probability': info.language_probability,
        }

        logger.info(f"Transcription: {transcription}")
        logger.info(f"Processing time: {elapsed:.2f}s")
        logger.info(f"Audio duration: {audio_duration:.2f}s")
        logger.info(f"RTF: {rtf:.2f}x (lower is faster)")

        return results

    except ImportError:
        logger.error(
            "faster-whisper not installed. Install with: pip install faster-whisper"
        )
        raise
