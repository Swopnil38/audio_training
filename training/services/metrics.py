"""
Evaluation metrics for bilingual ASR.
- WER (Word Error Rate) for English
- CER (Character Error Rate) for Nepali (Devanagari)
"""
import re
import unicodedata
from typing import Dict, List, Tuple

from jiwer import wer, cer, compute_measures


class NepaliTextNormalizer:
    """Normalize Nepali (Devanagari) text for evaluation"""

    # Devanagari punctuation to remove
    PUNCTUATION = '।॥,;:!?()[]{}"\'-'

    @staticmethod
    def normalize(text: str) -> str:
        """
        Normalize Nepali text for CER computation.

        Args:
            text: Input Nepali text

        Returns:
            Normalized text
        """
        # Unicode normalization (NFC = composed form)
        text = unicodedata.normalize('NFC', text)

        # Remove punctuation
        for p in NepaliTextNormalizer.PUNCTUATION:
            text = text.replace(p, '')

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text


class EnglishTextNormalizer:
    """Normalize English text for evaluation"""

    @staticmethod
    def normalize(text: str) -> str:
        """
        Normalize English text for WER computation.

        Args:
            text: Input English text

        Returns:
            Normalized text
        """
        # Lowercase
        text = text.lower()

        # Remove punctuation (keep apostrophes for contractions)
        text = re.sub(r"[^\w\s']", '', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text


def compute_wer(reference: str, hypothesis: str, normalize: bool = True) -> Dict:
    """
    Compute Word Error Rate for English.

    Args:
        reference: Ground truth text
        hypothesis: Predicted text
        normalize: Whether to normalize text before computing WER

    Returns:
        Dict with wer, insertions, deletions, substitutions
    """
    if normalize:
        reference = EnglishTextNormalizer.normalize(reference)
        hypothesis = EnglishTextNormalizer.normalize(hypothesis)

    # Handle empty reference
    if not reference:
        return {
            'wer': 1.0 if hypothesis else 0.0,
            'insertions': 0,
            'deletions': 0,
            'substitutions': 0,
        }

    # Compute measures
    measures = compute_measures(reference, hypothesis)

    return {
        'wer': measures['wer'],
        'insertions': measures['insertions'],
        'deletions': measures['deletions'],
        'substitutions': measures['substitutions'],
    }


def compute_cer(reference: str, hypothesis: str, normalize: bool = True) -> Dict:
    """
    Compute Character Error Rate for Nepali (Devanagari).
    CER is more appropriate for Devanagari as word boundaries are less clear.

    Args:
        reference: Ground truth text
        hypothesis: Predicted text
        normalize: Whether to normalize text before computing CER

    Returns:
        Dict with cer
    """
    if normalize:
        reference = NepaliTextNormalizer.normalize(reference)
        hypothesis = NepaliTextNormalizer.normalize(hypothesis)

    # Remove spaces for character-level comparison
    reference = reference.replace(' ', '')
    hypothesis = hypothesis.replace(' ', '')

    # Handle empty reference
    if not reference:
        return {'cer': 1.0 if hypothesis else 0.0}

    # Compute CER
    error_rate = cer(reference, hypothesis)

    return {'cer': error_rate}


def compute_metrics_batch(
    references: List[str],
    hypotheses: List[str],
    languages: List[str]
) -> Dict:
    """
    Compute metrics for a batch of predictions.
    Uses CER for Nepali, WER for English.

    Args:
        references: List of ground truth texts
        hypotheses: List of predicted texts
        languages: List of language tags ('nepali', 'english', 'mixed')

    Returns:
        Dict with per-language metrics and overall metrics
    """
    nepali_refs, nepali_hyps = [], []
    english_refs, english_hyps = [], []
    mixed_refs, mixed_hyps = [], []

    # Separate by language
    for ref, hyp, lang in zip(references, hypotheses, languages):
        if lang == 'nepali':
            nepali_refs.append(ref)
            nepali_hyps.append(hyp)
        elif lang == 'english':
            english_refs.append(ref)
            english_hyps.append(hyp)
        else:  # mixed
            mixed_refs.append(ref)
            mixed_hyps.append(hyp)

    results = {}

    # Nepali CER
    if nepali_refs:
        cer_scores = [compute_cer(r, h)['cer'] for r, h in zip(nepali_refs, nepali_hyps)]
        results['nepali'] = {
            'cer': sum(cer_scores) / len(cer_scores),
            'count': len(nepali_refs),
        }

    # English WER
    if english_refs:
        wer_scores = [compute_wer(r, h)['wer'] for r, h in zip(english_refs, english_hyps)]
        results['english'] = {
            'wer': sum(wer_scores) / len(wer_scores),
            'count': len(english_refs),
        }

    # Mixed (use both CER and WER)
    if mixed_refs:
        cer_scores = [compute_cer(r, h)['cer'] for r, h in zip(mixed_refs, mixed_hyps)]
        wer_scores = [compute_wer(r, h)['wer'] for r, h in zip(mixed_refs, mixed_hyps)]
        results['mixed'] = {
            'cer': sum(cer_scores) / len(cer_scores),
            'wer': sum(wer_scores) / len(wer_scores),
            'count': len(mixed_refs),
        }

    return results


def compute_detailed_metrics(
    reference: str,
    hypothesis: str,
    language: str
) -> Dict:
    """
    Compute detailed metrics for a single prediction.

    Args:
        reference: Ground truth text
        hypothesis: Predicted text
        language: Language tag ('nepali', 'english', 'mixed')

    Returns:
        Dict with detailed metrics
    """
    result = {
        'reference': reference,
        'hypothesis': hypothesis,
        'language': language,
    }

    if language == 'nepali':
        cer_result = compute_cer(reference, hypothesis)
        result.update(cer_result)
    elif language == 'english':
        wer_result = compute_wer(reference, hypothesis)
        result.update(wer_result)
    else:  # mixed
        cer_result = compute_cer(reference, hypothesis)
        wer_result = compute_wer(reference, hypothesis)
        result.update(cer_result)
        result.update(wer_result)

    return result


def format_metrics_report(metrics: Dict) -> str:
    """
    Format metrics dict into a readable report.

    Args:
        metrics: Metrics dict from compute_metrics_batch

    Returns:
        Formatted string report
    """
    lines = ["=" * 50, "EVALUATION METRICS", "=" * 50]

    if 'nepali' in metrics:
        lines.append(f"\nNepali (Devanagari):")
        lines.append(f"  Samples: {metrics['nepali']['count']}")
        lines.append(f"  CER: {metrics['nepali']['cer']:.2%}")

    if 'english' in metrics:
        lines.append(f"\nEnglish:")
        lines.append(f"  Samples: {metrics['english']['count']}")
        lines.append(f"  WER: {metrics['english']['wer']:.2%}")

    if 'mixed' in metrics:
        lines.append(f"\nMixed (Code-switched):")
        lines.append(f"  Samples: {metrics['mixed']['count']}")
        lines.append(f"  CER: {metrics['mixed']['cer']:.2%}")
        lines.append(f"  WER: {metrics['mixed']['wer']:.2%}")

    lines.append("=" * 50)

    return "\n".join(lines)


class MetricsTracker:
    """Track metrics across multiple batches"""

    def __init__(self):
        self.nepali_cer_scores = []
        self.english_wer_scores = []
        self.mixed_cer_scores = []
        self.mixed_wer_scores = []

    def add_batch(
        self,
        references: List[str],
        hypotheses: List[str],
        languages: List[str]
    ):
        """Add a batch of predictions"""
        for ref, hyp, lang in zip(references, hypotheses, languages):
            if lang == 'nepali':
                cer = compute_cer(ref, hyp)['cer']
                self.nepali_cer_scores.append(cer)
            elif lang == 'english':
                wer = compute_wer(ref, hyp)['wer']
                self.english_wer_scores.append(wer)
            else:  # mixed
                cer = compute_cer(ref, hyp)['cer']
                wer = compute_wer(ref, hyp)['wer']
                self.mixed_cer_scores.append(cer)
                self.mixed_wer_scores.append(wer)

    def get_metrics(self) -> Dict:
        """Get aggregated metrics"""
        results = {}

        if self.nepali_cer_scores:
            results['nepali'] = {
                'cer': sum(self.nepali_cer_scores) / len(self.nepali_cer_scores),
                'count': len(self.nepali_cer_scores),
            }

        if self.english_wer_scores:
            results['english'] = {
                'wer': sum(self.english_wer_scores) / len(self.english_wer_scores),
                'count': len(self.english_wer_scores),
            }

        if self.mixed_cer_scores:
            results['mixed'] = {
                'cer': sum(self.mixed_cer_scores) / len(self.mixed_cer_scores),
                'wer': sum(self.mixed_wer_scores) / len(self.mixed_wer_scores),
                'count': len(self.mixed_cer_scores),
            }

        return results

    def reset(self):
        """Reset all scores"""
        self.nepali_cer_scores = []
        self.english_wer_scores = []
        self.mixed_cer_scores = []
        self.mixed_wer_scores = []
