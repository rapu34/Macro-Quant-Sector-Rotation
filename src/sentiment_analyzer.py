"""
Sentiment Overlay: FinBERT-based news sentiment for macro/sector signals.

Analyzes headlines at rebalance time; caches results to avoid repeated inference.
Output: 0~1 Sentiment Score (mergeable with macro dataframe).
"""

import json
from pathlib import Path
from typing import Optional

import pandas as pd

_CACHE: dict[str, float] = {}
_FINBERT_MODEL = None


def _get_finbert():
    """Lazy-load FinBERT (heavy; load only when needed)."""
    global _FINBERT_MODEL
    if _FINBERT_MODEL is None:
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            name = "ProsusAI/finbert"
            tokenizer = AutoTokenizer.from_pretrained(name)
            model = AutoModelForSequenceClassification.from_pretrained(name)
            _FINBERT_MODEL = (tokenizer, model, torch)
        except Exception as e:
            raise ImportError(
                "FinBERT requires: pip install transformers torch. "
                f"Original error: {e}"
            ) from e
    return _FINBERT_MODEL


def _score_text(text: str) -> float:
    """
    Run FinBERT on text. Returns 0~1 Sentiment Score.
    positive -> 1, negative -> 0, neutral -> 0.5.
    """
    if not text or not str(text).strip():
        return 0.5
    tokenizer, model, torch = _get_finbert()
    inputs = tokenizer(text[:512], return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        out = model(**inputs)
    probs = torch.softmax(out.logits, dim=1).squeeze()
    if probs.dim() == 0:
        probs = probs.unsqueeze(0)
    probs = probs.tolist()
    # FinBERT: 0=positive, 1=negative, 2=neutral
    pos = probs[0] if len(probs) > 0 else 0.33
    neg = probs[1] if len(probs) > 1 else 0.33
    return float(pos) / (float(pos) + float(neg) + 1e-8)


def analyze_headlines(headlines: list[dict], cache_key: Optional[str] = None) -> float:
    """
    Compute aggregate Sentiment Score (0~1) from list of {title, ...}.
    Caches by cache_key to avoid re-running inference.
    """
    if cache_key and cache_key in _CACHE:
        return _CACHE[cache_key]
    if not headlines:
        return 0.5
    texts = [h.get("title", h.get("link", "")) for h in headlines if isinstance(h, dict)]
    if not texts:
        return 0.5
    try:
        scores = [_score_text(t) for t in texts if t]
        agg = sum(scores) / len(scores) if scores else 0.5
    except Exception:
        agg = 0.5
    if cache_key:
        _CACHE[cache_key] = agg
    return agg


def merge_sentiment_to_macro(
    macro_df: pd.DataFrame,
    sentiment_series: pd.Series,
) -> pd.DataFrame:
    """
    Merge sentiment (0~1) into macro dataframe by date.
    sentiment_series: index=date, values=0~1.
    """
    macro_df = macro_df.copy()
    macro_df.index = pd.to_datetime(macro_df.index)
    sent = sentiment_series.reindex(macro_df.index).ffill().bfill()
    macro_df["sentiment_score"] = sent
    return macro_df
