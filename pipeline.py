"""
pipeline.py
===========
Production-grade content-moderation guardrail pipeline.

Architecture (3 layers):
  Layer 1 → Regex filter (fast, rule-based)
  Layer 2 → Calibrated DistilBERT (ML, probabilistic)
  Layer 3 → Human review queue (edge-case escalation)
"""

import re
import os
import json
import pickle
import logging
from datetime import datetime
from typing import Optional, Dict, Any

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Layer 1: Regex Filter
# ─────────────────────────────────────────────────────────────

# Each category has >= the required number of patterns.
REGEX_PATTERNS: Dict[str, list] = {
    "direct_threats": [
        r"\bi('?ll| will| am going to| gonna| shall)\s+(kill|murder|shoot|stab|hurt|harm|destroy|eliminate|end)\s+you\b",
        r"\b(you('re|r| are)\s+dead(\s+meat)?|dead\s+man\s+walking)\b",
        r"\b(i('?ll| will)\s+(find|hunt|track)\s+you\s+(down)?)\b",
        r"\b(watch\s+your\s+back|you(\s+better)?\s+run)\b",
        r"\b(send\s+you\s+to\s+(the\s+)?hospital|put\s+you\s+in\s+(the\s+)?ground)\b",
        r"\b(threats?\s+to\s+(your\s+)?(life|family|kids|children))\b",
    ],
    "self_harm": [
        r"\b(kill\s+(my|your)self|end\s+(my|your)\s+life|take\s+(my|your)\s+own\s+life)\b",
        r"\b(want(s)?\s+to\s+die|want(s)?\s+to\s+be\s+dead|wish(es)?\s+(i|they)\s+were\s+dead)\b",
        r"\b(suicide\s+(methods?|how|plan|attempt)|how\s+to\s+commit\s+suicide)\b",
        r"\b(self[- ]harm|cut(ting)?\s+(my|your)self|hur(t|ting)\s+(my|your)self\s+on\s+purpose)\b",
        r"\b(overdos(e|ing)\s+on\s+purpose|swallow(ing)?\s+pills\s+to\s+die)\b",
    ],
    "doxxing": [
        r"\b(home\s+address|personal\s+address|lives?\s+at|street\s+address)\s*:?\s*\d",
        r"\b(phone\s*(number)?|cell|mobile)\s*:?\s*[\+\(]?\d[\d\s\-\(\)]{7,}",
        r"\bSSN\b|\bsocial\s+security\s*(number)?\b",
        r"\b(posting|sharing|releasing|publishing)\s+(personal|private|home)\s+(info|details|data|address)\b",
        r"\b(ip\s+address|ip\s+leak|dox(x)?(ing|ed)?)\b",
        r"\b(financial\s+records?|bank\s+account\s+number|credit\s+card\s+number)\b",
    ],
    "dehumanization": [
        r"\b(aren'?t?|are\s+not|isn'?t?|is\s+not)\s+(human|people|persons?)\b",
        r"\b(sub[- ]?human|less\s+than\s+human|not\s+fully\s+human)\b",
        r"\b(animals?|vermin|rats?|cockroach(es)?|parasites?|insects?)\s+(\(when referring to people\)|who\s+(should|must|need\s+to))\b",
        r"\b(exterminate|eradicate|cleanse|purge)\s+(the\s+)?(blacks?|jews?|muslims?|immigrants?|gays?|lgbtq)\b",
        r"\b(inferior\s+race|master\s+race|racial\s+hierarchy|genetic(ally)?\s+inferior)\b",
        r"\b(their\s+kind|those\s+people)\s+(don'?t?|do\s+not)\s+(deserve|belong)\b",
    ],
    "coordinated_harassment": [
        r"\b(everyone\s+(go|attack|report|flood|spam|dm|message))\b",
        r"\b(let'?s\s+(all\s+)?(target|harass|troll|brigade|dogpile)\s+@?\w+)\b",
        r"\b(raid(ing)?\s+(their|this|the)\s+(account|channel|stream|server|profile))\b",
        r"\b(mass\s+(report(ing)?|flag(ging)?|spam(ming)?))\b",
    ],
}

# Compile all patterns once at module load
_COMPILED_PATTERNS: Dict[str, list] = {
    cat: [re.compile(p, re.IGNORECASE | re.UNICODE) for p in patterns]
    for cat, patterns in REGEX_PATTERNS.items()
}


def regex_filter(text: str) -> Optional[str]:
    """
    Layer 1: Rule-based regex filter.

    Returns the category name (str) if a match is found, else None.
    """
    for category, compiled_list in _COMPILED_PATTERNS.items():
        for pattern in compiled_list:
            if pattern.search(text):
                logger.debug("Regex hit: category=%s", category)
                return category
    return None


# ─────────────────────────────────────────────────────────────
# Layer 2: Calibrated DistilBERT
# ─────────────────────────────────────────────────────────────

BLOCK_THRESHOLD = 0.6
ALLOW_THRESHOLD = 0.4


class CalibratedDistilBERT:
    """
    Wraps a fine-tuned DistilBERT checkpoint and applies isotonic calibration.
    Falls back to raw softmax probabilities if calibrator is not available.
    """

    def __init__(self, model_dir: str = "./model_checkpoint", calibrator_path: str = "./calibrator.pkl"):
        self.model_dir = model_dir
        self.calibrator_path = calibrator_path
        self.model = None
        self.tokenizer = None
        self.calibrator = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load()

    def _load(self):
        if not os.path.isdir(self.model_dir):
            logger.warning(
                "Model checkpoint not found at '%s'. "
                "Run part1.ipynb first to train and save the model.",
                self.model_dir,
            )
            return
        logger.info("Loading tokenizer and model from '%s'…", self.model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self.model.to(self.device)
        self.model.eval()

        if os.path.isfile(self.calibrator_path):
            with open(self.calibrator_path, "rb") as f:
                self.calibrator = pickle.load(f)
            logger.info("Calibrator loaded from '%s'.", self.calibrator_path)
        else:
            logger.info("No calibrator found — using raw softmax probabilities.")

    def predict_proba(self, text: str) -> float:
        """Return calibrated probability of TOXIC class."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "Model not loaded. Ensure model_checkpoint/ exists "
                "(run part1.ipynb first)."
            )
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=128,
            return_tensors="pt",
            padding=True,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self.model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        raw_toxic_prob = float(probs[1])

        if self.calibrator is not None:
            calibrated = self.calibrator.predict_proba([[raw_toxic_prob]])[0][1]
            return float(calibrated)
        return raw_toxic_prob

    def decide(self, prob: float) -> str:
        """Apply threshold logic."""
        if prob >= BLOCK_THRESHOLD:
            return "block"
        if prob <= ALLOW_THRESHOLD:
            return "allow"
        return "review"


# ─────────────────────────────────────────────────────────────
# Layer 3: Human Review Queue
# ─────────────────────────────────────────────────────────────

REVIEW_QUEUE_FILE = "./review_queue.jsonl"


def enqueue_for_review(text: str, prob: float, source: str = "model"):
    """Append item to the human review queue (JSONL file)."""
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "text": text[:500],        # truncate for storage
        "toxic_prob": round(prob, 4),
        "source": source,
        "status": "pending",
    }
    with open(REVIEW_QUEUE_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info("Enqueued for human review (prob=%.3f).", prob)


# ─────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────

class ModerationPipeline:
    """
    Three-layer content moderation guardrail pipeline.

    Usage
    -----
    >>> pipe = ModerationPipeline()
    >>> result = pipe.predict("Some comment text here")
    >>> print(result)
    """

    def __init__(
        self,
        model_dir: str = "./model_checkpoint",
        calibrator_path: str = "./calibrator.pkl",
    ):
        logger.info("Initialising ModerationPipeline…")
        self.ml_model = CalibratedDistilBERT(
            model_dir=model_dir,
            calibrator_path=calibrator_path,
        )

    # ── public interface ──────────────────────────────────────

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Run the full three-layer pipeline.

        Parameters
        ----------
        text : str
            Raw comment text.

        Returns
        -------
        dict with keys:
            decision  : "block" | "allow" | "review"
            layer     : 1 | 2 | 3
            category  : regex category name or None
            toxic_prob: float or None
            reason    : human-readable explanation
        """
        if not isinstance(text, str) or not text.strip():
            return self._result("allow", layer=0, reason="Empty or non-string input.")

        # ── Layer 1: Regex ────────────────────────────────────
        category = regex_filter(text)
        if category is not None:
            return self._result(
                "block",
                layer=1,
                category=category,
                reason=f"Matched regex rule: '{category}'.",
            )

        # ── Layer 2: ML Model ─────────────────────────────────
        try:
            prob = self.ml_model.predict_proba(text)
            decision = self.ml_model.decide(prob)
        except RuntimeError as exc:
            logger.error("ML model error: %s", exc)
            # Escalate to human review if model unavailable
            enqueue_for_review(text, prob=0.0, source="model_error")
            return self._result(
                "review",
                layer=3,
                reason="ML model unavailable — escalated to human review.",
            )

        if decision == "block":
            return self._result(
                "block",
                layer=2,
                toxic_prob=prob,
                reason=f"ML model score {prob:.3f} ≥ {BLOCK_THRESHOLD} (block threshold).",
            )
        if decision == "allow":
            return self._result(
                "allow",
                layer=2,
                toxic_prob=prob,
                reason=f"ML model score {prob:.3f} ≤ {ALLOW_THRESHOLD} (allow threshold).",
            )

        # ── Layer 3: Human Review ─────────────────────────────
        enqueue_for_review(text, prob=prob, source="model_uncertain")
        return self._result(
            "review",
            layer=3,
            toxic_prob=prob,
            reason=(
                f"ML model score {prob:.3f} is in uncertain zone "
                f"({ALLOW_THRESHOLD}<p<{BLOCK_THRESHOLD}) — escalated to human review."
            ),
        )

    # ── helpers ───────────────────────────────────────────────

    @staticmethod
    def _result(
        decision: str,
        layer: int = 2,
        category: Optional[str] = None,
        toxic_prob: Optional[float] = None,
        reason: str = "",
    ) -> Dict[str, Any]:
        return {
            "decision": decision,
            "layer": layer,
            "category": category,
            "toxic_prob": round(toxic_prob, 4) if toxic_prob is not None else None,
            "reason": reason,
        }

    def batch_predict(self, texts: list) -> list:
        """Run pipeline over a list of texts and return list of result dicts."""
        return [self.predict(t) for t in texts]


# ─────────────────────────────────────────────────────────────
# CLI entry-point (quick sanity check)
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    pipe = ModerationPipeline()

    test_cases = [
        "I will kill you dead man walking.",                    # Layer 1 hit
        "This is a wonderful day, I love everyone!",           # allow
        "You are subhuman vermin who should be exterminated.", # Layer 1 hit
        "I think this policy is somewhat problematic.",        # borderline → layer 2
    ]

    if len(sys.argv) > 1:
        test_cases = [" ".join(sys.argv[1:])]

    print("\n" + "=" * 70)
    print("  ModerationPipeline — Quick Test")
    print("=" * 70)
    for text in test_cases:
        result = pipe.predict(text)
        print(f"\nText   : {text[:80]}")
        print(f"Decision : {result['decision'].upper()}")
        print(f"Layer  : {result['layer']}")
        print(f"Reason : {result['reason']}")
        if result["toxic_prob"] is not None:
            print(f"P(toxic): {result['toxic_prob']:.4f}")
    print("=" * 70 + "\n")
