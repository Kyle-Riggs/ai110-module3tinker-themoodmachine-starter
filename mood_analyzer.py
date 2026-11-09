# mood_analyzer.py
"""
Rule based mood analyzer for short text snippets.

This class starts with very simple logic:
  - Preprocess the text
  - Look for positive and negative words
  - Compute a numeric score
  - Convert that score into a mood label
"""

import re
from typing import List, Dict, Tuple, Optional

from dataset import POSITIVE_WORDS, NEGATIVE_WORDS

# Words that flip the polarity of the sentiment word right after them.
NEGATORS = {"not", "no", "never", "dont", "don't", "isnt", "isn't", "wasnt", "wasn't", "cant", "can't", "wont", "won't", "didnt", "didn't", "aint", "ain't"}


class MoodAnalyzer:
    """
    A very simple, rule based mood classifier.
    """

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        # Use the default lists from dataset.py if none are provided.
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS

        # Store as sets for faster lookup.
        self.positive_words = set(w.lower() for w in positive_words)
        self.negative_words = set(w.lower() for w in negative_words)

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """
        Convert raw text into a list of tokens the model can work with.

        TODO: Improve this method.

        Right now, it does the minimum:
          - Strips leading and trailing whitespace
          - Converts everything to lowercase
          - Splits on spaces

        Ideas to improve:
          - Remove punctuation
          - Handle simple emojis separately (":)", ":-(", "🥲", "😂")
          - Normalize repeated characters ("soooo" -> "soo")
        """
        cleaned = text.strip().lower()
        # Collapse 3+ repeated characters down to 2 ("soooo" -> "soo").
        cleaned = re.sub(r"(.)\1{2,}", r"\1\1", cleaned)
        # Words (keeping apostrophes, e.g. "i'm") or any single non-space,
        # non-word symbol, which keeps emojis/punctuation as their own tokens.
        tokens = re.findall(r"[a-z0-9]+(?:'[a-z]+)*|[^\w\s]", cleaned)

        return tokens

    # ---------------------------------------------------------------------
    # Scoring logic
    # ---------------------------------------------------------------------

    def score_text(self, text: str) -> int:
        """
        Compute a numeric "mood score" for the given text.

        Positive words increase the score.
        Negative words decrease the score.

        TODO: You must choose AT LEAST ONE modeling improvement to implement.
        For example:
          - Handle simple negation such as "not happy" or "not bad"
          - Count how many times each word appears instead of just presence
          - Give some words higher weights than others (for example "hate" < "annoyed")
          - Treat emojis or slang (":)", "lol", "💀") as strong signals
        """
        score, _, _ = self._analyze(text)
        return score

    def _analyze(self, text: str) -> Tuple[int, List[str], List[str]]:
        """
        Score the text and return (score, positive_hits, negative_hits).

        Enhancement: simple negation handling. If a negator ("not", "never",
        "no", "don't", ...) appears right before a sentiment word, the word's
        polarity is flipped ("not happy" counts as negative).
        """
        tokens = self.preprocess(text)
        score = 0
        positive_hits: List[str] = []
        negative_hits: List[str] = []

        for i, token in enumerate(tokens):
            negated = i > 0 and tokens[i - 1] in NEGATORS
            if token in self.positive_words:
                is_positive = not negated
            elif token in self.negative_words:
                is_positive = negated
            else:
                continue

            label = f"not {token}" if negated else token
            if is_positive:
                score += 1
                positive_hits.append(label)
            else:
                score -= 1
                negative_hits.append(label)

        return score, positive_hits, negative_hits

    # ---------------------------------------------------------------------
    # Label prediction
    # ---------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """
        Turn the numeric score for a piece of text into a mood label.

        The default mapping is:
          - score > 0  -> "positive"
          - score < 0  -> "negative"
          - score == 0 -> "neutral"

        TODO: You can adjust this mapping if it makes sense for your model.
        For example:
          - Use different thresholds (for example score >= 2 to be "positive")
          - Add a "mixed" label for scores close to zero
        Just remember that whatever labels you return should match the labels
        you use in TRUE_LABELS in dataset.py if you care about accuracy.
        """
        score, positive_hits, negative_hits = self._analyze(text)
        if score > 0:
            return "positive"
        if score < 0:
            return "negative"
        # Score is 0: either signals cancelled out (mixed) or there were none.
        if positive_hits and negative_hits:
            return "mixed"
        return "neutral"

    # ---------------------------------------------------------------------
    # Explanations (optional but recommended)
    # ---------------------------------------------------------------------

    def explain(self, text: str) -> str:
        """
        Return a short string explaining WHY the model chose its label.

        TODO:
          - Look at the tokens and identify which ones counted as positive
            and which ones counted as negative.
          - Show the final score.
          - Return a short human readable explanation.

        Example explanation (your exact wording can be different):
          'Score = 2 (positive words: ["love", "great"]; negative words: [])'

        The current implementation is a placeholder so the code runs even
        before you implement it.
        """
        score, positive_hits, negative_hits = self._analyze(text)
        return (
            f"Score = {score} "
            f"(positive: {positive_hits or '[]'}, "
            f"negative: {negative_hits or '[]'})"
        )
