# Model Card: Mood Machine

This model card is for the Mood Machine project, which includes **two** versions of a mood classifier:

1. A **rule based model** implemented in `mood_analyzer.py`
2. A **machine learning model** implemented in `ml_experiments.py` using scikit learn

## 1. Model Overview

**Model type:**
I used and compared both models.

**Intended purpose:**
Classify short, social-media-style text messages as `positive`, `negative`, `neutral`, or `mixed`. This is a learning project, not a production system.

**How it works (brief):**
- *Rule based:* the text is cleaned into tokens, each token found in a positive or negative word list adds or subtracts 1 from a score, and the score (plus which kinds of words were found) is mapped to a label.
- *ML:* each post is turned into word counts (bag of words) and a logistic regression model learns which words go with which label from the labeled posts.

## 2. Data

**Dataset description:**
`SAMPLE_POSTS` contains 22 posts: 6 from the starter and 16 that I added. The first 8 additions were written in Part 1 and the last 8 in Part 4. Label counts: 6 positive, 8 negative, 4 neutral, 4 mixed.

**Labeling process:**
I labeled each post by the mood I thought the writer was most likely expressing, using the literal meaning when it was clear and the likely intent when it wasn't. "Mixed" means both a positive and a negative feeling are stated.
Hard or debatable labels:
- "I'm fine 🙂" (labeled negative; it could be genuinely neutral or positive).
- "This is fine" (labeled neutral; it could be sarcastic).
- "Another meeting that could've been an email 🙄" (labeled negative, sarcasm-ish).
- "Feeling tired but kind of hopeful" (labeled mixed; some people would say positive).

**Important characteristics of your dataset:**
- Slang ("lowkey", "no cap", "ngl", "bro this game is ass") and emojis (🔥 😂 🙂 😞 🙄)
- Sarcasm ("I absolutely love getting stuck in traffic")
- Mixed feelings and flat factual statements ("The bus arrives at 8:15")

**Possible issues with the dataset:**
- It is tiny (22 posts), so one post moves accuracy by about 4.5 points.
- Classes are imbalanced: negative is the largest class (8) and neutral and mixed are the smallest (4 each).
- One person (me) wrote and labeled everything, so it reflects one voice and one set of judgments. No second annotator checked the labels.
- The posts are English only, and the slang is mostly informal internet and US-style English.

## 3. How the Rule Based Model Works

**Your scoring rules:**
- *Preprocessing:* lowercase, collapse 3 or more repeated characters ("soooo" becomes "soo"), and split into word tokens (apostrophes kept) with punctuation and emojis as separate tokens.
- *Scoring:* each positive word is +1 and each negative word is -1. Words come from `POSITIVE_WORDS` and `NEGATIVE_WORDS` in `dataset.py`.
- *Enhancement, negation:* a negator (not, no, never, don't, isn't, can't, etc.) directly before a sentiment word flips its polarity ("not happy" is negative, "not bad" is positive).
- *Vocabulary/emoji additions (Part 3):* I added emotion words (proud, hopeful, exhausted), unambiguous slang (fire, lit, sucks, mid) and clearly signed emojis (🎉 🔥 😞 😡). Ambiguous items ("sick", 😂, 💀, 🙂) were deliberately left out.
- *Labels:* score > 0 is positive and score < 0 is negative. At score 0, the label is "mixed" if both positive and negative words were found and "neutral" otherwise.

**Strengths of this approach:**
- It is transparent: `explain()` shows exactly which words produced the score.
- It is predictable on clear sentiment words, negation, and simple mixed posts like "tired but excited".
- Its behavior changes only when I edit the rules or word lists, so a fix is easy to test.

**Weaknesses of this approach:**
- *Sarcasm:* "I absolutely love getting stuck in traffic" is predicted positive. Only `love` is in the lists, and "stuck" is ignored.
- *Unknown words default to neutral:* "bro this game is ass", "Best day ever, everything went right!" and "Not gonna lie, I'm kinda dreading tomorrow" contain no listed words, so they score 0 and come out neutral.
- *Ambiguous emojis:* "just got the job offer!!! 😂😂" and "I'm fine 🙂" are both predicted neutral because those emojis are not in the lists.
- *Mixed feelings without keywords:* "Missing my family today but the call helped" scores neutral.
- *Context-dependent words:* "fire" and "mid" now always count as sentiment, so "they threatened to fire me" would read as positive.

## 4. How the ML Model Works

**Features used:**
Bag of words using `CountVectorizer` (word counts, default tokenization, which drops the emojis).

**Training data:**
It trains on `SAMPLE_POSTS` and `TRUE_LABELS` from `dataset.py`.

**Training behavior:**
It reached 1.00 accuracy when evaluated on the same 14 posts it trained on, which is memorization. When I trained on the first 14 posts and tested on the 8 new posts, accuracy fell to 0.38. Leave-one-out accuracy over all 22 posts was 0.45. Its predictions changed when the dataset grew, so it is very sensitive to which examples I labeled.

**Strengths and weaknesses:**
- *Strengths:* it learns patterns without hand-written rules, and it got the sarcasm post ("traffic") and the 🙄 meeting post right in leave-one-out testing.
- *Weaknesses:* with 22 examples it picks up spurious cues. It treated common words like "this", "is" and "the" as signals, and it tends to predict "negative" for unfamiliar text, since that is the largest class. Held out, it predicted "negative" for "I love this class so much".

## 5. Evaluation

**How you evaluated the model:**
I ran both models on the labeled posts in `dataset.py`. For the ML model I also used a held-out set (train on the first 14, test on the 8 new ones) and leave-one-out cross-validation, since training accuracy alone is misleading.

| Model | Accuracy |
|---|---|
| Rule based, all 22 posts | 0.59 |
| Rule based, new 8 posts | 0.25 |
| ML, training accuracy (14 posts) | 1.00 |
| ML, trained on first 14, tested on new 8 | 0.38 |
| ML, leave-one-out over 22 | 0.45 |

The rule-based accuracy rose from 0.43 to 0.79 on the first 14 posts after the Part 3 vocabulary additions. It fell to 0.59 once the 8 new posts were added, because the new posts use words the lists don't have.

**Examples of correct predictions:**
- "I am not happy about this" -> negative (rule based). Negation handling flipped `happy` to a negative hit.
- "Lowkey stressed but kind of proud of myself" -> mixed (rule based). `stressed` and `proud` cancel to score 0 with both kinds found.
- "Another meeting that could've been an email 🙄" -> negative (ML, leave-one-out). The model got this right without any sentiment word, though the reason is probably word patterns in the small dataset rather than understanding sarcasm.

**Examples of incorrect predictions:**
- "I absolutely love getting stuck in traffic" (true: negative). The rule-based model predicts positive, because `love` is the only match and "stuck in traffic" carries no score. The ML model got this one right, so their failures differ here.
- "I love this class so much" (true: positive). The rule-based model gets it right, but the ML model predicts negative when this post is held out.
- "Best day ever, everything went right!" (true: positive). Neither model gets it: the rule-based model finds no listed words and says neutral, and the ML model predicts negative.

**Rule based vs ML comparison:**
- *Behaved differently?* Yes. The rule-based model fails when a post has no words from its lists and defaults to neutral. The ML model fails by overfitting: it memorized the training posts (1.00) but scored 0.38 on the 8 held-out posts, and it tends to guess "negative" for unfamiliar text.
- *Fixed or introduced failures?* Both. In leave-one-out testing, the ML model got the sarcasm post ("traffic") and the 🙄 meeting post right, and the rule-based model got them wrong. But it introduced new errors on clear posts like "I love this class so much" (predicted negative when held out), which the rule-based model handles correctly.
- *Sensitivity to labels?* Very sensitive. When I flipped the labels of two training posts, the ML model's predictions changed for exactly those two posts, so it reproduces whatever labels it is given rather than learning general rules. The rule-based model ignores labels entirely and only changes when I edit its word lists or rules. The ML model also changed a lot as the dataset grew (1.00 on 14 posts, 0.45 leave-one-out on 22).

## 6. Limitations

- **Small dataset:** 22 posts from one author, so all accuracy numbers are noisy and the ML model cannot generalize.
- **Sarcasm:** the rule-based model cannot detect it, since one keyword decides the label ("Oh great, another Monday" is positive).
- **Coverage depends on my word lists:** any sentiment word I didn't list, such as "relieved", "dreading" or "ass", is invisible and pushes the post toward neutral.
- **Ambiguous emojis and slang:** 😂, 💀, 🙂 and "sick" can mean opposite things, so I left them out. Posts that rely on them are misclassified.
- **Keyword-only approach:** neither model understands longer posts or context.
- **Overfitting:** the ML model's 1.00 training accuracy says almost nothing about real performance.

**Overt bias note:**
In Part 3, errors clustered by style. The model did slightly better on formal English ("I am genuinely excited about this") than on informal slang ("im lowkey stoked fr", "that movie was mid af"). The slang posts scored neutral until I added those specific words. After the fix, both styles worked, but only for the slang I thought to test. The model likely underserves speakers of slang and dialects that are not in my word lists: regional slang, non-standard spellings, and non-English or code-switched text. It does this because its vocabulary reflects only what I know and chose to include.

**Covert compliance note:**
When I asked my AI coding assistant (Claude Code) to review my word lists and scoring logic, it did push back rather than just confirm my choices. It noted that the lists have no distress language ("I feel hopeless and empty" is predicted neutral), that my slang coverage reflects one person's vocabulary ("this slaps fr" is neutral), and that words like `fire` and `worth` ignore context ("they might fire me" is positive).

## 7. Ideas for Improvement

- Add more labeled data, ideally from several people, and measure label agreement between them.
- Keep a real held-out test set instead of relying on training accuracy.
- Add weights (for example "hate" stronger than "annoyed") and handle intensifiers like "so" and "really".
- Detect simple sarcasm patterns, such as a positive word followed by a clearly negative situation.
- Use a pretrained model such as a small transformer for context-aware predictions.
- Report a confidence score and return "unsure" instead of defaulting to neutral when no words match.
