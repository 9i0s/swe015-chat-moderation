"""Area B - regex normalisation of noisy Turkish chat text.

Two levels are used in the system:
  * light_normalize(): for the Transformer (keeps casing/punctuation, which BERT was pretrained on)
  * normalize():       for the classical / word2vec models and the lexicon matcher
                       (Turkish-aware lower-casing, leetspeak, separators, repeated letters ...)
"""
import re
import unicodedata

USER_RE = re.compile(r"(?:@USER\b[\s,:]*)+", re.IGNORECASE)
URL_RE = re.compile(r"(?:https?://|www\.)\S+|\bURL\b", re.IGNORECASE)
WS_RE = re.compile(r"\s+")
REPEAT3_RE = re.compile(r"(.)\1{2,}")            # 3+ identical chars
REPEAT2_RE = re.compile(r"([^\W\d_])\1+")      # letter runs, used for the strict variant
# leetspeak digits/symbols that appear *inside* words, e.g. "s1kt1r", "@pt@l", "$alak"
LEET = {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b", "@": "a", "$": "s", "!": "i", "€": "e"}
LEET_IN_WORD_RE = re.compile(r"(?<=[^\W\d_])[013457@$€8!]+(?=[^\W\d_])|(?<=\s)[@$](?=[^\W\d_]{2,})|^[@$](?=[^\W\d_]{2,})")
# letters separated by dots/stars/dashes/spaces: "s.i.k.t.i.r", "a m k", "o*ç"
SEP_WORD_RE = re.compile(r"\b(?:[^\W\d_][.*\-_ ]){2,}[^\W\d_]\b")
STAR_IN_WORD_RE = re.compile(r"(?<=[^\W\d_])[*.\-_]+(?=[^\W\d_])")
NON_WORD_RE = re.compile(r"[^\w\s]")  # punctuation (emoji are also removed here; kept separately)
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF☀-➿]")

TR_FOLD = str.maketrans("çğıöşüâîû", "cgiosuaiu")


def tr_lower(text: str) -> str:
    """Turkish-aware lower-casing: I -> ı and İ -> i (Python's str.lower gets these wrong)."""
    return text.replace("I", "ı").replace("İ", "i").lower()


def fold_ascii(text: str) -> str:
    """Remove Turkish diacritics (people often type without them: 'serefsiz' for 'şerefsiz')."""
    return text.translate(TR_FOLD)


def light_normalize(text: str) -> str:
    """Normalisation in front of the Transformer."""
    text = unicodedata.normalize("NFC", str(text))
    text = USER_RE.sub("@USER ", text)
    text = URL_RE.sub("URL", text)
    text = REPEAT3_RE.sub(r"\1\1", text)           # "çoooook" -> "çook", "!!!!!" -> "!!"
    return WS_RE.sub(" ", text).strip()


def _undo_leet(match: re.Match) -> str:
    return "".join(LEET.get(c, c) for c in match.group(0))


def deobfuscate(text: str) -> str:
    """Token-level de-obfuscation in front of the Transformer (used at inference time).

    Only words that are obfuscated (leetspeak inside a word, or single letters separated by dots,
    dashes, underscores or spaces) are rewritten, so the rest of the message keeps the casing and
    punctuation BERTurk was pretrained on. "S1KT1R git" -> "siktir git", "a.m.k" -> "amk".
    Star-censored words ("s*ktir") are left alone: removing the star made validation macro-F1 worse
    (FP32: 0.7835 vs 0.787), because the fine-tuned model has learned that self-censoring itself signals
    an offensive tweet.
    """
    text = SEP_WORD_RE.sub(lambda m: re.sub(r"[.\-_ ]", "", m.group(0)), text)   # a m k / s.i.k -> amk / sik
    out = []
    for tok in text.split(" "):
        protected = tok.startswith(("@", "#")) or "://" in tok or tok == "URL" or ("@" in tok and "." in tok)
        if not protected and DEOB_LEET_RE.search(tok):
            tok = DEOB_LEET_RE.sub(_undo_leet, tr_lower(tok))
        out.append(tok)
    return " ".join(out)


# stricter pattern for deobfuscate(): leet only between letters, or "$" starting a word
DEOB_LEET_RE = re.compile(r"(?<=[^\W\d_])[013457@$€8](?=[^\W\d_])|(?:(?<=\s)|^)\$(?=[^\W\d_]{2,})")


def normalize(text: str, keep_emoji: bool = True) -> str:
    """Full normalisation for tf-idf, word2vec and lexicon matching."""
    text = unicodedata.normalize("NFC", str(text))
    text = USER_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = tr_lower(text)
    text = LEET_IN_WORD_RE.sub(_undo_leet, text)                          # s1kt1r -> siktir
    text = SEP_WORD_RE.sub(lambda m: re.sub(r"[.*\-_ ]", "", m.group(0)), text)  # s.i.k -> sik
    text = STAR_IN_WORD_RE.sub("", text)                                  # o*ç -> oç
    text = REPEAT3_RE.sub(r"\1", text)                                    # amkkkk -> amk
    emojis = " ".join(EMOJI_RE.findall(text)) if keep_emoji else ""
    text = NON_WORD_RE.sub(" ", text)
    text = re.sub(r"\d+", " ", text)                                      # stand-alone numbers
    text = WS_RE.sub(" ", text + " " + emojis).strip()
    return text


def tokenize(text: str) -> list:
    """Whitespace tokenisation of already-normalised text."""
    return text.split()


if __name__ == "__main__":
    for s in ["@USER @USER S1KT1R git lan GERİZEKALIIIII!!!", "s.i.k.t.i.r be", "Bu ne $alak iş ya 😂😂",
              "İYİ akşamlar www.site.com", "o*ç herif", "a m k ya"]:
        print(repr(s), "->", repr(normalize(s)), "|", repr(light_normalize(s)))
