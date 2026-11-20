"""Area B - minimum edit distance + lexicon matcher (rule baseline).

A token is flagged when it is within a small Levenshtein distance of a lexicon entry
(after normalisation and ASCII folding), or when it starts with a long entry
(Turkish is agglutinative: "salak" -> "salaksın", "şerefsiz" -> "şerefsizler").
Entries written as "=word" must match the whole token exactly (short / ambiguous words).
"""
from functools import lru_cache

from .normalize import fold_ascii, normalize

# Hand-curated list of Turkish insults / profanity, written in ASCII-folded form.
LEXICON = [
    # exact-only (short or ambiguous)
    "=amk", "=amq", "=aq", "=mk", "=mq", "=oc", "=pic", "=got", "=mal", "=it", "=am", "=sik", "=bok",
    "=pust", "=esek", "=anani", "=ananin", "=anasini", "=bacini", "=avradini", "=boka",
    # fuzzy / prefix entries
    "amina", "aminakoy", "amcik", "siktir", "hassiktir", "sikerim", "sikeyim", "sikik", "sikis", "sikim",
    "sokayim", "orospu", "orosbu", "pezevenk", "kahpe", "yavsak", "serefsiz", "gerizekali", "salak",
    "aptal", "dangalak", "ahmak", "beyinsiz", "ibne", "ipne", "gotveren", "gotunu", "gotun", "kaltak",
    "surtuk", "fahise", "haysiyetsiz", "namussuz", "ahlaksiz", "karaktersiz", "pislik", "hiyar",
    "embesil", "mankafa", "angut", "gavat", "kevase", "yarrak", "yarak", "tasak", "dalyarak",
    "sapik", "sictim", "sicayim", "boktan", "dingil", "keriz", "enayi", "lavuk", "zibidi",
    "terbiyesiz", "soysuz", "utanmaz", "asagilik", "kodumun", "belasini", "belanizi", "yalaka",
    "cahil", "mankurt",
]


def levenshtein(a: str, b: str) -> int:
    """Minimum edit distance with unit costs for insertion, deletion and substitution (DP table)."""
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1,              # deletion
                         cur[j - 1] + 1,           # insertion
                         prev[j - 1] + (ca != cb))  # substitution
        prev = cur
    return prev[-1]


def max_distance(entry: str) -> int:
    n = len(entry)
    return 0 if n <= 3 else (1 if n <= 7 else 2)


class LexiconMatcher:
    def __init__(self, entries=LEXICON):
        self.exact, self.fuzzy = set(), []
        for e in entries:
            e = fold_ascii(e.replace(" ", ""))
            if e.startswith("="):
                self.exact.add(e[1:])
            else:
                self.fuzzy.append(e)

    @lru_cache(maxsize=100_000)
    def token_score(self, token: str):
        """Return (score in [0,1], matched entry or None) for one normalised, folded token."""
        if len(token) < 2:
            return 0.0, None
        if token in self.exact:
            return 1.0, token
        best = (0.0, None)
        for e in self.fuzzy:
            thr = max_distance(e)
            if token.startswith(e):
                return 1.0, e
            if abs(len(token) - len(e)) <= thr:
                d = levenshtein(token, e)
            elif len(e) >= 6 and len(token) > len(e):    # fuzzy prefix + suffix, e.g. "serefsizler"
                d = levenshtein(token[:len(e)], e)
            else:
                continue
            if d <= thr:
                s = 1.0 - d / (len(e) + 1)
                if s > best[0]:
                    best = (s, e)
        return best

    def matches(self, text: str):
        """List of (normalised token, matched entry, score) in a message."""
        toks = fold_ascii(normalize(text, keep_emoji=False)).split()
        out = []
        for t in toks:
            s, e = self.token_score(t)
            if e:
                out.append((t, e, s))
        return out

    def score(self, text: str) -> float:
        """Message-level offensiveness score = best token score (0 if nothing matched)."""
        m = self.matches(text)
        return max((s for _, _, s in m), default=0.0)

    def mask(self, text: str) -> str:
        """Mask the original words that match the lexicon, keeping their first letter."""
        out = []
        for word in text.split(" "):
            norm = fold_ascii(normalize(word, keep_emoji=False)).split()
            if any(self.token_score(t)[1] for t in norm):
                letters = [i for i, c in enumerate(word) if c.isalnum() or c in "@$*"]
                w = list(word)
                for i in letters[1:]:
                    w[i] = "*"
                out.append("".join(w))
            else:
                out.append(word)
        return " ".join(out)


if __name__ == "__main__":
    m = LexiconMatcher()
    for s in ["Ş3r3fs1zler hepiniz", "sen ne salaksın ya", "eziyet çekiyoruz", "a.p.t.a.l herif", "Ananız nasıl?",
              "gerizekalııı", "malzeme aldım", "bu iş mal gibi"]:
        print(s, "->", m.matches(s), "|", m.mask(s))
    print(levenshtein("kitten", "sitting"), levenshtein("şerefsiz", "serefsiz"))
