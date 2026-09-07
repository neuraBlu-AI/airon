"""
Getting a name out of a sentence.

Small and deliberately dumb. An LLM will do this properly in Phase 4, along
with everything else about the conversation; until then a handful of patterns
covers what people actually say when a robot asks their name, in the two
languages aiRon speaks.

The patterns are searched for rather than anchored, which is not tidiness but
error tolerance: whisper-base transcribed "Mein Name ist Elizabeth" as
"Weiname ist Elizabeth", and searching still finds "name ist Elizabeth" inside
that. Anchoring would have thrown the sentence away over a mangled first word.

What this cannot do is tell a name from any other noun. "Ich bin müde" parses
as the name "Müde", because at this level nothing here knows what a name is.
That is why the caller says the name back out loud before it is worth
anything - a human hearing "Nice to meet you, Müde" will correct it, and no
amount of pattern matching here would have caught it.
"""

from __future__ import annotations

import re

#: Ways people hand over a name. Searched in order, first match wins.
LEAD_INS = [
    r"name\s+(?:is|ist)\s+",
    r"\bi\s*(?:'m|’m|\s+am)\s+",
    r"\bich\s+hei(?:ß|ss)e\s+",
    r"\bich\s+bin\s+",
    r"\b(?:you\s+can\s+)?call\s+me\s+",
    r"\b(?:man\s+)?nenn(?:t|e)?\s+mich\s+",
    r"\b(?:it|this)\s*(?:'s|’s|\s+is)\s+",
    r"\bdas\s+ist\s+",
    r"\bhier\s+ist\s+",
]

#: A name-shaped word: letters, and the punctuation real names contain.
WORD = re.compile(r"^[^\W\d_][\w'’\-]{1,19}$", re.UNICODE)

#: Shaped like a name, but nobody is called this. Answers to "what is your
#: name?" that mean something other than a name.
NOT_NAMES = {
    "yes", "no", "yeah", "nope", "hi", "hello", "hey", "what", "who", "sorry",
    "nothing", "nobody", "none", "stop", "nevermind", "never", "maybe", "ok",
    "okay", "thanks", "thank", "please", "robot", "human", "someone", "me",
    "you", "it", "not", "dunno", "know",
    "ja", "nein", "hallo", "was", "wer", "wie", "nichts", "niemand", "halt",
    "aufhören", "vielleicht", "danke", "bitte", "mensch", "roboter", "ich",
    "du", "es", "nicht", "egal", "weiß",
}

#: Answers that mean "I would rather not say", which deserve a different reply
#: from an answer that simply could not be understood.
REFUSALS = {
    "no", "nope", "no thanks", "no thank you", "not telling", "rather not",
    "i'd rather not", "id rather not", "none of your business", "go away",
    "stop", "leave me alone", "never mind", "nevermind",
    "nein", "nein danke", "lieber nicht", "geht dich nichts an", "keine ahnung",
    "lass mich", "hör auf", "kein interesse",
}


def _clean(text: str) -> str:
    """Drop punctuation that is never part of a name, keeping case and length."""
    return re.sub(r"[^\w\s'’\-]", " ", text, flags=re.UNICODE).strip()


def _fold(text: str) -> str:
    """Lower case and single-spaced, for comparing against word lists."""
    return re.sub(r"\s+", " ", _clean(text).casefold())


def is_refusal(text: str) -> bool:
    """Did they decline, rather than simply not get understood?"""
    bare = _fold(text)
    return bare in REFUSALS or any(bare.startswith(r + " ") for r in REFUSALS)


def _tidy(word: str) -> str:
    """
    Capitalise a name without flattening one that is already capitalised.

    Each hyphenated part is treated separately, so "jean-luc" becomes
    "Jean-Luc", while "McDonald" is left exactly as it was heard.
    """
    return "-".join(part[0].upper() + part[1:] if part and part[0].islower() else part
                    for part in word.split("-"))


def parse_name(text: str) -> str | None:
    """
    The name in `text`, or None if there is not clearly one.

    Accepts a bare name ("Anna") or a name behind one of the usual lead-ins.
    Anything longer than two words with no lead-in is treated as conversation
    rather than an introduction.
    """
    if not text or is_refusal(text):
        return None

    # Matched case-insensitively against the ORIGINAL text, never a casefolded
    # copy. Casefolding German turns "heiße" into "heisse" and every offset
    # after it shifts by one, which silently ate the first letter of the name.
    remainder = None
    for pattern in LEAD_INS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            remainder = text[match.end():]
            break
    if remainder is None:
        remainder = text
        if len(_fold(text).split()) > 2:
            return None

    words = [w for w in re.split(r"[\s,]+", _clean(remainder)) if w][:2]
    if not words:
        return None

    kept = []
    for word in words:
        word = word.strip("'’-")
        if not WORD.match(word) or word.casefold() in NOT_NAMES:
            break
        kept.append(_tidy(word))
    if not kept:
        return None
    return " ".join(kept)
