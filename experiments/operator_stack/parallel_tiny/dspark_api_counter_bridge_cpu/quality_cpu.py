"""CPU text fingerprints and the repository's non-overlapping repetition rule.

These are cleanliness heuristics, not GSM8K, vision, or semantic accuracy proof.
The repetition algorithm follows tools/ctx_agent_probe.py: win=40, thresh=3,
cover=0.5. A separate token-window indicator is reported without replacing it.
"""
import re
import unicodedata

from .runtime_guards import token_list

_CTRL = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SURR = re.compile(r'[\ud800-\udfff]')


def repeat_loop(text, win=40, thresh=3, cover=0.5):
    if len(text) < win * thresh:
        return False
    seen = {}
    for i in range(0, len(text) - win + 1, win):
        fragment = text[i:i + win]
        if not fragment.strip():
            continue
        seen[fragment] = seen.get(fragment, 0) + 1
    if not seen:
        return False
    top = max(seen.values())
    return top >= thresh and top * win >= cover * len(text)


def quality(text, tokens):
    tokens = token_list(tokens)
    if type(text) is not str:
        raise ValueError('HTTP completion text is unavailable')
    fp = {'u_fffd': text.count('\ufffd'), 'nul': text.count('\x00'),
        'ctrl': len(_CTRL.findall(text)), 'surrogate': len(_SURR.findall(text)),
        'nonchar': sum(1 for ch in text if unicodedata.category(ch) == 'Cn')}
    char_repeat = repeat_loop(text)
    chunks = [tuple(tokens[i:i + 8]) for i in range(0, len(tokens) - 7, 8)]
    top = max((chunks.count(c) for c in set(chunks)), default=0)
    token_repeat = top >= 4 and top * 8 >= 0.5 * len(tokens)
    return {'nonempty': bool(text.strip()), 'fingerprints': fp,
        'repeat_loop': char_repeat, 'repeated_token_window': token_repeat,
        'clean': bool(text.strip()) and not any(fp.values()) and not char_repeat and not token_repeat,
        'quality_scope': 'CPU nonempty/fingerprints/repetition heuristics; semantic accuracy untested'}
