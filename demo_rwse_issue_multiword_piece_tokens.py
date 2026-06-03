#!/usr/bin/env python

"""Minimal demonstration of two RWSE-checker issues.

Test sentence (two real-word spelling errors):

    Peter hat gesagt, das er uns Gepäck zum Kaffee anbieten wird.

      das    -> dass    (subordinating conjunction; das/dass confusion)
      Gepäck -> Gebäck  ("luggage" vs "pastry" — pastry is what you offer
                         with coffee, so the model should prefer Gebäck)

It scores each error word two ways:

    check()        the current single-[MASK] / `targets=` method
    check_multi()  a proposed extension (this fork) that scores each
                   candidate as a span of one-or-more sub-word tokens

Findings the output illustrates:

  1. SINGLE-TOKEN LIMIT of check(): "Gepäck"/"Gebäck" each tokenize to
     more than one word-piece, so the fill-mask `targets=` mechanism
     cannot score them. Hugging Face warns
        "The specified target token `gebäck` does not exist in the model
         vocabulary. Replacing with `geb`."
     and the returned scores are meaningless (~1e-7). check_multi()
     scores them as spans and ranks "Gebäck" first.
     For the single-token pair das/dass, check() and check_multi() agree
     (check_multi is a strict generalisation).

  2. correct() SUBSTRING BUG (shown at the very bottom): when one
     confusion member is a substring of another ("das" ⊂ "dass"),
     correct() reads the original token's score off the wrong candidate
     (its `search_token in sequence` test matches "das" inside "dass"),
     so the magnitude threshold saturates and it never flags the error.
     The bug is DIRECTION-ASYMMETRIC: correct() flags dass->das (where
     the original "dass" is not a substring of the candidate "das"), but
     misses das->dass (where the original "das" IS a substring of
     "dass"). The bottom section shows both directions of the same pair.


    python demo_rwse_problems.py
"""

from pathlib import Path

from rwse_checker.rwse import RWSE_Checker, MASK

# Uncased German model — pairs with the case-insensitive confusion sets
# (the checker lowercases input in case_sensitive=False mode).
MODEL = "bert-base-german-dbmdz-uncased"
CONFUSION_SETS = str(
    Path(__file__).parent / "rwse_checker" / "data" / "de_sets_ci.txt"
)

SENTENCE = "Peter hat gesagt, das er uns Gepäck zum Kaffee anbieten wird."


def fmt(results) -> str:
    """One-line 'token=score  token=score' rendering of a results list."""
    if not results:
        return "(empty list)"
    return "   ".join(f"{r['token_str']}={r['score']:.6e}" for r in results)


def mask_in(sentence: str, target: str) -> str:
    """Return ``sentence`` with the first whitespace token equal to
    ``target`` replaced by the generic MASK placeholder."""
    out, replaced = [], False
    for w in sentence.split():
        if not replaced and w == target:
            out.append(MASK)
            replaced = True
        else:
            out.append(w)
    return " ".join(out)


def main() -> None:
    checker = RWSE_Checker(
        model_name=MODEL,
        confusion_sets=CONFUSION_SETS,
        case_sensitive=False,
    )

    print("Sentence:", SENTENCE)
    print()

    has_multi = hasattr(checker, "check_multi")
    for target in ["das", "Gepäck"]:
        masked = mask_in(SENTENCE, target)
        conf = checker.confusion_sets.get(target.lower())
        print("=" * 72)
        print(f"TARGET: {target!r}    confusion set: {conf}")
        print(f"masked: {masked}")
        print()
        print(f"  check()       : {fmt(checker.check(target, masked))}")
        if has_multi:
            print(f"  check_multi() : {fmt(checker.check_multi(target, masked))}")
        else:
            print("  check_multi() : <not available —  proposed "
                  "extension>")
        print()

    # ------------------------------------------------------------------
    # Separate issue: correct()'s substring bug is direction-asymmetric.
    # Same das/dass pair, two example sentences, opposite outcomes.
    # ------------------------------------------------------------------
    print("=" * 72)
    print("correct() — substring bug is DIRECTION-ASYMMETRIC (same das/dass pair)")
    print()

    cases = [
        # (sentence, error token, expected correction)
        ("Ich habe ihm dass Kabel gegeben.", "dass", "das"),
        (SENTENCE, "das", "dass"),
    ]
    for sentence, token, expected in cases:
        masked = mask_in(sentence, token)
        suggestion, certainty, _oov = checker.correct(token, masked)
        flagged = suggestion.lower() != token.lower()
        status = "flagged" if flagged else "NOT flagged"
        print(f"  {sentence}")
        print(f"    correct({token!r}) -> suggestion={suggestion!r}, "
              f"certainty={certainty:.2f}   (expected {expected!r}) [{status}]")
        print()

    print("  Both involve the das/dass confusion set. correct() catches "
          "dass->das,")
    print("  but for das->dass its `search_token in sequence` test finds "
          "'das'")
    print("  inside 'dass', mis-reads the original's score, and the "
          "threshold")
    print("  saturates — so the error is silently NOT flagged.")


if __name__ == "__main__":
    main()
