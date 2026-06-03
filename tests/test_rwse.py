import pytest
from rwse_checker.rwse import RWSE_Checker
from pathlib import Path
from cassis import Cas, load_cas_from_xmi
from cassis.typesystem import load_typesystem


def test_check():
    rwse = RWSE_Checker(
        confusion_sets=[['their','there'],['to','too','two']], 
        model_name="bert-base-uncased"
    )
    assert rwse.__str__() == "RWSE_Checker(model='bert-base-uncased', gpu=-1, mask_token='[MASK]', confusion_set_keys=['their', 'there', 'to', 'too', 'two'])"
    
    token = "there"
    masked_sentence = "I want to buy __MASK__ cars."
    
    assert rwse.check(token, masked_sentence) is not None

    for res in rwse.check(token, masked_sentence):
        print(f"Prediction: {res['token_str']} | Score: {res['score']:.6f}")

def test_correct():
    rwse = RWSE_Checker(
        confusion_sets=[['their','there'],['to','too','two']], 
        model_name="bert-base-uncased"
    )
    
    correction, certainty, _ = rwse.correct("there", "I want to buy __MASK__ cars.")
    assert correction == "their"
    assert certainty == pytest.approx(2.1128, 0.001)

    correction, certainty, _ = rwse.correct("too", "I want __MASK__ buy their cars.")
    assert correction == "to"
    assert certainty == pytest.approx(5.3800, 0.001)

def test_models():
    token = "there"
    masked_sentence = "I want to buy __MASK__ cars."
    
    models = ["bert-base-uncased", "roberta-base", "distilbert-base-uncased"]
    for model in models:
        rwse = RWSE_Checker(
            confusion_sets=[['their','there'],['to','too','two']],
            model_name=model
        )
        assert rwse.check(token, masked_sentence) is not None

        for res in rwse.check(token, masked_sentence):
            print(f"Prediction: {res['token_str']} | Score: {res['score']:.10f}")


# ---------------------------------------------------------------------------
# Regression tests for two issues the existing suite doesn't exercise.
#
# Both are marked xfail(strict=True): they assert the *desired* behavior of
# the existing check()/correct() methods, so they currently xfail (the
# suite stays green) and will turn into a hard failure — prompting removal
# of the marker — the moment the underlying issue is fixed.
#
# Why the existing tests miss these: every confusion word they use
# (their/there/to/too/two) is a single word-piece in all three test
# models, and the only correct() calls are correct("there", ...) and
# correct("too", ...) — neither of which is a substring of its
# alternatives, so the substring branch is never taken.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "check() can only score single-token candidates via the fill-mask "
        "`targets=` mechanism. 'borrows' tokenizes to ['borrow', '##s'], so "
        "Hugging Face drops it to its first sub-word ('borrow') and the real "
        "confusion-set member is never scored. (check_multi() handles this.)"
    ),
)
def test_check_scores_multi_subword_candidate():
    """A confusion-set member that is more than one word-piece must still be
    scored as itself. 'borrows' (= borrow + ##s in bert-base-uncased) vs
    'burrows'; in 'He always ___ money ...' the model should rank 'borrows'
    first. Today check() returns {'borrow', 'burrows'} — the multi-sub-word
    member is mangled — so it never even offers 'borrows'."""
    rwse = RWSE_Checker(
        confusion_sets=[['borrows', 'burrows']],
        model_name="bert-base-uncased",
    )
    masked = "He always __MASK__ money from his friends ."
    results = rwse.check("burrows", masked)

    returned = {r["token_str"] for r in results}
    assert returned == {"borrows", "burrows"}
    assert results[0]["token_str"] == "borrows"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "correct() reads the original token's score with a substring test "
        "(`search_token in result['sequence']`). When the original is a "
        "substring of an alternative ('to' in 'too'), it matches the wrong "
        "row, the magnitude threshold saturates, and the error is never "
        "flagged. Direction-asymmetric: correct('too', ...) works, "
        "correct('to', ...) does not."
    ),
)
def test_correct_flags_substring_prefix_direction():
    """correct() must flag to -> too even though 'to' is a substring of
    'too'. In 'The coffee is much ___ hot to drink .' the model strongly
    prefers 'too' (~0.998), yet correct('to', ...) returns 'to' today."""
    rwse = RWSE_Checker(
        confusion_sets=[['to', 'too', 'two']],
        model_name="bert-base-uncased",
    )
    masked = "The coffee is much __MASK__ hot to drink ."
    correction, _certainty, _oov = rwse.correct("to", masked)

    assert correction == "too"
