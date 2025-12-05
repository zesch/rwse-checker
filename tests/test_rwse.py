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
