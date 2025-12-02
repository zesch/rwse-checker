from rwse import RWSE_Checker, MASK
from pathlib import Path
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

checker = RWSE_Checker(
    confusion_sets=[['their','there'],['to','too','two']], 
    model_name="bert-base-uncased"
)

print("Example predictions for 'there' and 'too':")
for result in checker.check("there", "I want to buy __MASK__ cars."):
    print(f"Prediction: {result['token_str']} | Score: {result['score']:.10f}")

for result in checker.check("too", f"I want {MASK} buy their cars."):
    print(f"Prediction: {result['token_str']} | Score: {result['score']:.10f}")


# check whole sentence
print("\nChecking whole sentence:")
for token_res in checker.check_sentence(['I', 'want', 'too', 'buy', 'there', 'cars', '.']):
    for result in token_res:
        print(f"Prediction: {result['token_str']} | Score: {result['score']:.10f}")

# setting magnitude parameter to control certainty threshold
print("\nCorrections with different magnitude settings:")
print(checker.correct("there", "I want to buy __MASK__ cars.")) # expected output: "their"
print(checker.correct("there", "I want to buy __MASK__ cars.", magnitude=1000)) # expected output: "there", as their is less than 100x more likely

# load confusion set from file and use different model
print("\nUsing confusion sets from file and different model:")
confset_path = Path(__file__).parent / "data" / "en_sets_ci.txt"
check_more = RWSE_Checker(
    confusion_sets=confset_path, 
    model_name="roberta-base"
)

for result in check_more.check("save", "I want to buy a __MASK__ car."):
    print(f"Prediction: {result['token_str']} | Score: {result['score']:.10f}")