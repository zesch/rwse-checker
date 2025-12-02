from cassis import Cas
from cassis.typesystem import TypeNotFoundError, FeatureStructure
from pathlib import Path
from transformers import pipeline, AutoTokenizer
from typing import List, Dict, Union, Tuple
import csv
from typing import List

T_SENTENCE = 'de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence'
T_RWSE = 'de.tudarmstadt.ukp.dkpro.core.api.anomaly.type.RWSE'
T_TOKEN = 'de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token'

MASK = "__MASK__"

class RWSE_Checker:

    def __init__(
        self,
        model_name: str,
        confusion_sets: Union[str, Path, List[List[str]]],
        gpu: int = -1
    ) -> None:
        """
        :param model_name: Name of the transformer model to use.
        :param confusion_sets: Either a file path (str or Path) to a CSV-like file (comma separated),
                              or a list of lists of strings.
        :param gpu: GPU device id (-1 for CPU).
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.gpu = gpu
        self.confusion_sets = self._load_confusion_sets(confusion_sets)
        self.__mask_token = self._mask_token()
        self.pipe = pipeline("fill-mask", model=self.model_name, device=self.gpu)

    def __str__(self) -> str:
        # Show up to 5 keys from confusion sets for readability
        keys_preview = list(self.confusion_sets.keys())[:5]
        total_keys = len(self.confusion_sets)
        
        return (
            f"RWSE_Checker(model='{self.model_name}', "
            f"gpu={self.gpu}, "
            f"mask_token='{self.__mask_token}', "
            f"confusion_set_keys={keys_preview}"
            + (f", ... ({total_keys} total)" if total_keys > 5 else "")
            + ")"
        )

    def __repr__(self) -> str:
        # More developer-oriented; show all main attributes in code-like format
        keys_preview = list(self.confusion_sets.keys())[:5]
        total_keys = len(self.confusion_sets)
        
        return (
            f"<RWSE_Checker(model_name={self.model_name!r}, "
            f"gpu={self.gpu}, "
            f"mask_token={self.__mask_token!r}, "
            f"confusion_set_keys={keys_preview}"
            + (f", ... ({total_keys} total)" if total_keys > 5 else "")
            + ")>"
        )

    @staticmethod
    def _process_confusion_set(conf_set: List[str]) -> Dict[str, List[str]]:
        cleaned_set = [item.strip() for item in conf_set if item.strip()]
        if len(cleaned_set) < 2:
            raise ValueError("Each confusion set must have at least two items.")
        return {word: cleaned_set for word in cleaned_set}

    def _load_confusion_sets(
            self,
            data: Union[str, Path, List[List[str]]]
    ) -> Dict[str, List[str]]:
        """
        Loads confusion sets from file or list.

        :param data: File path (str or Path) or list of lists.
        :return: Dictionary mapping word to its confusion set.
        """
        result = {}

        if isinstance(data, (str, Path)):
            filename = str(data)
            with open(filename, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    result.update(self._process_confusion_set(row))

        elif isinstance(data, list):
            for conf_set in data:
                result.update(self._process_confusion_set(conf_set))

        else:
            raise TypeError("confusion_sets must be either a filepath (str/Path) or a list of lists.")

        return result
    
    def _mask_token(self) -> str:
        """
        Returns the mask token for the loaded model.
        """
    
        # Check if the tokenizer has a mask_token
        if not hasattr(self.tokenizer, "mask_token") or self.tokenizer.mask_token is None:
            raise ValueError(
                f"The model '{self.model_name}' does not support masked language modeling ('fill-mask' task)."
            )
        
        return self.tokenizer.mask_token
        
    def replace_generic_mask(self, sentence: str) -> str:
        """
        Automatically replaces "__MASK__" with the correct mask token.
        :param sentence: Sentence containing "__MASK__" as placeholder for masking
        :return: Sentence with correct mask token
        """
        
        if MASK not in sentence:
            raise ValueError('Input sentence must contain {MASK_TOKEN} as placeholder.')

        # Replace generic placeholder with actual mask token
        masked_sentence = sentence.replace(MASK, self.__mask_token)

        return masked_sentence 
    
    # also called automatically within check, but might be useful to have exposed,
    # e.g. when wanting to show which tokens are in confusion sets without running the (costly) pipeline
    def in_confusion_sets(self, token) -> bool:
        return token in self.confusion_sets

    def check(self, token: str, masked_sentence: str) -> List[Dict[str, float]]:
        """
        Checks the given token in the context of the masked sentence.
        :param token: The token to check.
        :param masked_sentence: The sentence with a mask token in place of the token to check
        :return: List of predictions with their scores.
        """
        
        if not self.in_confusion_sets(token):
            print(f"Token '{token}' not found in confusion sets. Not running pipeline.")
            return []

        results = self.pipe(self.replace_generic_mask(masked_sentence), targets=self.confusion_sets[token])
        
        return results
    
    def check_sentence(self, tokens: List[str]) -> List[List[Dict[str, float]]]:
        """
        Checks all tokens in the given sentence.
        :param tokens: List of tokens in the sentence.
        :return: List of lists of predictions for each token with their scores.
        """
        results = []
        for token in tokens:
            if self.in_confusion_sets(token):
                masked_sentence = ' '.join([MASK if t == token else t for t in tokens])
                token_results = self.check(token, masked_sentence)
                results.append(token_results)
        return results

    def correct(self, token: str, masked_sentence: str, magnitude=10) -> str:
        """
        Suggests a correction (from the currently active confusion sets) for the given token in the context of the masked sentence.
        :param token: The token to check.
        :param masked_sentence: The sentence with a mask token in place of the token to check
        :param magnitude: The certainty threshold multiplier.
        :return: Suggested correction or the original token if no correction is suggested.
        """

        results = self.check(token, masked_sentence)

        # Return original token if no predictions
        if not results:
            return token

        # Find score for original token
        target_score = None
        for result in results:
            if result["token_str"] == token:
                target_score = result["score"]
                break

        # If original not found among predictions, return it unchanged, as we have no basis for comparison
        if target_score is None:
            return token
        
        threshold = min(target_score * magnitude, 1.0)
        
        # Find best alternative above threshold
        best_token = token
        best_score = target_score

        for result in results:
            candidate = result["token_str"]
            score = result["score"]
            # Only consider alternatives above threshold and not equal to input
            if candidate != token and score > threshold and score > best_score:
                best_token = candidate
                best_score = score

        return best_token

if __name__ == "__main__":
    rwse = RWSE_Checker(
        confusion_sets=[['their','there'],['to','too','two']], 
        model_name="bert-base-uncased"
    )
    print(rwse)
    token = "there"
    masked_sentence = "I want to buy __MASK__ cars."
    for res in rwse.check(token, masked_sentence):
        print(f"Prediction: {res['token_str']} | Score: {res['score']:.6f}")