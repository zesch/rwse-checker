from cassis import Cas
from cassis.typesystem import TypeNotFoundError
from transformers import pipeline
import csv


T_SENTENCE = 'de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence'
T_RWSE = 'de.tudarmstadt.ukp.dkpro.core.api.anomaly.type.RWSE'
T_TOKEN = 'de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token'

class RWSE_Checker:

    def __init__(self, gpu=-1, language='en') -> None:
        # TODO make language aware
        # TODO make model configurable
        if language == 'en':
            self.pipe = pipeline("fill-mask", model="bert-base-cased", device=gpu)
        elif language == 'de':
            self.pipe = pipeline("fill-mask", model="bert-base-multilingual-cased", device=gpu)
        # self.pipe = pipeline("fill-mask", model="answerdotai/ModernBERT-base", torch_dtype=torch.bfloat16, device=gpu)

        self.confusion_sets = None
    
    def _create_masked_sentence(self, cas, i, tokens):
        text = cas.sofa_string

        token = tokens[i]

        prefix_start = tokens[0].begin
        prefix_end = tokens[0].begin
        if i > 0:
            prefix_end = tokens[i-1].end

        suffix_start = tokens[len(tokens)-1].end
        suffix_end = tokens[len(tokens)-1].end
        if i < len(tokens)-1:
            suffix_start = tokens[i+1].begin

        # get string from start to token and after token to end
        prefix = text[prefix_start:prefix_end]
        suffix = text[suffix_start:suffix_end]

        return prefix + " [MASK] " + suffix

    def set_confusion_sets(self, input_data) -> None:
        '''

        :param input_data: a file path string pointing to a csv file containing a list of confusion sets;
        a python list of confusion sets or a dictionary of confusion sets where each item is a word with
        a corresponding confusion set.
        '''
        def load_confusion_sets_from_file(file_path):
            result = {}
            with open(file_path, 'r') as file:
                for row in csv.reader(file):
                    for key in row:
                        result[key] = list(row)
            return result

        def load_confusion_sets_from_list(input_list):
            result = {}
            for input_set in input_list:
                for item in input_set:
                    result[item] = list(input_set)
                    if len(result[item]) < 2:
                        raise AttributeError
            return result

        def load_confusion_sets_from_dict(input_dict):
            result = dict()
            for key, value in input_dict.items():
                new_values = [item.strip() for item in value]
                if len(new_values) < 2:
                    raise AttributeError
                result[key.strip()] = new_values
            return result

        input_map = {
            str: load_confusion_sets_from_file,
            list: load_confusion_sets_from_list,
            dict: load_confusion_sets_from_dict
        }

        try:
            input_type = type(input_data)
            mapping = input_map.get(input_type)
            self.confusion_sets = mapping(input_data)
        except TypeError:
            print(f'Error: {input_type} not supported as confusion sets input.')
        except AttributeError:
            print('Confusion set input is malformed. A file should contain more than one item per line '
                'and a list or dictionary more than one item per entry.')


    # also called automatically within check, but might be useful to have exposed,
    # e.g. when wanting to show which tokens are in confusion sets without running the (costly) pipeline
    def needs_checking(self, token):
        return token in self.confusion_sets

    def check_cas(self, cas: Cas, ts, magnitude=10):
        try:
            RWSE = ts.get_type(T_RWSE)
        except TypeNotFoundError:
            print("RWSE type not found")
            ts.create_type(T_RWSE)
            RWSE = ts.get_type(T_RWSE)

        for sentence in cas.select(T_SENTENCE):
            tokens = [token for token in cas.select_covered(T_TOKEN, sentence)]
            for i in range(len(tokens)):
                masked_sentence = self._create_masked_sentence(cas, i, tokens)

                token_str = tokens[i].get_covered_text()

                correct_token_str, correct_token_crt = self.check(token_str, masked_sentence=masked_sentence, magnitude=magnitude)
                if token_str.lower() != correct_token_str.lower():
                    # TODO set correction feature
                    cas_rwse = RWSE(begin=tokens[i].begin, end=tokens[i].end, suggestion=correct_token_str,
                                    certainty=correct_token_crt)
                    cas.add(cas_rwse)
                    # print(cas_rwse)
        # return cas

    def check(self, token: str, masked_sentence: str, magnitude=10, return_all=False) -> (str, float):
        # if no need to check return token
        if not self.needs_checking(token):
            return token, None

        results = self.pipe(masked_sentence, targets=self.confusion_sets[token])
        if return_all:
            return results
        return self.evaluate(token, results, magnitude)

    def evaluate(self, token, results, magnitude) -> (str, float):
        # In order to be sure about a change, we should be orders of magnitude more probable
        correct_token = token
        highest_prob = 0.0
        target_prob = 0.0
        for result in results:
            if result["token_str"] == token:
                target_prob = min(result["score"] * magnitude, 1.0)
                highest_prob = result["score"]
        for result in results:
            # print(f'{result["token_str"]}: {result["score"]}')
            if result["score"] > target_prob and result["score"] > highest_prob:
                highest_prob = result["score"]
                correct_token = result["token_str"]
        return correct_token, highest_prob

if __name__ == "__main__":
    rwse = RWSE_Checker()
    rwse.set_confusion_sets('test_confusion_sets.csv')
    token = "there"
    masked_sentence = "I want to buy [MASK] cars."
    correct_token, certainty = rwse.check(token, masked_sentence)
    print(token, correct_token, certainty)