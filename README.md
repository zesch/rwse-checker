# rwse-checker

Real-word spelling errors (RWSEs) pose special challenges for detection methods, as they ‘hide’ in the form of another existing word and in many cases even fit in syntactically.
rwse-checker is a modern Transformer-based implementation of earlier probabilistic methods based on confusion sets.
It detects RWSEs with a good balance between missing errors and raising too many false alarms. 
The confusion sets are dynamically configurable, allowing teachers to easily adjust which errors trigger feedback.

## Example Usage

```
from rwse import RWSE_Checker 

checker = RWSE_Checker()
checker.set_confusion_sets([['their','there'],['to','too','two']])

print(checker.check("there", "I want to buy [MASK] cars."))
print(checker.check("too", "I want [MASK] buy their cars."))
```
which yields
```
('their', 0.003510827198624611)
('to', 0.9989504218101501)
```


## Citation
If you are using this tool, please cite
[Transformer-Based Real-Word Spelling Error Feedback with Configurable Confusion Sets](https://aclanthology.org/2025.bea-1.29/) (Zesch et al., BEA 2025)

The experimental code for this paper is found in [https://github.com/zesch/rwse-experiments](https://github.com/zesch/rwse-experiments)

```
@inproceedings{zesch-etal-2025-transformer,
    title = "Transformer-Based Real-Word Spelling Error Feedback with Configurable Confusion Sets",
    author = "Zesch, Torsten  and
      Gardner, Dominic  and
      Bexte, Marie",
    booktitle = "Proceedings of the 20th Workshop on Innovative Use of NLP for Building Educational Applications (BEA 2025)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.bea-1.29/",
    doi = "10.18653/v1/2025.bea-1.29",
    pages = "375--383",
    ISBN = "979-8-89176-270-1",
}
```
