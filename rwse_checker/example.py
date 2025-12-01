from rwse import RWSE_Checker 

checker = RWSE_Checker()
checker.set_confusion_sets([['their','there'],['to','too','two']])

print(checker.check("there", "I want to buy [MASK] cars."))
print(checker.check("too", "I want [MASK] buy their cars."))