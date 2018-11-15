# -*- coding: utf-8 -*-
from naive_bayes_classifier import tokenizer
from naive_bayes_classifier.trainer import Trainer
from naive_bayes_classifier.classifier import Classifier

newsTrainer = Trainer(tokenizer.Tokenizer(stop_words = [], signs_to_remove = ["?!#%&"]))

newsSet =[
    {'text': 'not to eat too much is not enough to lose weight', 'category': 'health'},
    {'text': 'Russia is trying to invade Ukraine', 'category': 'politics'},
    {'text': 'do not neglect exercise', 'category': 'health'},
    {'text': 'Syria is the main issue, Obama says', 'category': 'politics'},
    {'text': 'eat to lose weight', 'category': 'health'},
    {'text': 'you should not eat much', 'category': 'health'}
]

for news in newsSet:
    newsTrainer.train(news['text'], news['category'])

newsClassifier = Classifier(newsTrainer.data, tokenizer.Tokenizer(stop_words = [], signs_to_remove = ["?!#%&"]))

unknownInstance = "Russia is better then Ukraine"
classification = newsClassifier.classify(unknownInstance)

print "Введенное предложение: " + unknownInstance
print classification