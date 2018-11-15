from __future__ import division
import operator
from functools import reduce

from naiveBayesClassifier.ExceptionNotSeen import NotSeen


class Classifier(object):
    def __init__(self, trainedData, tokenizer):
        super(Classifier, self).__init__()
        self.data = trainedData
        self.tokenizer = tokenizer
        self.defaultProb = 0.000000001

    # ali ata bak
    def classify(self, text):
        
        documentCount = self.data.getDocCount()
        classes = self.data.getClasses()

        tokens = list(set(self.tokenizer.tokenize(text)))
        
        probsOfClasses = {}

        for className in classes:

            tokensProbs = [self.getTokenProb(token, className) for token in tokens]
            try:
                tokenSetProb = reduce(lambda a,b: a*b, (i for i in tokensProbs if i) ) 
            except:
                tokenSetProb = 0
            
            probsOfClasses[className] = tokenSetProb * self.getPrior(className)
        
        return sorted(probsOfClasses.items(), 
            key=operator.itemgetter(1), 
            reverse=True)


    def getPrior(self, className):
        return self.data.getClassDocCount(className) /  self.data.getDocCount()

    def getTokenProb(self, token, className):
        classDocumentCount = self.data.getClassDocCount(className)
        try:
            tokenFrequency = self.data.getFrequency(token, className)
        except NotSeen as e:
            return None

        if tokenFrequency is None:
            return self.defaultProb

        probablity = tokenFrequency / classDocumentCount
        return probablity
