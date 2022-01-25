import random
import pickle
import numpy as np
import nltk

from nltk.stem import WordNetLemmatizer
import json
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.models import load_model

lemma=WordNetLemmatizer()
intents=json.loads(open('intents.json').read())


words=pickle.load(open('words.pkl','rb'))
classes=pickle.load(open('class.pkl','rb'))
model=load_model('pixis.model')

def clean_sentence(sentence):
    sentence_words=nltk.word_tokenize(sentence)
    sentence_words=[lemma.lemmatize(word)for word in sentence_words]
    return sentence_words
def bag(sentence):
    sentence_words=clean_sentence(sentence)
    bag=[0]*len(words)
    for w in sentence_words:
        for i,word in enumerate(words):
            if word==w:
                bag[i]=1
    return np.array(bag)
def predict_classes(sentence):
    bow=bag(sentence)
    res=model.predict(np.array([bow]))[0]
    error=0.25
    results=[[i,r] for i,r in enumerate(res) if r>error]
    results.sort(key=lambda x:x[1],reverse=True)
    return_list=[]
    print(results)
    for r in results:
        return_list.append({'intent':classes[r[0]],'probability': str(r[1])})
        print(return_list)
    return return_list
def get_response(intents_list, intents_json):
    tag=intents_list[0]['intent']
    print(intents_list)
    list_of_intents=intents_json['intents']
    for i in list_of_intents:
        if i['tag']==tag:
            result=random.choice(i['responses'])
            break
    return  result
print('Pixis is live!!')
while True:
    message=input("")
    ints=predict_classes(message)
    res=get_response(ints,intents)
    print(res)


