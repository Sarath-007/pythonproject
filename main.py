import random
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import  Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop


filepath=tf.keras.utils.get_file('shakespeare.txt','https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text=open(filepath,'rb').read().decode(encoding='utf-8').lower()

#selecting
text=text[200000:800000]
characters=sorted(set(text))

char_to_index=dict((c,i) for i,c in enumerate(characters))
index_to_char=dict((i,c) for i,c in enumerate(characters))

SEQ_LENGTH=40
STEP_SIZE=3
##  training data
sentences=[]
next_characters=[]

##for i in range(0, len(text)-SEQ_LENGTH, STEP_SIZE):
  ##  sentences.append(text[i:i+SEQ_LENGTH])
   ## next_characters.append(text[i+SEQ_LENGTH])

## turning it into numerical data
##x=np.zeros((len(sentences),SEQ_LENGTH,len(characters)),dtype=np.bool)
##y=np.zeros((len(sentences), len(characters)),dtype=np.bool)

## filling these arrays

##for i, sentence in enumerate(sentences):
  ##  for t,character in enumerate(sentence):
    ##    x[i,t,char_to_index[character]]=1
   ## y[i,char_to_index[next_characters[i]]]=1

## model building
...
##model=Sequential()
##model.add(LSTM(128,input_shape=(SEQ_LENGTH,len(characters))))
##model.add(Dense(len(characters)))
##model.add(Activation('softmax'))

##model.compile(loss='categorical_crossentropy',optimizer=RMSprop(lr=0.01))
##model.fit(x,y,batch_size=256,epochs=5)

##model.save('textgenerator.model')


model=tf.keras.models.load_model('textgenerator.model')


def sample(preds, temperature):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(length,temperature):
    start_index=random.randint(0,len(text)-SEQ_LENGTH-1)
    generated=''
    sentence=text[start_index:start_index+SEQ_LENGTH]
    print('seed text:'+ sentence)
    generated+=sentence

    for i in range(length):
        x=np.zeros((1,SEQ_LENGTH,len(characters)))
        for t, character in enumerate(sentence):
            x[0,t,char_to_index[character]]=1
        predictions=model.predict(x,verbose=0)[0]
        next_index=sample(predictions,temperature)
        next_character=index_to_char[next_index]


        generated+=next_character
       ## print('generated text:' + generated)
        sentence=sentence[1:]+next_character
    return generated

print(generate_text(300,0.2))
print('with high temperature of 0.8:  '+ generate_text(300,0.8))