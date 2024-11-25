import pandas as pd
import tensorflow as tf
import openpyxl
import numpy as np
import nltk
import string
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Bidirectional, Input, Embedding, LSTM, Dense, Flatten
from tensorflow.keras.models import Model

datainductive = pd.read_csv(r"data2x.csv", sep=";")

#REMOVE PUNCTUATION AND APPLY LOWERCASE
datainductive["Pertanyaan"] = datainductive["Pertanyaan"].apply(lambda wrd:[ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation])

datainductive["Pertanyaan"] = datainductive["Pertanyaan"].apply(lambda wrd: ''.join(wrd))

tag_to_answer = dict(zip(datainductive['Kategori'], datainductive['Jawaban']))

tokenizer = Tokenizer(num_words = 2000)
tokenizer.fit_on_texts(datainductive["Pertanyaan"])
train = tokenizer.texts_to_sequences(datainductive["Pertanyaan"])

x_train = pad_sequences(train)

le = LabelEncoder()
y_train = le.fit_transform(datainductive["Kategori"])

input_shape = x_train.shape[1]

unique_words = len(tokenizer.word_index)
output_length = le.classes_.shape[0]

model = tf.keras.Sequential()
model.add(Input(shape=(input_shape,)))
model.add(Embedding(unique_words+1, 20, input_length= (input_shape,)))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Flatten())
model.add(Dense(units= 64, activation='relu'))
model.add(Dense(units= 32, activation='relu'))
model.add(Dense(units= output_length, activation='softmax'))

model.compile(loss = "sparse_categorical_crossentropy", optimizer ='adam', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=200)

while True:
    textList = []
    user_input = input("You: ")
    prediction_input = []

    for letter in user_input:
        if letter not in string.punctuation:
            prediction_input.append(letter.lower())

    prediction_input = ''.join(prediction_input)
    textList.append(prediction_input)
    
    prediction_input = tokenizer.texts_to_sequences(textList)
    prediction_input = np.array(prediction_input).reshape(-1)
    prediction_input = pad_sequences([prediction_input], input_shape)
    
    output = model.predict(prediction_input)
    output = output.argmax()
    
    response_tag = le.inverse_transform([output])[0]
    bot_response = tag_to_answer.get(response_tag, "Sorry, I don't understand that question.")

    print("You: ", user_input)
    print("Bot: ", bot_response)
    
    if user_input == 'goodbye':
        bot_response == 'Good Bye! Thank you for using this feature!'
        print("Goodbye!")
        
        break