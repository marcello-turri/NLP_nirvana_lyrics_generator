import tensorflow as tf
from preprocessing import Preproccessor
from keras.utils import pad_sequences
import numpy as np

class Model():
    def __init__(self,total_words,max_sequence_len):
        self.total_words = total_words
        self.max_sequence_len = max_sequence_len
        self.model_ = self.build_model()


    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.total_words,128,input_length=self.max_sequence_len-1),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,return_sequences=True)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)),
            tf.keras.layers.Dense(self.total_words,activation='softmax')
        ])
        return model

    def compile(self):
        self.model_.compile(loss='categorical_crossentropy',
                            optimizer='Adam',
                            metrics=['accuracy'])

    def fit(self,x,y,epochs):
        self.model_.fit(x,y,epochs=epochs)

    def predict_new_text(self,seed_text,tokenizer,next_words):
        for _ in range(next_words):
            token_list = tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list], maxlen=self.max_sequence_len - 1, padding='pre')
            probabilities = self.model_.predict(token_list)
            predicted = np.argmax(probabilities, axis=-1)[0]
            if predicted != 0:
                output_word = tokenizer.index_word[predicted]
                seed_text += " " + output_word
        return seed_text

