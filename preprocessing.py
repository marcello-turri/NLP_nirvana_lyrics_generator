import tensorflow as tf
import numpy as np
import neattext as nt
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

class Preproccessor:
    def __init__(self):
        self.path = "nirvana_lyrics.txt"

    def importing_dataset(self):
        f = open(self.path,'r')
        dataset = []
        for line in f:
          dataset.append(line[:-1])
        return dataset

    def preprocess_data(self,dataset):
      cleaned_sentences = []
      for sentence in dataset:
        sentence = sentence.lower()
        sentence = nt.TextFrame(sentence)
        sentence = sentence.remove_emails()
        sentence = sentence.remove_emojis()
        sentence = sentence.remove_puncts()
        sentence = sentence.remove_special_characters()
        sentence = sentence.remove_stopwords()
        sentence = sentence.fix_contractions()
        cleaned_sentences.append(sentence)
      return cleaned_sentences

    def tokenize(self,dataset):
        tokenizer = Tokenizer(oov_token='<OOV>', num_words=50000)
        tokenizer.fit_on_texts(dataset)
        self.total_words = len(tokenizer.word_index) + 1
        return tokenizer,self.total_words

    def text_to_sequences(self,dataset,tokenizer):
        input_sequences = []
        for seq in dataset:
            token_list = tokenizer.texts_to_sequences([seq])[0]
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i + 1]
                input_sequences.append(n_gram_sequence)
        return input_sequences

    def compute_max_seq_len(self,input_sequences):
        self.max_sequence_len = max([len(x) for x in input_sequences])
        return self.max_sequence_len

    def pad_sequences(self,input_sequences,max_sequence_len):
        return np.array(pad_sequences(input_sequences,maxlen=max_sequence_len,padding='pre'))

    def split_data_into_x_and_y(self,input_sequences,total_words):
        xs = input_sequences[:, :-1]  # [ 0 0 0 0 0 0 326 327] -> xs = [ 0 0 0 0 0 0 326]
        labels = input_sequences[:, -1] # [327]
        ys = tf.keras.utils.to_categorical(labels, num_classes=total_words) # [327, 247, 118, 138 ...] -> [0 ... 1 ...0][0 ..1...0]...
        return xs,ys

