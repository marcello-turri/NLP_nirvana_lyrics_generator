from preprocessing import Preproccessor
from model import Model

preprocessor = Preproccessor()
raw_dataset = preprocessor.importing_dataset()
dataset = preprocessor.preprocess_data(raw_dataset)
tokenizer,total_words = preprocessor.tokenize(dataset)
input_sequences = preprocessor.text_to_sequences(dataset,tokenizer)
max_sequence_len = preprocessor.compute_max_seq_len(input_sequences)
input_sequences_padded = preprocessor.pad_sequences(input_sequences,max_sequence_len)
xs,ys = preprocessor.split_data_into_x_and_y(input_sequences_padded,total_words)
model = Model(total_words,max_sequence_len)
model.compile()
model.fit(xs,ys,epochs=60)
print(model.predict_new_text("Hi this is me",tokenizer,20))
print(model.predict_new_text("what do you think about",tokenizer,20))

