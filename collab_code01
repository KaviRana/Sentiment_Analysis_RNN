pip install gensim
#importing all files here----
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from keras.preprocessing.text import text_to_word_sequence
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt
#turning csv s into dataframes
train_df = pd.read_csv('Train.csv')
test_df = pd.read_csv('Test.csv')
valid_df = pd.read_csv('Valid.csv')
merged_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
merged_df.dropna(inplace=True)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]*>', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text

merged_df['text'] = merged_df['text'].apply(preprocess_text)
train_size = len(train_df)
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt

# Define callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss', 
        patience=3,
        mode='min',
        verbose=1
    ),
    ModelCheckpoint(
        filepath='sentiment_analysis_rnn,h5', 
        monitor='val_loss', 
        save_best_only=True,
        mode='min',
        verbose=1
    ),
    TensorBoard(
        log_dir='./logs', 
        histogram_freq=0, 
        write_graph=True, 
        write_images=False,
        update_freq='epoch'
    )
]

# Plot training and validation curves
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
val_size = len(valid_df)
test_size = len(test_df)

train_df = merged_df[:train_size]
valid_df = merged_df[train_size:train_size+val_size]
test_df = merged_df[train_size+val_size:]
max_words = 10000
max_len = 150

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train_df['text'])

train_sequences = tokenizer.texts_to_sequences(train_df['text'])
train_padded_sequences = pad_sequences(train_sequences, maxlen=max_len, padding='post')

val_sequences = tokenizer.texts_to_sequences(valid_df['text'])
val_padded_sequences = pad_sequences(val_sequences, maxlen=max_len, padding='post')

test_sequences = tokenizer.texts_to_sequences(test_df['text'])
test_padded_sequences = pad_sequences(test_sequences, maxlen=max_len, padding='post')
sentences = [text_to_word_sequence(sentence) for sentence in train_df['text']]
w2v_model = Word2Vec(sentences=train_df['text'], vector_size=100, window=5, min_count=1, workers=4)
w2v_model.save('word2vec_model.bin')
word2vec_model = Word2Vec.load('word2vec_model.bin')
num_words = min(max_words, len(tokenizer.word_index)) + 1
embedding_matrix = np.zeros((num_words, word2vec_model.vector_size))
for word, i in tokenizer.word_index.items():
    if i >= max_words:
        continue
    if word in word2vec_model.wv.key_to_index:
      embedding_matrix[i] = word2vec_model.wv[word]


embedding_size = 32
rnn_units = 64

model = Sequential()
model.add(Embedding(max_words, embedding_size, input_length=max_len, embeddings_regularizer=regularizers.l2(0.01)))
model.add(SimpleRNN(rnn_units))


model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_padded_sequences, train_df['label'], 
                    epochs=10, batch_size=66,
                    validation_data=(val_padded_sequences, valid_df['label']))

# Test the model
loss, accuracy = model.evaluate(test_padded_sequences, test_df['label'], verbose=False)
print("Test Accuracy: {:.4f}".format(accuracy*100))
model.save('sentiment_analysis_rnn.h5')
