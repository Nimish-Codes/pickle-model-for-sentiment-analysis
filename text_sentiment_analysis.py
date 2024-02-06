import numpy as np
import pandas as pd
import zipfile
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
import pickle
import os

# Load your ZIP file (replace 'your_data.zip' with your actual file)
with zipfile.ZipFile('IMDB detaset.csv.zip', 'r') as zip_ref:
    zip_ref.extractall('temp_folder')  # Extract contents to a temporary folder

# Load CSV data from the extracted file
df = pd.read_csv('temp_folder/IMDB detaset.csv')

# Neural network model
max_words = 1000
max_len = 100

# Check if a pre-trained model exists
if os.path.exists('trained_model.pkl'):
    with open('trained_model.pkl', 'rb') as f:
        model = pickle.load(f)
else:
  model = Sequential()
  model.add(Embedding(max_words, 8, input_length=max_len))
  model.add(Flatten())
  model.add(Dense(1, activation='sigmoid'))
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

  # Tokenization and padding
  tokenizer = Tokenizer(num_words=max_words)
  tokenizer.fit_on_texts(df['review'])
  sequences = tokenizer.texts_to_sequences(df['review'])
  x_train = pad_sequences(sequences, maxlen=max_len)

  # Training labels
  y_train = df['sentiment'].values

  # Convert labels to binary (assuming binary sentiment labels)
  y_train = np.array([1 if label == 1 else 0 for label in y_train])  # Convert to NumPy array

  # Train the neural network
  model.fit(x_train, y_train, epochs=2, batch_size=2, verbose=1)

  # Save the trained model using pickle
  with open('sentiment.pkl', 'wb') as f:
    pickle.dump(model, f)
