import sys
sys.path = [path for path in sys.path if "nltk" not in path]
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np
import random
import json
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

intents_file = open('intents.json').read()
intents = json.loads(intents_file)

words = []
classes = []
documents = []
ignore_words = ['?', '!']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize and lemmatize the words
        tokenized_words = word_tokenize(pattern.lower())
        lemmatized_words = [lemmatizer.lemmatize(w) for w in tokenized_words if w not in ignore_words]
        words.extend(lemmatized_words)
        documents.append((lemmatized_words, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Remove duplicates from the word list and sort it alphabetically
words = sorted(list(set(words)))

# Sort classes alphabetically
classes = sorted(list(set(classes)))

# Save the preprocessed data to disk
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Initialize training data
training_data = []
output_empty = [0] * len(classes)

# Create training data
for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
   
