import nltk
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

import json

# Load your intents JSON file
with open('intents.json') as file:
    data = json.load(file)

# Tokenize and lemmatize each word in the input data
words = []
classes = []
documents = []
ignore_chars = ['?', '!', '.', ',']
for intent in data['intents']:
    for pattern in intent['patterns']:
        tokenized_words = nltk.word_tokenize(pattern)
        words.extend(tokenized_words)
        documents.append((tokenized_words, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_chars]
words = sorted(set(words))
classes = sorted(set(classes))

# Save the preprocessed data as pickle files
import pickle

with open('words.pkl', 'wb') as f:
    pickle.dump(words, f)
    
with open('classes.pkl', 'wb') as f:
    pickle.dump(classes, f)
    
with open('documents.pkl', 'wb') as f:
    pickle.dump(documents, f)
