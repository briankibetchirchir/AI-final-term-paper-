# ## Remember you should download the dataset in your local machine
# ## Sentiment Analysis on United States Airline Reviews from passengers and user experience
# ## Before Running this code in your local machine, make sure you install the required python libraries
# ## You can follow online reviews to install the libraries using pip install
# In[1]:

import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense, Dropout, SpatialDropout1D
from tensorflow.keras.layers import Embedding

# reading the dataset
df = pd.read_csv("./Tweets.csv")


# viewing the head of the dataset that includes columns and rows 
df.head()

# In[23]:

df.columns

# In[4]:

# viewing the first 5 tweets
tweet_df = df[['text','airline_sentiment']]
print(tweet_df.shape)
tweet_df.head(5)


# In[22]:

# dropping the neautral comments/tweets because they won't help us in our data
tweet_df = tweet_df[tweet_df['airline_sentiment'] != 'neutral']
print(tweet_df.shape)
tweet_df.head(5)


# In[21]:

# we need to see the values of the airline_sentiment column
tweet_df["airline_sentiment"].value_counts()


# In[6]:

# our values/labels are categorical, yet machine learning models understand numerical data
# so we convert our data to numerical data using factorize() method 
sentiment_label = tweet_df.airline_sentiment.factorize()
sentiment_label


# In[7]:

# tokenization by the help of a TOKENIZER(breaking down all words to small texts called tokens)
tweet = tweet_df.text.values
tokenizer = Tokenizer(num_words=5000)

# the below function creates an association between the texts and the numerics, which is the stored as a dictionary below(tokenizer.word_index)
tokenizer.fit_on_texts(tweet)
vocab_size = len(tokenizer.word_index) + 1
encoded_docs = tokenizer.texts_to_sequences(tweet)

# padding makes sure the words are of equal length, since the sentences are of different length as the dataset comes in
padded_sequence = pad_sequences(encoded_docs, maxlen=200)


# In[8]:

print(tokenizer.word_index)


# In[9]:

print(tweet[0])
print(encoded_docs[0])


# In[10]:

print(padded_sequence[0])


# In[11]:

# For our sentiment analysis, we will use LSTM layers in the machine learning model, which involves embedding layers(LSTM and Dense Layer)
# We add a drop out to avoid overfitting in between the layers(REGULARIZATION TECHNIQUE) making our model robust
embedding_vector_length = 32
model = Sequential() 
model.add(Embedding(vocab_size, embedding_vector_length, input_length=200) )
model.add(SpatialDropout1D(0.25))
model.add(LSTM(50, dropout=0.5, recurrent_dropout=0.5))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid')) 
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])  
print(model.summary()) 


# In[12]:

# Training our model for 5 epochs on the whole dataset with a 32 batch_size and a validation split of 20%
history = model.fit(padded_sequence,sentiment_label[0],validation_split=0.2, epochs=5, batch_size=32)


# In[16]:
# Ploting the accuracy of the trained model using matplotlib library

plt.plot(history.history['accuracy'], label='acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()
plt.savefig("Accuracy plot.jpg")


# In[25]:

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()
plt.savefig("Loss plot.jpg")


# In[18]:

# We define a function that takes a text as input and outputs its prediction label
def predict_sentiment(text):
    tw = tokenizer.texts_to_sequences([text])
    tw = pad_sequences(tw,maxlen=200)
    prediction = int(model.predict(tw).round().item())
    print("Predicted label: ", sentiment_label[1][prediction])


# In[19]:


# Test this on your Jupyter Notebook to see if the tweets/comments are positive or negative
test_sentence1 = "I enjoyed my journey on this flight."
predict_sentiment(test_sentence1)

test_sentence2 = "This is the worst flight experience of my life!"
predict_sentiment(test_sentence2)

