# type: ignore
# flake8: noqa
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| include: false
pip install nrclex
```
#
#| include: false

pip install tabulate
```
#
#| include: false
pip install transformers
#
#
#
#| include: false
pip install pyLDAvis
#
#
#
#| include: false
 pip install afinn
#
#
#
#| include: false
# General imports
import pickle
from joblib import dump, load
import os
import pandas as pd
import re
import numpy as np
import string

# NLTK imports
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS

# Preprocessing imports
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Model selection imports
from sklearn.model_selection import train_test_split, GridSearchCV

# Machine learning model imports
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from catboost import Pool, cv, CatBoostClassifier

# Word embedding imports
from gensim.models import Word2Vec

# LDA model imports
from gensim.models import LdaModel
from gensim import corpora
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis

# Metrics import
from sklearn.metrics import classification_report, accuracy_score
#
#
#
#
#| echo: false
#| eval: false
#| include: false
folder_path = 'speeches'  
files = os.listdir(folder_path)
files = sorted([file for file in files if os.path.isfile(os.path.join(folder_path, file)) and file.endswith('.txt')])

president_names = []

pattern = r'_(.+?)\.txt'  
for file in files:
    match = re.search(pattern, file)
    if match:
        president_name = match.group(1)
        # Remove the "_2" suffix from the president names here
        cleaned_president_name = president_name.replace('_2', '')
        president_names.append(cleaned_president_name)
    else:
        print(f"Warning: No match found in filename: {file}")
        president_names.append('Unknown')  # Placeholder for missing names

# Check  lengths
if len(files) != len(president_names):
    print(f"Warning: Number of files ({len(files)}) does not match number of president names ({len(president_names)})")

def preprocess_speeches(speech):
    # Tokenize the text
    tokens = word_tokenize(speech)
    # Remove punctuation and convert to lowercase
    tokens = [token.lower() for token in tokens if token not in string.punctuation]
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    for i in range(len(tokens)):
        clean_token = ''.join(char for char in tokens[i] if char.isalpha())
        tokens[i] = clean_token
    tokens = [token for token in tokens if len(token)>2]
    return tokens


df = pd.DataFrame(columns=['Presidents', 'Sentences'])
tokenised_speeches = pd.DataFrame(columns=['Presidents', 'Tokens'])  # To store the speeches tokenized by word

# Iterate over all files and extract sentence
for file_index in range(len(files)):
    file_path = os.path.join(folder_path, files[file_index])
    with open(file_path, 'r', encoding='utf-8') as file:
        speech = file.read()    
        tokens = preprocess_speeches(speech)

        lines = file.readlines()[2:] 

    text = ' '.join(lines)
    sentences = sent_tokenize(text)
    cleaned_sentences = [sentence.replace('\n', '') for sentence in sentences]

    current_president = president_names[file_index]
    dftemp = pd.DataFrame({'Presidents': [current_president] * len(cleaned_sentences), 'Sentences': cleaned_sentences})
    dftemp2 = pd.DataFrame({'Presidents': [current_president] * len(tokens), 'Tokens': tokens})
    df = pd.concat([df, dftemp], axis=0, ignore_index=True)
    tokenised_speeches = pd.concat([tokenised_speeches, dftemp2], axis=0, ignore_index=True)

df.reset_index(drop=True, inplace=True)
tokenised_speeches.reset_index(drop=True, inplace=True)
tokenised_speeches = tokenised_speeches[~tokenised_speeches['Tokens'].str.contains(r'[0-9]', na=False)]  ## Remove numeric tokens


# Save the DataFrame to a CSV file
#df.to_csv('finalSentences2.csv', index=False)
#tokenised_speeches.to_csv('finalTokens.csv', index=False)

#
#
#
#
#| include: false
a2data = pd.read_csv("finalSentences2.csv")
a2tokens = pd.read_csv("finalTokens.csv")
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| include: false
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import numpy as np
from scipy.special import softmax
#
#
#
#
#
#| echo: false
#| eval: false
#| include: false
data = a2data

import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

# Assuming a2data is loaded
data = a2data

# Define the preprocess function
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

# Define the model details
MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Preprocess the sentences
data['Preprocessed_Sentences'] = data['Sentences'].apply(preprocess)

# Compute the weighted sentiment score and get the model's sentiment classification
def compute_sentiment_data(text):
    # Tokenize and get model output
    encoded_input = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    probabilities = softmax(scores)
    
    # Compute the weighted sentiment score
    sentiment_values = np.array([-1, 0, 1])  # Corresponding to Negative, Neutral, and Positive
    sentiment_score = np.dot(sentiment_values, probabilities)
    
    # Get the model's sentiment classification
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    label = config.id2label[ranking[0]]
    
    return sentiment_score, label

# Predict sentiment data for the preprocessed sentences
sentiment_data = data['Preprocessed_Sentences'].apply(compute_sentiment_data)
data['Sentiment_Score'] = [s[0] for s in sentiment_data]
data['Predicted_Label'] = [s[1] for s in sentiment_data]

# Save the results
data.to_csv("HFM1_SentimentData.csv")


#
#
#
#
#
#
#
#
#| echo: false
#| fig-cap: "Figure X: RoBERTa 1 Mean Sentiment Scores per President"
HFM1 = pd.read_csv("HFM1_SentimentData.csv")
presidents_order = ['Mandela', 'deKlerk', 'Mbeki', ' Motlanthe', 'Zuma', 'Ramaphosa']
mean_scores_per_president = HFM1.groupby('Presidents')['Sentiment_Score'].mean()
adjusted_presidents_order = ['deKlerk'] + [president for president in presidents_order if president != 'deKlerk']
mean_scores_reordered = mean_scores_per_president.reindex(adjusted_presidents_order)

# Plotting the reordered mean sentiment scores
#plt.figure(figsize=(12, 7))
mean_scores_reordered.plot(kind='bar', color='skyblue')
plt.xlabel('President')
plt.ylabel('Mean Sentiment Score')
plt.axhline(0, color='red', linestyle='--')  #line at y=0 for reference
plt.tight_layout()
plt.show()
#
#
#
#
#
#| echo: false
#| fig-cap: "Figure X: RoBERTa 1 Sentiment Distribution per President "
colors = {'negative': '#FF9999', 'neutral': '#99CCFF', 'positive': '#99FF99'}
sentiment_counts = HFM1.groupby('Presidents')['Predicted_Label'].value_counts(normalize=True).unstack().fillna(0)

sentiment_counts.reindex(adjusted_presidents_order).plot(kind='bar', stacked=True, color=[colors[col] for col in sentiment_counts.columns])
plt.ylabel('Percentage')
plt.xlabel('President')
plt.legend(title='Sentiment')
plt.tight_layout()
plt.show()
#
#
#
#
#
#| include: false
#| fig-cap: "Figure X: RoBERTa 1 Distribution of Sentiment Scores for each President"
pastel_colors = ['#FFB3BA', '#FFDFBA', '#FFFFBA', '#BAFFC9', '#BAE1FF', '#D9BAFF']

def plot_histogram(president, color, ax):
    """Plot histogram of sentiment scores for a given president on a given axes."""
    president_df = HFM1[HFM1['Presidents'] == president]
    ax.hist(president_df['Sentiment_Score'], bins=30, color=color, edgecolor='white')
    ax.set_title(f'{president}')
    ax.set_xlabel('Sentiment Score')
    ax.set_ylabel('Frequency')
    ax.grid(axis='y', linestyle='--', linewidth=0.5)

# Creating a 2x3 grid plot for histograms of all presidents
fig, axes = plt.subplots(2, 3)

for president, color, ax in zip(adjusted_presidents_order, pastel_colors, axes.ravel()):
    plot_histogram(president, color, ax)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.90)
plt.show()
#
#
#
#
#
#| eval: false
#| include: false
# Assuming a2data is loaded
data = a2data

# Define the preprocess function
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

# Define the model details
MODEL = "cardiffnlp/xlm-twitter-politics-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Preprocess the sentences
data['Preprocessed_Sentences'] = data['Sentences'].apply(preprocess)

# Compute the weighted sentiment score and get the model's sentiment classification
def compute_sentiment_data(text):
    # Tokenize and get model output
    encoded_input = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    probabilities = softmax(scores)
    
    # Compute the weighted sentiment score
    sentiment_values = np.array([-1, 0, 1])  # Corresponding to Negative, Neutral, and Positive
    sentiment_score = np.dot(sentiment_values, probabilities)
    
    # Get the model's sentiment classification
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    label = config.id2label[ranking[0]]
    
    return sentiment_score, label

# Predict sentiment data for the preprocessed sentences
sentiment_data = data['Preprocessed_Sentences'].apply(compute_sentiment_data)
data['Sentiment_Score'] = [s[0] for s in sentiment_data]
data['Predicted_Label'] = [s[1] for s in sentiment_data]

# Save the results
data.to_csv("HFM2_SentimentData.csv")

data.head()
#
#
#
#
#
#| echo: false
#| fig-cap: "Figure X: RoBERTa 2 Mean Sentiment Scores per President "
HFM2 = pd.read_csv("HFM2_SentimentData.csv")
presidents_order = ['Mandela', 'deKlerk', 'Mbeki', ' Motlanthe', 'Zuma', 'Ramaphosa']
mean_scores_per_president = HFM2.groupby('Presidents')['Sentiment_Score'].mean()
adjusted_presidents_order = ['deKlerk'] + [president for president in presidents_order if president != 'deKlerk']
mean_scores_reordered = mean_scores_per_president.reindex(adjusted_presidents_order)

# Plotting the reordered mean sentiment scores
#plt.figure(figsize=(12, 7))
mean_scores_reordered.plot(kind='bar', color='skyblue')
plt.xlabel('President')
plt.ylabel('Mean Sentiment Score')
plt.axhline(0, color='red', linestyle='--')  #line at y=0 for reference
plt.tight_layout()
plt.show()
#
#
#
#
#
#
#
#| echo: false
#| fig-cap: "Figure X: RoBERTa 2 Sentiment Distribution per President"
colors = {'Negative': '#FF9999', 'Neutral': '#99CCFF', 'Positive': '#99FF99'}
sentiment_counts = HFM2.groupby('Presidents')['Predicted_Label'].value_counts(normalize=True).unstack().fillna(0)

sentiment_counts.reindex(adjusted_presidents_order).plot(kind='bar', stacked=True, color=[colors[col] for col in sentiment_counts.columns])
plt.ylabel('Percentage')
plt.xlabel('President')
plt.legend(title='Sentiment')
plt.tight_layout()
plt.show()
#
#
#
#
#
#
#
#| echo: false
#| fig-cap: "Figure X: RoBERTa 2 Distribution of Sentiment Scores for each President"
pastel_colors = ['#FFB3BA', '#FFDFBA', '#FFFFBA', '#BAFFC9', '#BAE1FF', '#D9BAFF']

def plot_histogram(president, color, ax):
    """Plot histogram of sentiment scores for a given president on a given axes."""
    president_df = HFM2[HFM2['Presidents'] == president]
    ax.hist(president_df['Sentiment_Score'], bins=30, color=color, edgecolor='white')
    ax.set_title(f'{president}')
    ax.set_xlabel('Sentiment Score')
    ax.set_ylabel('Frequency')
    ax.grid(axis='y', linestyle='--', linewidth=0.5)

# Creating a 2x3 grid plot for histograms of all presidents
fig, axes = plt.subplots(2, 3)

for president, color, ax in zip(adjusted_presidents_order, pastel_colors, axes.ravel()):
    plot_histogram(president, color, ax)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.90)
plt.show()
#
#
#
#
#
#
#| echo: false
#| eval: false
a2tokens = pd.read_csv("finalTokens.csv")
tokens = [text.split() for text in a2tokens["Tokens"]]

# Create a dictionary and corpus
dct = corpora.Dictionary(tokens)
corpus = [dct.doc2bow(t) for t in tokens]

# Train the LDA models
m_lda_3 = LdaModel(corpus=corpus, num_topics=3, id2word=dct, passes=20)
m_lda_4 = LdaModel(corpus=corpus, num_topics=4, id2word=dct, passes=20)
m_lda_5 = LdaModel(corpus=corpus, num_topics=5, id2word=dct, passes=20)


# Print the topics and associated words
topics = m_lda.print_topics(num_words=10)
for topic in topics:
    print(topic)

# Prepare the visualization
vis3 = gensimvis.prepare(m_lda_3, corpus, dct)
vis4 = gensimvis.prepare(m_lda_4, corpus, dct)
vis5 = gensimvis.prepare(m_lda_5, corpus, dct)

# Save model and visualisation objects
# pyLDAvis.save_html(vis3, "lda_3topic_vis.html")
# m_lda_3.save("m_lda_3")
# pyLDAvis.save_html(vis4, "lda_4topic_vis.html")
# m_lda_4.save("m_lda_4")
# pyLDAvis.save_html(vis5, "lda_5topic_vis.html")
# m_lda_5.save("m_lda")

#
#
#
#
#| echo: false
m_lda_3 = LdaModel.load("topic_models/m_lda_3")
m_lda_4 = LdaModel.load("topic_models/m_lda_4")
m_lda_5 = LdaModel.load("topic_models/m_lda")


def extract_topics(lda_model, m):
    df=pd.DataFrame({})
    topics = lda_model.print_topics(num_words=10)
    for i, topic in enumerate(topics):
        heading = i+1
        content = topic[1]
        items = content.split(' + ')
        tmp_df = pd.DataFrame({f'M{m} Topic {heading}': [item.strip().replace('"', '').replace('*', ' | ') for item in items]})
        df = pd.concat([df,tmp_df],axis=1)
    return df

topics_df = pd.concat([extract_topics(m_lda_3, 3),extract_topics(m_lda_4, 4),extract_topics(m_lda_5, 5)], axis=1)

md_tbl = topics_df.to_markdown(index=False)
md_tbl

#
#
#
#
#| echo: false
#| eval: true
#| include: false


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from sklearn.preprocessing import StandardScaler
from afinn import Afinn
 
#instantiate afinn
afn = Afinn()
data = pd.read_csv("finalSentences2.csv")

# compute scores (polarity) and labels
scores = [afn.score(sentence) for sentence in data["Sentences"]]
sentiment = ['positive' if score > 0
						else 'negative' if score < 0
							else 'neutral'
								for score in scores]
	
# dataframe creation
afn_df = pd.DataFrame()
afn_df['Presidents'] = data['Presidents']
afn_df['Sentences'] = data['Sentences']
afn_df['scores'] = scores
afn_df['sentiments'] = sentiment

# scaler = StandardScaler()
# scaled_data = scaler.fit_transform(afn_df['scores'].values.reshape(-1,1))
# afn_df['scores'] = scaled_data.squeeze()

presidents_order = ['Mandela', 'deKlerk', 'Mbeki', ' Motlanthe', 'Zuma', 'Ramaphosa']
mean_scores_per_president = afn_df.groupby('Presidents')['scores'].mean()
adjusted_presidents_order = ['deKlerk'] + [president for president in presidents_order if president != 'deKlerk']
mean_scores_reordered = mean_scores_per_president.reindex(adjusted_presidents_order)

# save object
# mean_scores_reordered.to_csv('wes_appendix/mean_scores_reordered.csv')
#
#
#
#
#
#
#| echo: False
mean_scores_reordered = pd.read_csv('wes_appendix/mean_scores_reordered.csv')
# Plotting the reordered mean sentiment scores
plt.figure(figsize=(12, 7))
#mean_scores_reordered.plot(kind='bar', color='skyblue')
plt.bar(mean_scores_reordered['Presidents'], mean_scores_reordered['scores'], color='skyblue')

plt.title('Mean Sentiment Scores per President')

plt.title("Mean Sentiment Scores per President", fontsize=20)
plt.xlabel('President', fontsize=20)
plt.ylabel('Mean Sentiment Score', fontsize=20)
plt.xticks(rotation=45, fontsize=16)
plt.yticks(fontsize=16)  # Set the font size for y-axis tick labels
plt.tight_layout()
plt.show()
#
#
#
#| echo: false
#| include: false
colors = {'negative': '#FF9999', 'neutral': '#99CCFF', 'positive': '#99FF99'}
sentiment_counts = afn_df.groupby('Presidents')['sentiments'].value_counts(normalize=True).unstack().fillna(0)
#
#
#
#
#
#
# save object
# sentiment_counts.reindex(adjusted_presidents_order).to_csv('wes_appendix/sentiment_counts.csv')
sentiment_counts = pd.read_csv("wes_appendix/sentiment_counts.csv")
sentiment_counts.plot(kind='bar', stacked=True, figsize=(12,7), color=[colors[col] for col in sentiment_counts.columns if col != 'Presidents'])
plt.title('Sentiment Distribution per President', fontsize=20)
plt.ylabel('Percentage', fontsize=20)
plt.xlabel('President', fontsize=20)
plt.legend(title='Sentiment', fontsize=20, title_fontsize=20)
plt.tight_layout()
plt.xticks(rotation=45, fontsize=16)
plt.yticks(fontsize=16) 
plt.show()
#
#
#
#
#
#
#| eval: true

from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
from matplotlib import cm

def plot_for_president_gradient(president, df):
    # Filter dataframe for the selected president
    president_df = df[df['Presidents'] == president]
    
    # Sorting data by order (using the 'Unnamed: 0' column as the order)
    president_df = president_df.sort_values(by='Unnamed: 0')
    
    # Normalize the scores to [0,1] for colormap
    norm = plt.Normalize(-1, 1)
    
    # Custom colormap: Red -> Gray -> Blue
    custom_cmap = LinearSegmentedColormap.from_list("custom", ["red", "gray", "blue"])
    
    # Plotting with a black background and custom colormap
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(np.arange(len(president_df)), president_df['Sentiment_Score'], 
                  color=custom_cmap(norm(president_df['Sentiment_Score'])), width=1.0)
    
    ax.set_title(f'Sentiment Score over time for President {president}')
    ax.set_xlabel('Order of Sentences')
    ax.set_ylabel('Sentiment Intensity')
    ax.axhline(0, color='white',linewidth=0.5)
    ax.grid(axis='y', color='white', linestyle='--', linewidth=0.5)
    ax.set_facecolor('black')
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=custom_cmap), ax=ax, label='Sentiment Score')
    
    plt.show()


presidents_list = afn_df['Presidents'].unique()
def plot_in_grid(president, ax):
    """Plot sentiment scores for a given president on a given axes."""
    president_df = afn_df[afn_df['Presidents'] == president]
    #president_df = president_df.sort_values(by='Unnamed: 0')
    
    # Normalize the scores for colormap
    norm = plt.Normalize(-1, 1)
    custom_cmap = LinearSegmentedColormap.from_list("custom", ["red", "gray", "blue"])

    bars = ax.bar(np.arange(len(president_df)), president_df['scores'], 
                  color=custom_cmap(norm(president_df['scores'])), width=1.0)
    ax.set_title(f'{president}', fontsize=20)
    ax.axhline(0, color='white', linewidth=0.5)
    ax.grid(axis='y', color='white', linestyle='--', linewidth=0.5)
    ax.set_facecolor('black')

# Creating a 2x3 grid plot for all presidents
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Sentiment Scores over time for each President', fontsize=20)

for president, ax in zip(adjusted_presidents_order, axes.ravel()):
    plot_in_grid(president, ax)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.90)

for ax in axes.ravel():
    ax.tick_params(axis='x', labelrotation=45, labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

# Set x and y labels fontsize
for ax in axes[1, :]:
    ax.set_xlabel('Sentences', fontsize=20)
for ax in axes[:, 0]:
    ax.set_ylabel('Scores', fontsize=20)

plt.show()
#
#
#
#
#
#
#| echo: false
#| eval: true

pastel_colors = ['#FFB3BA', '#FFDFBA', '#FFFFBA', '#BAFFC9', '#BAE1FF', '#D9BAFF']

def plot_histogram(president, color, ax):
    """Plot histogram of sentiment scores for a given president on a given axes."""
    president_df = afn_df[afn_df['Presidents'] == president]
    
    x_range = max(president_df['scores']) - min(president_df['scores'])
    half_x_range = x_range / 2
    
    # Set the x-axis limits to center the histogram around 0
    ax.set_xlim(-10, 10)

    ax.hist(president_df['scores'], bins=30, color=color, edgecolor='white')
    ax.set_title(f'{president}', fontsize=20)
    ax.set_xlabel('Sentiment Score', fontsize=20)
    ax.set_ylabel('Frequency', fontsize=20)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.grid(axis='y', linestyle='--', linewidth=0.5)

# Creating a 2x3 grid plot for histograms of all presidents
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Distribution of Sentiment Scores for each President', fontsize=20)

for president, color, ax in zip(adjusted_presidents_order, pastel_colors, axes.ravel()):
    plot_histogram(president, color, ax)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.90)
plt.show()

#
#
#
#
#
#
#
#| echo: false
#| eval: true
#| include: false
import pandas as pd
from nrclex import NRCLex

# Function to find the top emotions for a text
def find_top_emotions(text):
    emotion = NRCLex(text)
    return emotion.top_emotions

# Apply the function to each row in the DataFrame
data['Top_Emotions'] = data['Sentences'].apply(find_top_emotions)

unique_emotion_columns = ['fear', 'anger', 'anticipation','anticip', 'trust', 'surprise', 'positive', 'negative', 'sadness', 'disgust', 'joy']
presidents = data['Presidents'].unique()

president_dataframes = {president: pd.DataFrame(0, columns=unique_emotion_columns, index=[0]) for president in presidents}

for index, row in data.iterrows():
    president = row['Presidents']
    emotions = dict(row['Top_Emotions'])
    #president_df = president_dataframes[president]
    
    # Iterate through unique_emotion_columns
    for emotion_column in emotions.keys():
        if emotion_column in emotions:
            value = emotions[emotion_column]
            president_dataframes[president][emotion_column] += value

president_dataframes['Mandela']

for pres in presidents:
    president_dataframes[pres] = president_dataframes[pres].div(president_dataframes[pres].sum(axis=1), axis=0)

for president, df in president_dataframes.items():
    if 'anticipation' in df.columns and 'anticip' in df.columns:
        df['anticipation'] += df['anticip']  # Add 'anticip' to 'anticipation'
        df.drop(columns=['anticip'], inplace=True)  # Remove 'anticip' column

pastel_colors = ['#FFB3BA', '#FFDFBA', '#FFFFBA', '#BAFFC9', '#BAE1FF', '#D9BAFF']
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Distribution of Sentiment Scores for each President', fontsize=20)

order = ['deKlerk','Mandela', 'Mbeki', ' Motlanthe', 'Zuma', 'Ramaphosa']

for president, ax, colour in zip(order, axes.ravel(), pastel_colors):
    data = president_dataframes[president]
    x = list(data.keys())
    x.remove('positive')
    x.remove('negative')
    y = data[x].values.squeeze()
    ax.set_ylim(0, 0.25)
    ax.bar(x, y, color=colour)
    ax.set_title(president, fontsize=20)
    # ax.set_xlabel('Sentiment Category', fontsize=16)
    ax.set_ylabel('Sentiment Score', fontsize=16)
    ax.tick_params(axis='x', labelrotation=45, labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.90)
plt.show()
#
#
#
#
#
#
#


m_pLSA_4 = LsiModel.load("topics_models/m_pLSA_4")

num_words=20

topics = m_pLSA_4.print_topics(num_topics=4, num_words=num_words)
df4=pd.DataFrame({})
for topic in topics:
    content = topic[1]
    words=[]
    scores=[]
    scorewords = content.split(' + ')
    for item in scorewords:
        spl=item.split('*')
        scores.append(spl[0])
        words.append(spl[1].replace('"',''))
    topic_df=pd.DataFrame({f'T{topic[0]+1} Words': words, f'T{topic[0]+1} Scores': scores})
    df4 = pd.concat([df4, topic_df], axis=1)


# Create a 2x2 grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
for i in [0,1]:
    for j in [0,1]:
        t=2*i+j+1
        axs[i, j].barh(df4[f'T{t} Words'], df4[f'T{t} Scores'])
        axs[i, j].set_xlabel('Words')
        axs[i, j].set_ylabel('Weight')
        axs[i, j].set_title(f'M4: Topic {t}')
        axs[i, j].tick_params(axis='x', rotation=90, labelsize=10)
        axs[i, j].tick_params(axis='y', rotation=0, labelsize=9)
plt.tight_layout()
plt.show()
#
#
#
#
#
#
#
