import os
import pandas as pd
from bs4 import BeautifulSoup
from textblob import TextBlob
import requests
import syllables
import nltk

# Download NLTK resources (if not already downloaded)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Load input Excel file
input_file_path = 'Input.xlsx'
input_data = pd.read_excel(input_file_path)

# Create a directory to store extracted articles
output_dir = 'extracted_articles'
os.makedirs(output_dir, exist_ok=True)

# Function to extract and save article text
def extract_and_save_article(url, url_id):
    try:
        # Send an HTTP GET request to the URL
        response = requests.get(url)

        # Parse the HTML content of the page using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find and extract the article title
        article_title = soup.title.text.strip()

        # Find and extract the article text
        article_text = ' '.join([p.text for p in soup.find_all('p')])

        # Create a text file and save the article
        output_filename = os.path.join(output_dir, f'{url_id}.txt')
        with open(output_filename, 'w', encoding='utf-8') as file:
            file.write(f"{article_title}\n\n{article_text}")

        print(f'Article saved: {output_filename}')

    except Exception as e:
        print(f'Error extracting article from {url_id}: {str(e)}')

# Function for textual analysis
def text_analysis(article_text):
    blob = TextBlob(article_text)

    # Sentiment analysis
    sentiment = blob.sentiment
    positive_score = sentiment.polarity if sentiment.polarity > 0 else 0
    negative_score = -sentiment.polarity if sentiment.polarity < 0 else 0
    polarity_score = sentiment.polarity
    subjectivity_score = sentiment.subjectivity

    # Word and sentence analysis
    word_count = len(blob.words)
    sentence_lengths = [len(sentence.split()) for sentence in blob.sentences]
    avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths)
    avg_words_per_sentence = word_count / len(sentence_lengths)

    # Complexity analysis
    complex_word_count = sum(1 for word, pos in nltk.pos_tag(blob.words) if pos in ['VBG', 'VBN'])
    percentage_complex_words = (complex_word_count / word_count) * 100
    fog_index = 0.4 * (avg_words_per_sentence + percentage_complex_words)

    # Syllable analysis
    syllable_per_word = sum(syllables.estimate(w) for w in blob.words) / word_count

    # Personal pronoun analysis
    personal_pronouns = sum(1 for word, pos in nltk.pos_tag(blob.words) if pos == 'PRP')

    # Average word length
    avg_word_length = sum(len(word) for word in blob.words) / word_count

    return {
        'PositiveScore': positive_score,
        'NegativeScore': negative_score,
        'PolarityScore': polarity_score,
        'SubjectivityScore': subjectivity_score,
        'AvgSentenceLength': avg_sentence_length,
        'PercentageOfComplexWords': percentage_complex_words,
        'FogIndex': fog_index,
        'AvgNumberOfWordsPerSentence': avg_words_per_sentence,
        'ComplexWordCount': complex_word_count,
        'WordCount': word_count,
        'SyllablePerWord': syllable_per_word,
        'PersonalPronouns': personal_pronouns,
        'AvgWordLength': avg_word_length
    }

# Perform data extraction and analysis for each article
for index, row in input_data.iterrows():
    url_id = row['URL_ID']
    url = row['URL']

    # Extract and save article text
    extract_and_save_article(url, url_id)

    # Load the saved article text
    article_path = os.path.join(output_dir, f'{url_id}.txt')
    with open(article_path, 'r', encoding='utf-8') as file:
        article_text = file.read()

    # Perform textual analysis
    analysis_result = text_analysis(article_text)

    # Update the input_data DataFrame with the analysis results
    input_data.loc[index, 'PositiveScore'] = analysis_result['PositiveScore']
    input_data.loc[index, 'NegativeScore'] = analysis_result['NegativeScore']
    input_data.loc[index, 'PolarityScore'] = analysis_result['PolarityScore']
    input_data.loc[index, 'SubjectivityScore'] = analysis_result['SubjectivityScore']
    input_data.loc[index, 'AvgSentenceLength'] = analysis_result['AvgSentenceLength']
    input_data.loc[index, 'PercentageOfComplexWords'] = analysis_result['PercentageOfComplexWords']
    input_data.loc[index, 'FogIndex'] = analysis_result['FogIndex']
    input_data.loc[index, 'AvgNumberOfWordsPerSentence'] = analysis_result['AvgNumberOfWordsPerSentence']
    input_data.loc[index, 'ComplexWordCount'] = analysis_result['ComplexWordCount']
    input_data.loc[index, 'WordCount'] = analysis_result['WordCount']
    input_data.loc[index, 'SyllablePerWord'] = analysis_result['SyllablePerWord']
    input_data.loc[index, 'PersonalPronouns'] = analysis_result['PersonalPronouns']
    input_data.loc[index, 'AvgWordLength'] = analysis_result['AvgWordLength']

# Save the updated input_data DataFrame to the output Excel file
output_data_structure_path = 'Output Data Structure.xlsx'
input_data.to_excel(output_data_structure_path, index=False)
