import pandas as pd
import nltk
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# python main.py --predict --all_models --prediction_file test.csv --output_path C:\palm\trained_injury_classifiers
def extract_column_to_new_file(input_filepath, output_filepath, column_name):
    df = pd.read_csv(input_filepath)
    print(len(df))
    if column_name not in df.columns:
        print(f"Error: Column '{column_name}' not found in {input_filepath}")
        return
    df[column_name] = df[column_name].apply(tokenize_and_normalize)
    print(len(df[column_name]))
    df[[column_name]].to_csv(output_filepath, index=False)
    print(f"Column '{column_name}' extracted and saved to {output_filepath}")


def tokenize_and_normalize(s):
    word_punct_token = WordPunctTokenizer().tokenize(s)
    clean_token = []
    for token in word_punct_token:
        token = token.lower()
        new_token = re.sub(r'[^a-zA-Z]+', '', token)
        if new_token != "" and len(new_token) >= 2:
            vowels = len([v for v in new_token if v in "aeiou"])
            if vowels != 0:
                clean_token.append(new_token)
    stop_words = stopwords.words('english')
    stop_words.extend(["could", "though", "would", "also", "many", 'much'])
    tokens = [x for x in clean_token if x not in stop_words]
    data_tagset = nltk.pos_tag(tokens)
    df_tagset = pd.DataFrame(data_tagset, columns=['Word', 'Tag'])
    lemmatizer = WordNetLemmatizer()
    lemmatize_text = []
    for word in tokens:
        output = [word, lemmatizer.lemmatize(word, pos='n'), lemmatizer.lemmatize(word, pos='a'),
                  lemmatizer.lemmatize(word, pos='v')]
        lemmatize_text.append(output)
    df = pd.DataFrame(lemmatize_text, columns=['Word', 'Lemmatized Noun', 'Lemmatized Adjective', 'Lemmatized Verb'])
    df['Tag'] = df_tagset['Tag']
    df = df.replace(['NN', 'NNS', 'NNP', 'NNPS'], 'n')
    df = df.replace(['JJ', 'JJR', 'JJS'], 'a')
    df = df.replace(['VBG', 'VBP', 'VB', 'VBD', 'VBN', 'VBZ'], 'v')
    df_lemmatized = df.copy()
    df_lemmatized['Tempt Lemmatized Word'] = df_lemmatized['Lemmatized Noun'] + ' | ' + df_lemmatized[
        'Lemmatized Adjective'] + ' | ' + df_lemmatized['Lemmatized Verb']
    df_lemmatized.head(5)
    lemma_word = df_lemmatized['Tempt Lemmatized Word']
    tag = df_lemmatized['Tag']
    i = 0
    new_word = []
    while i < len(tag):
        words = lemma_word[i].split('|')
        if tag[i] == 'n':
            word = words[0]
        elif tag[i] == 'a':
            word = words[1]
        elif tag[i] == 'v':
            word = words[2]
        new_word.append(word)
        i += 1
    df_lemmatized['Lemmatized Word'] = new_word
    lemma_word = [str(x).strip() for x in df_lemmatized['Lemmatized Word']]
    result_str = ', '.join(lemma_word)
    return result_str

# Example Usage
input_file = "C:\palm\EntityExtraction\combined_result.csv"
output_file = "C:\palm\EntityExtraction\AllcaseInjuryonly.csv"
column_to_extract = "case_name"

extract_column_to_new_file(input_file, output_file, column_to_extract)
