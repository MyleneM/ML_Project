# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 16:59:24 2022

@author: tourniert
"""

# =============================================================================
# Import des packages
# =============================================================================
import nltk
import numpy as np
import pandas as pd
import pickle
# from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm


pd.options.mode.chained_assignment = None

# =============================================================================
# Import des données
# =============================================================================
df = pd.read_excel("../Data/Overall_Corpus.xlsx")

model = pickle.load(open('../Save Models/finalized_model_tree_v3_sentence_just_sentence.sav', 'rb'))


# =============================================================================
# Fonctions création prédicteurs
# =============================================================================
list_features_review = ['Review - Compound', 'Review - Count Character']
list_features_sentence = ['Sentence - Word_Average', 'Sentence - Start Index']


def features_sentence(sentence, review):
    vec_sentence = np.zeros(len(list_features_sentence))

    # Sentence - Word_Average
    list_word = sentence.split()
    vec_sentence[0] = sum(len(word) for word in list_word) / len(list_word)

    # Sentence - Start Index
    vec_sentence[1] = review.find(sentence)

    return vec_sentence

# =============================================================================
# 
# =============================================================================
def prediction(df, model):
    df['Sentences_review'] = None
    df['Zones_review'] = None

    for i, review in tqdm(enumerate(df['Review'])):
        type_last_sentence = 0
        consecutive_type = 0

        sentence_review = ''
        zone_review = ''

        X_review = np.array(df[list_features_review].iloc[i])

        list_sentences = nltk.tokenize.sent_tokenize(review)
        for sentence in list_sentences:
            # Calcul of features for the sentence
            X_sentence = features_sentence(sentence, review)   

            # Adding all features
            X = np.expand_dims(np.hstack([X_review, X_sentence, consecutive_type]),
                               axis=0)

            try:
                y = model.predict(X)
                sentence_review += sentence + '|'
                zone_review += str(int(y)) + '|'
            except ValueError:
                pass

            # Mise à jour du predicteur 'Sentence - Consecutive sentence type'
            if y == type_last_sentence:
                consecutive_type += 1
            else:
                consecutive_type = 0
            type_last_sentence = y

        df['Sentences_review'][i] = sentence_review
        df['Zones_review'][i] = zone_review

    return df


# =============================================================================
# Execution
# =============================================================================
prediction(df, model)
df_zoning = df[['Corpus', 'Title of the Play',
                'Sentences_review', 'Zones_review']]
df_zoning.to_excel('../Data/Corpus_with_zoning.xlsx', index=False)