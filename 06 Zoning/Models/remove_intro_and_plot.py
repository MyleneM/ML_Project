# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 14:34:59 2022

@author: tourniert
"""
# =============================================================================
# Import des packages
# =============================================================================
import numpy as np
import pandas as pd


# =============================================================================
# Import des données
# =============================================================================
df = pd.read_excel('../Data/Corpus_with_zoning.xlsx')


# =============================================================================
# Fonctions
# =============================================================================
def remove_intro_and_plot(df):
    list_clean_review = ['' for i in range(len(df))]

    for i, (list_sentence_by_review, list_zone_by_review) in enumerate(zip(df['Sentences_review'],
                                                                           df['Zones_review'])):
        review_without_intro_and_plot = ''
        list_sentence = list_sentence_by_review.split("|")
        list_zone = list_zone_by_review.split("|")
        for zone, sentence in zip(list_zone, list_sentence):
            if zone == '1':
                review_without_intro_and_plot += sentence + ' '

        list_clean_review[i] = review_without_intro_and_plot

    return list_clean_review


list_clean_review = remove_intro_and_plot(df)
df['Review_without_intro_and_plot'] = list_clean_review
df.to_excel('../Data/Corpus_with_zoning.xlsx', index=False)
