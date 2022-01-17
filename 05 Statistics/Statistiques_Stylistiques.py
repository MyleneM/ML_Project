# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 11:08:13 2020

@author: Marie
"""

import os
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import statistics as stat
import matplotlib.pyplot as plt
from matplotlib import pyplot

os.chdir("C:/Users/Marie/Documents/cours_particuliers/mylène/projet6_analyses")

###################################################

# FONCTIONS : 

def nettoyage(ch): 
    ch = ch.lower()
    car_spec = [',',';',':','/','-','_','\t','\n','^','<','>','"',
                '$','£','€','*','#','%','{','}','[',']','(',')','&',
                '+','=',"'"]
    for c in car_spec: 
        ch = ch.replace(c," ")
    car_fin = ['...','?','!','.']
    for c in car_fin: 
        ch = ch.replace(c," .")
    ch = ch.replace("  "," ")
    return ch

def nettoyage_stopwords(ch): 
    l = ch.split(" ")
    sw = stopwords.words('english') 
    l = [w for w in l if w not in sw]
    l=[x for x in l if x!='']
    l=[x for x in l if (x=='.' or len(x)!=1)] 
    l=[x for x in l if len(x)<20] 
    return " ".join(l)

def count_words(ch): 
    l = ch.split(" ")
    l=[x for x in l if x!='']
    l=[x for x in l if x!='.']
    return len(l)  

def count_unique_words(ch): 
    l = ch.split(" ")
    l=[x for x in l if x!='']
    l=[x for x in l if x!='.']
    l = np.array(l)
    return len(np.unique(l))

def count_sentences(ch): 
    return len(ch.split('. '))-1 

def longueur_mots(ch): 
    ch  =  ch.replace('.',' ')
    ch = ch.replace("  "," ")
    l = ch.split(" ")
    l=[x for x in l if x!='']
    l=[x for x in l if len(x)!=1]
    res = [] 
    for k in range(len(l)):
        res.append(len(l[k]))
    return [min(res),max(res),stat.mean(res),stat.median(res)]

def longueur_phrases(ch): 
    l = ch.split(". ") 
    res = [] 
    for k in range(len(l)): 
        l1 = l[k].split(" ") 
        l1=[x for x in l1 if x!='']
        l1=[x for x in l1 if len(x)!=1]
        res.append(len(l1))
    return [min(res),max(res),stat.mean(res),stat.median(res)]

###################################################

# Pour le moment on se concentre sur EXEUNT et WHAT'SON STAGE.

#df_lt = pd.read_csv('20201006_df_exeunt_londontheater.csv')
#df_n = pd.read_csv('20201006_df_exeunt_national.csv')
#df_et = pd.read_csv('20201006_df_everythingtheater.csv')
df_wos = pd.read_csv('20201001_df_whatsonstage.csv')

# Selectionner les donnes pour 2010 (en fonction de la date de la critique): 
# London Theater: no data for 2010
# National: no data for 2010

year_wos = []
for k in range(len(df_wos['Date'])):
    if pd.isna(df_wos['Date'][k]):
        year_wos.append(None)
    else: 
        year_wos.append(df_wos['Date'][k].split(sep='-')[0])
df_wos['Year'] = year_wos

df_wos_2010 = df_wos.loc[df_wos['Year']=='2010']
# 529 critiques

# Compilation des dataframes 2010 (1-2,3,4 et 5): 
df12 = pd.read_csv('20201008_2010_df_1-2.csv') # 310 critiques
df3 = pd.read_csv('20201008_2010_df_3.csv') # 180 critiques
df4 = pd.read_csv('20201008_2010_df_4.csv') # 198 critiques
df5 = pd.read_csv('20201008_2010_df_5.csv') # 206 critiques

# ATTENTION : on va comparer 2 dataframes qui sont de tailles completement 
# differentes... 
# wos d'un cote et les df de l'autre

# Clean : 
    
wos_review_clean = []
wos_review_nostop = []
for k in df_wos_2010.axes[0]: 
    print(k)
    if pd.isna(df_wos_2010['Review'][k]):
        wos_review_clean.append(None)
        wos_review_nostop.append(None)
    else: 
        wos_review_clean.append(nettoyage(df_wos_2010['Review'][k]))
        wos_review_nostop.append(nettoyage_stopwords(wos_review_clean[-1]))
df_wos_2010['Review_clean'] = wos_review_clean
df_wos_2010['Review_no_stopword'] = wos_review_nostop 

df12_review_clean = []
df12_review_nostop = []
for k in df12.axes[0]: 
    if pd.isna(df12['Review'][k]):
        df12_review_clean.append(None)
        df12_review_nostop.append(None)
    else: 
        df12_review_clean.append(nettoyage(df12['Review'][k]))
        df12_review_nostop.append(nettoyage_stopwords(df12_review_clean[-1]))
df12['Review_clean'] = df12_review_clean
df12['Review_no_stopword'] = df12_review_nostop

df3_review_clean = []
df3_review_nostop = []
for k in df3.axes[0]: 
    if pd.isna(df3['Review'][k]):
        df3_review_clean.append(None)
        df3_review_nostop.append(None)
    else: 
        df3_review_clean.append(nettoyage(df3['Review'][k]))
        df3_review_nostop.append(nettoyage_stopwords(df3_review_clean[-1]))
df3['Review_clean'] = df3_review_clean
df3['Review_no_stopword'] = df3_review_nostop 

df4_review_clean = []
df4_review_nostop = []
for k in df4.axes[0]: 
    if pd.isna(df4['Review'][k]):
        df4_review_clean.append(None)
        df4_review_nostop.append(None)
    else: 
        df4_review_clean.append(nettoyage(df4['Review'][k]))
        df4_review_nostop.append(nettoyage_stopwords(df4_review_clean[-1]))
df4['Review_clean'] = df4_review_clean
df4['Review_no_stopword'] = df4_review_nostop 
    
df5_review_clean = []
df5_review_nostop = []
for k in df5.axes[0]: 
    if pd.isna(df5['Review'][k]):
        df5_review_clean.append(None)
        df5_review_nostop.append(None)
    else: 
        df5_review_clean.append(nettoyage(df5['Review'][k]))
        df5_review_nostop.append(nettoyage_stopwords(df5_review_clean[-1]))
df5['Review_clean'] = df5_review_clean
df5['Review_no_stopword'] = df5_review_nostop 

# Comparaisons : 

wos_nb_mots = []
wos_nb_mots_nostop = []
wos_nb_phrases = []
for k in df_wos_2010.axes[0]: 
    print(k)
    if pd.isna(df_wos_2010['Review'][k])==False:
        wos_nb_mots.append(count_words(df_wos_2010['Review_clean'][k]))
        wos_nb_mots_nostop.append(count_words(df_wos_2010['Review_no_stopword'][k]))
        wos_nb_phrases.append(count_sentences(df_wos_2010['Review_clean'][k]))
    
df_nb_mots = []
df_nb_mots_nostop = []
df_nb_phrases = []
for k in df12.axes[0]: 
    if pd.isna(df12['Review'][k])==False:
        df_nb_mots.append(count_words(df12['Review_clean'][k]))
        df_nb_mots_nostop.append(count_words(df12['Review_no_stopword'][k]))
        df_nb_phrases.append(count_sentences(df12['Review_clean'][k]))
for k in df3.axes[0]: 
    if pd.isna(df3['Review'][k])==False:
        df_nb_mots.append(count_words(df3['Review_clean'][k]))
        df_nb_mots_nostop.append(count_words(df3['Review_no_stopword'][k]))
        df_nb_phrases.append(count_sentences(df3['Review_clean'][k]))
for k in df4.axes[0]: 
    if pd.isna(df4['Review'][k])==False:
        df_nb_mots.append(count_words(df4['Review_clean'][k]))
        df_nb_mots_nostop.append(count_words(df4['Review_no_stopword'][k]))
        df_nb_phrases.append(count_sentences(df4['Review_clean'][k]))
for k in df5.axes[0]: 
    if pd.isna(df5['Review'][k])==False:
        df_nb_mots.append(count_words(df5['Review_clean'][k]))
        df_nb_mots_nostop.append(count_words(df5['Review_no_stopword'][k]))
        df_nb_phrases.append(count_sentences(df5['Review_clean'][k]))

pyplot.boxplot([wos_nb_mots,df_nb_mots])
pyplot.gca().xaxis.set_ticklabels(["What's On Stage", "Theater Record"])
pyplot.title('Number of Words in Reviews')

pyplot.boxplot([wos_nb_mots,df_nb_mots],showfliers=False)
pyplot.gca().xaxis.set_ticklabels(["What's On Stage", "Theater Record"])
pyplot.title('Number of Words in Reviews')

pyplot.boxplot([wos_nb_mots_nostop,df_nb_mots_nostop])
pyplot.gca().xaxis.set_ticklabels(["What's On Stage", "Theater Record"])
pyplot.title('Number of Words in Reviews (Excluding Stopwords)')

pyplot.boxplot([wos_nb_mots_nostop,df_nb_mots_nostop],showfliers=False)
pyplot.gca().xaxis.set_ticklabels(["What's On Stage", "Theater Record"])
pyplot.title('Number of Words in Reviews (Excluding Stopwords)')

pyplot.boxplot([wos_nb_phrases,df_nb_phrases])
pyplot.gca().xaxis.set_ticklabels(["What's On Stage", "Theater Record"])
pyplot.title('Number of Sentences in Reviews')

pyplot.boxplot([wos_nb_phrases,df_nb_phrases],showfliers=False)
pyplot.gca().xaxis.set_ticklabels(["What's On Stage", "Theater Record"])
pyplot.title('Number of Sentences in Reviews')

# More stats : 

wos_mot_min = []
wos_mot_max = []
wos_mot_moy = []
wos_mot_med = []
wos_phr_min = []
wos_phr_max = []
wos_phr_moy = []
wos_phr_med = [] 

for k in df_wos_2010.axes[0]:
    if pd.isna(df_wos_2010['Review'][k])==False:
        wos_mot_min.append(longueur_mots(df_wos_2010['Review_no_stopword'][k])[0])
        wos_mot_max.append(longueur_mots(df_wos_2010['Review_no_stopword'][k])[1])
        wos_mot_moy.append(longueur_mots(df_wos_2010['Review_no_stopword'][k])[2])
        wos_mot_med.append(longueur_mots(df_wos_2010['Review_no_stopword'][k])[3])
        wos_phr_min.append(longueur_phrases(df_wos_2010['Review_no_stopword'][k])[0])
        wos_phr_max.append(longueur_phrases(df_wos_2010['Review_no_stopword'][k])[1])
        wos_phr_moy.append(longueur_phrases(df_wos_2010['Review_no_stopword'][k])[2])
        wos_phr_med.append(longueur_phrases(df_wos_2010['Review_no_stopword'][k])[3])

df_mot_min = []
df_mot_max = []
df_mot_moy = []
df_mot_med = []
df_phr_min = []
df_phr_max = []
df_phr_moy = []
df_phr_med = [] 

for k in df12.axes[0]:
    if pd.isna(df12['Review'][k])==False:
        df_mot_min.append(longueur_mots(df12['Review_no_stopword'][k])[0])
        df_mot_max.append(longueur_mots(df12['Review_no_stopword'][k])[1])
        df_mot_moy.append(longueur_mots(df12['Review_no_stopword'][k])[2])
        df_mot_med.append(longueur_mots(df12['Review_no_stopword'][k])[3])
        df_phr_min.append(longueur_phrases(df12['Review_no_stopword'][k])[0])
        df_phr_max.append(longueur_phrases(df12['Review_no_stopword'][k])[1])
        df_phr_moy.append(longueur_phrases(df12['Review_no_stopword'][k])[2])
        df_phr_med.append(longueur_phrases(df12['Review_no_stopword'][k])[3])
for k in df3.axes[0]:
    if pd.isna(df3['Review'][k])==False:
        df_mot_min.append(longueur_mots(df3['Review_no_stopword'][k])[0])
        df_mot_max.append(longueur_mots(df3['Review_no_stopword'][k])[1])
        df_mot_moy.append(longueur_mots(df3['Review_no_stopword'][k])[2])
        df_mot_med.append(longueur_mots(df3['Review_no_stopword'][k])[3])
        df_phr_min.append(longueur_phrases(df3['Review_no_stopword'][k])[0])
        df_phr_max.append(longueur_phrases(df3['Review_no_stopword'][k])[1])
        df_phr_moy.append(longueur_phrases(df3['Review_no_stopword'][k])[2])
        df_phr_med.append(longueur_phrases(df3['Review_no_stopword'][k])[3])
for k in df4.axes[0]:
    if pd.isna(df4['Review'][k])==False:
        df_mot_min.append(longueur_mots(df4['Review_no_stopword'][k])[0])
        df_mot_max.append(longueur_mots(df4['Review_no_stopword'][k])[1])
        df_mot_moy.append(longueur_mots(df4['Review_no_stopword'][k])[2])
        df_mot_med.append(longueur_mots(df4['Review_no_stopword'][k])[3])
        df_phr_min.append(longueur_phrases(df4['Review_no_stopword'][k])[0])
        df_phr_max.append(longueur_phrases(df4['Review_no_stopword'][k])[1])
        df_phr_moy.append(longueur_phrases(df4['Review_no_stopword'][k])[2])
        df_phr_med.append(longueur_phrases(df4['Review_no_stopword'][k])[3])
for k in df5.axes[0]:
    if pd.isna(df5['Review'][k])==False:
        df_mot_min.append(longueur_mots(df5['Review_no_stopword'][k])[0])
        df_mot_max.append(longueur_mots(df5['Review_no_stopword'][k])[1])
        df_mot_moy.append(longueur_mots(df5['Review_no_stopword'][k])[2])
        df_mot_med.append(longueur_mots(df5['Review_no_stopword'][k])[3])
        df_phr_min.append(longueur_phrases(df5['Review_no_stopword'][k])[0])
        df_phr_max.append(longueur_phrases(df5['Review_no_stopword'][k])[1])
        df_phr_moy.append(longueur_phrases(df5['Review_no_stopword'][k])[2])
        df_phr_med.append(longueur_phrases(df5['Review_no_stopword'][k])[3])
        
pyplot.boxplot([wos_mot_min,df_mot_min])
pyplot.gca().xaxis.set_ticklabels(["What's On Stage", "Theater Record"])
pyplot.title('Minimum Word Length in Reviews')

pyplot.boxplot([wos_mot_min,df_mot_min],showfliers=False)
pyplot.gca().xaxis.set_ticklabels(["What's On Stage", "Theater Record"])
pyplot.title('Minimum Word Length in Reviews')

pyplot.boxplot([wos_mot_max,df_mot_max])
pyplot.gca().xaxis.set_ticklabels(["What's On Stage", "Theater Record"])
pyplot.title('Maximum Word Length in Reviews')

pyplot.boxplot([wos_mot_max,df_mot_max],showfliers=False)
pyplot.gca().xaxis.set_ticklabels(["What's On Stage", "Theater Record"])
pyplot.title('Maximum Word Length in Reviews')

pyplot.boxplot([wos_mot_moy,df_mot_moy])
pyplot.gca().xaxis.set_ticklabels(["What's On Stage", "Theater Record"])
pyplot.title('Average Word Length in Reviews')

pyplot.boxplot([wos_mot_moy,df_mot_moy],showfliers=False)
pyplot.gca().xaxis.set_ticklabels(["What's On Stage", "Theater Record"])
pyplot.title('Average Word Length in Reviews')

pyplot.boxplot([wos_mot_med,df_mot_med])
pyplot.gca().xaxis.set_ticklabels(["What's On Stage", "Theater Record"])
pyplot.title('Median Word Length in Reviews')

pyplot.boxplot([wos_mot_med,df_mot_med],showfliers=False)
pyplot.gca().xaxis.set_ticklabels(["What's On Stage", "Theater Record"])
pyplot.title('Median Word Length in Reviews')

pyplot.boxplot([wos_phr_min,df_phr_min])
pyplot.gca().xaxis.set_ticklabels(["What's On Stage", "Theater Record"])
pyplot.title('Minimum Sentence Length in Reviews')

pyplot.boxplot([wos_phr_min,df_phr_min],showfliers=False)
pyplot.gca().xaxis.set_ticklabels(["What's On Stage", "Theater Record"])
pyplot.title('Minimum Sentence Length in Reviews')

pyplot.boxplot([wos_phr_max,df_phr_max])
pyplot.gca().xaxis.set_ticklabels(["What's On Stage", "Theater Record"])
pyplot.title('Maximum Sentence Length in Reviews')

pyplot.boxplot([wos_phr_max,df_phr_max],showfliers=False)
pyplot.gca().xaxis.set_ticklabels(["What's On Stage", "Theater Record"])
pyplot.title('Maximum Sentence Length in Reviews')

pyplot.boxplot([wos_phr_moy,df_phr_moy])
pyplot.gca().xaxis.set_ticklabels(["What's On Stage", "Theater Record"])
pyplot.title('Average Sentence Length in Reviews')

pyplot.boxplot([wos_phr_moy,df_phr_moy],showfliers=False)
pyplot.gca().xaxis.set_ticklabels(["What's On Stage", "Theater Record"])
pyplot.title('Average Sentence Length in Reviews')

pyplot.boxplot([wos_phr_med,df_phr_med])
pyplot.gca().xaxis.set_ticklabels(["What's On Stage", "Theater Record"])
pyplot.title('Median Sentence Length in Reviews')

pyplot.boxplot([wos_phr_med,df_phr_med],showfliers=False)
pyplot.gca().xaxis.set_ticklabels(["What's On Stage", "Theater Record"])
pyplot.title('Median Sentence Length in Reviews')

wos_mots_uniques = []
wos_mots_uniques_nostop = []
wos_ratio_unique = []
wos_ratio_unique_nostop = []
i = 0
for k in df_wos_2010.axes[0]:
    if pd.isna(df_wos_2010['Review'][k])==False:
        wos_mots_uniques.append(count_unique_words(df_wos_2010['Review_clean'][k]))
        wos_mots_uniques_nostop.append(count_unique_words(df_wos_2010['Review_no_stopword'][k]))
        wos_ratio_unique.append(wos_mots_uniques[-1]/wos_nb_mots[i])
        wos_ratio_unique_nostop.append(wos_mots_uniques_nostop[-1]/wos_nb_mots_nostop[i])
        i = i+1

df_mots_uniques = []
df_mots_uniques_nostop = []
df_ratio_unique = []
df_ratio_unique_nostop = []
i = 0
for k in df12.axes[0]:
    if pd.isna(df12['Review'][k])==False:
        df_mots_uniques.append(count_unique_words(df12['Review_clean'][k]))
        df_mots_uniques_nostop.append(count_unique_words(df12['Review_no_stopword'][k]))
        df_ratio_unique.append(df_mots_uniques[-1]/df_nb_mots[i])
        df_ratio_unique_nostop.append(df_mots_uniques_nostop[-1]/df_nb_mots_nostop[i])
        i = i+1
for k in df3.axes[0]:
    if pd.isna(df3['Review'][k])==False:
        df_mots_uniques.append(count_unique_words(df3['Review_clean'][k]))
        df_mots_uniques_nostop.append(count_unique_words(df3['Review_no_stopword'][k]))
        df_ratio_unique.append(df_mots_uniques[-1]/df_nb_mots[i])
        df_ratio_unique_nostop.append(df_mots_uniques_nostop[-1]/df_nb_mots_nostop[i])
        i = i+1
for k in df4.axes[0]:
    if pd.isna(df4['Review'][k])==False:
        df_mots_uniques.append(count_unique_words(df4['Review_clean'][k]))
        df_mots_uniques_nostop.append(count_unique_words(df4['Review_no_stopword'][k]))
        df_ratio_unique.append(df_mots_uniques[-1]/df_nb_mots[i])
        df_ratio_unique_nostop.append(df_mots_uniques_nostop[-1]/df_nb_mots_nostop[i])
        i = i+1
for k in df5.axes[0]:
    if pd.isna(df5['Review'][k])==False:
        df_mots_uniques.append(count_unique_words(df5['Review_clean'][k]))
        df_mots_uniques_nostop.append(count_unique_words(df5['Review_no_stopword'][k]))
        df_ratio_unique.append(df_mots_uniques[-1]/df_nb_mots[i])
        df_ratio_unique_nostop.append(df_mots_uniques_nostop[-1]/df_nb_mots_nostop[i])
        i = i+1

pyplot.boxplot([wos_mots_uniques,df_mots_uniques])
pyplot.gca().xaxis.set_ticklabels(["What's On Stage", "Theater Record"])
pyplot.title('Number of Unique Words in Reviews')

pyplot.boxplot([wos_mots_uniques,df_mots_uniques],showfliers=False)
pyplot.gca().xaxis.set_ticklabels(["What's On Stage", "Theater Record"])
pyplot.title('Number of Unique Words in Reviews')

pyplot.boxplot([wos_mots_uniques_nostop,df_mots_uniques_nostop])
pyplot.gca().xaxis.set_ticklabels(["What's On Stage", "Theater Record"])
pyplot.title('Number of Unique Words in Reviews (Excluding Stopwords)')

pyplot.boxplot([wos_mots_uniques,df_mots_uniques],showfliers=False)
pyplot.gca().xaxis.set_ticklabels(["What's On Stage", "Theater Record"])
pyplot.title('Number of Unique Words in Reviews (Excluding Stopwords)')

pyplot.boxplot([wos_ratio_unique,df_ratio_unique])
pyplot.gca().xaxis.set_ticklabels(["What's On Stage", "Theater Record"])
pyplot.title('Number of Unique Words / Number of Words')

pyplot.boxplot([wos_ratio_unique,df_ratio_unique],showfliers=False)
pyplot.gca().xaxis.set_ticklabels(["What's On Stage", "Theater Record"])
pyplot.title('Number of Unique Words / Number of Words')

pyplot.boxplot([wos_ratio_unique_nostop,df_ratio_unique_nostop])
pyplot.gca().xaxis.set_ticklabels(["What's On Stage", "Theater Record"])
pyplot.title('Number of Unique Words / Number of Words (Excluding Stopwords)')

pyplot.boxplot([wos_ratio_unique_nostop,df_ratio_unique_nostop],showfliers=False)
pyplot.gca().xaxis.set_ticklabels(["What's On Stage", "Theater Record"])
pyplot.title('Number of Unique Words / Number of Words (Excluding Stopwords)')