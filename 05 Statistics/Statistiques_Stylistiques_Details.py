# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 18:32:27 2020

@author: Marie
"""

import os
import pandas as pd
import numpy as np
import nltk
#import nltk.tokenize
#import nltk.tag
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#from nltk.corpus import stopwords
import statistics as stat
import matplotlib.pyplot as plt

os.chdir("C:/Users/Marie/Documents/cours_particuliers/mylène/PROJET4_syntax_analysis")

# FONCTIONS: 

# Calcul des % pour la ponctuation interne aux phrases: 
def int_punct(df): 
    res_vi = []
    res_pv = []
    res_ti = []
    for k in range(len(df.axes[0])): 
        a = df['Review'][k]
        v = 0
        p = 0
        t = 0
        if pd.isna(a)==False: 
            for c in a: 
                if c==',': 
                    v = v+1
                if c==';':
                    p = p+1
                if c==' - ': # espace pour ne pas prendre en compte les mots composes 
                    t = t+1
        s = v+p+t
        if s==0: 
            res_vi.append(None)
            res_pv.append(None)
            res_ti.append(None)
        else:     
            res_vi.append(v/s*100)
            res_pv.append(p/s*100)
            res_ti.append(t/s*100)
    res = pd.DataFrame({'ID':range(len(df.axes[0])),
                        'Virgules_pct':res_vi,
                        'Point_virgules_pct':res_pv,
                        'Tirets_pct':res_ti})
    return(res)    

# Calcul des % pour la ponctuation externe aux phrases: 
def ext_punct(df): 
    res_dec = []
    res_int = []
    res_exc = []
    for k in range(len(df.axes[0])): 
        a = df['Review'][k]
        d = 0
        i = 0
        e = 0
        if pd.isna(a)==False: 
            for c in a: 
                if c=='.': 
                    d = d+1
                if c=='?':
                    i = i+1
                if c=='!': 
                    e = e+1
        s = d+i+e
        if s==0: 
            res_dec.append(None)
            res_int.append(None)
            res_exc.append(None)
        else:     
            res_dec.append(d/s*100)
            res_int.append(i/s*100)
            res_exc.append(e/s*100)
    res = pd.DataFrame({'ID':range(len(df.axes[0])),
                        'Declaratives_pct':res_dec,
                        'Interrogatives_pct':res_int,
                        'Exclamatives_pct':res_exc})
    return(res)
    
# Calcul des % des differents types de mots dans chaque critique: 
def type_mots(df):
    # Les types de mots recherches: 
    adv = ['RB','RBR','RBS']
    nn = ['NN','NNS','NNP','NNPS']
    vb = ['VB','VBD','VBG','VBN','VBP','VBZ']
    adj = ['JJ','JJR']
    ads = ['JJS']
    # Listes vides pour stocker les resultats
    res_mots = []
    res_adv = []
    res_nn = []
    res_vb = []
    res_adj = []
    res_ads = []
    for k in range(len(df.axes[0])):
    #for k in range(10):
        t = df['Review'][k]
        # Set counters to 0: 
        ca = 0
        cn = 0
        cv = 0
        cj = 0
        cs = 0
        if pd.isna(t)==False: 
            t_tok = nltk.word_tokenize(t)
            res_mots.append(len(t_tok))
            if len(t_tok)==0: # ie si pas de texte de critique
                res_adv.append(None)
                res_nn.append(None)
                res_vb.append(None)
                res_adj.append(None)
                res_ads.append(None)
            else: 
                t_tag = nltk.pos_tag(t_tok)
                for x in t_tag: 
                    if x[1] in adv: 
                        ca = ca+1
                    if x[1] in nn: 
                        cn = cn+1
                    if x[1] in vb: 
                        cv = cv+1
                    if x[1] in adj: 
                        cj = cj+1
                    if x[1] in ads: 
                        cs = cs+1
                res_adv.append(ca/len(t_tok))
                res_nn.append(cn/len(t_tok))
                res_vb.append(cv/len(t_tok))
                res_adj.append(cj/len(t_tok))
                res_ads.append(cs/len(t_tok))
        else: 
            res_mots.append(None)
            res_adv.append(None)
            res_nn.append(None)
            res_vb.append(None)
            res_adj.append(None)
            res_ads.append(None)
    res = pd.DataFrame({'ID':range(len(df.axes[0])),
                        'Nb_mots':res_mots,
                        'Adverbes':res_adv,
                        'Noms':res_nn,
                        'Verbes':res_vb,
                        'Adjectifs':res_adj,
                        'Superlatifs':res_ads})
    return res

# 15 mots les plus frequents, selon le type de mots: 
# ty = type de mot dont on souhaite obtenir le top 15
# Valeurs possibles: 'nom', 'adj', 'adv', 'sup', 'vb'
def top_15(df,ty): 
    # Def. des tags a rechercher: 
    if ty=='nom': tg = ['NN','NNS','NNP','NNPS']
    if ty=='adj': tg = ['JJ','JJR']
    if ty=='sup': tg = ['JJS']
    if ty=='adv': tg = ['RB','RBR','RBS']
    if ty=='vb': tg = ['VB','VBD','VBG','VBN','VBP','VBZ']
    l = {} # dictionnaire qui prendra en cle les mots et en valeur leur nb d'occurrences 
    for k in range(len(df.axes[0])):
        t = df['Review'][k]
        if pd.isna(t)==False: 
            t = t.lower()
            t_tok = nltk.word_tokenize(t)
            if len(t_tok)!=0:
                t_tag = nltk.pos_tag(t_tok)
                for x in t_tag: 
                    if x[1] in tg: 
                        if x[0] in l: 
                            l[x[0]] = l[x[0]]+1
                        else: 
                            l[x[0]]=1
    res = sorted(l.items(), key=lambda x: x[1]) # liste de couples
    # Retirer les 'parasites': 
    pb = ['"','“','’','—','”','t','n','‘','s']
    res = [x for x in res if x[0] not in pb]        
    res15 = res[len(res)-16:len(res)-1]
    return res15

# % verbes conjugues au passe / present / futur: 
def tps_vb(df): 
    pr = ['VBP','VBZ','VBG']
    ps = ['VBD','VBN']
    # VB: cas particulier (present / futur)
    res_pr = []
    res_ps = []
    res_fu = []
    for k in range(len(df.axes[0])):
        t = df['Review'][k]
        cpr = 0
        cps = 0
        cfu = 0
        if pd.isna(t)==False: 
            t_tok = nltk.word_tokenize(t)
            if len(t_tok)==0: # ie si pas de texte de critique
                res_pr.append(None)
                res_ps.append(None)
                res_fu.append(None)
            else: 
                t_tag = nltk.pos_tag(t_tok)
                for j in range(len(t_tag)): 
                    if t_tag[j][1] in pr: 
                        cpr = cpr+1
                    if t_tag[j][1] in ps: 
                        cps = cps+1
                    if t_tag[j][1]=='VB': 
                        if j!=0: 
                            if t_tag[j-1][0]=='will': 
                                cfu = cfu+1
                        else: 
                            cpr = cpr+1
                res_pr.append(cpr)
                res_ps.append(cps)
                res_fu.append(cfu)
        else: 
            res_pr.append(None)
            res_ps.append(None)
            res_fu.append(None)
    res = pd.DataFrame({'ID':range(len(df.axes[0])),
                        'Passe':res_ps,
                        'Present':res_pr,
                        'Futur':res_fu})
    return res

# Pronoms personnels utilises:
def pronoms_perso(df): 
    # Pronoms que l'on veut compter: 
    i = ['I','me','my','mine']
    you = ['you','yours','your']
    he = ['he','she','him','his','her','hers']
    we = ['we','us','our','ours']
    they = ['they','them','their','theirs']
    res_i = [] 
    res_you = [] 
    res_he = [] 
    res_we = [] 
    res_they = [] 
    for k in range(len(df.axes[0])):
        t = df['Review'][k]
        ci = 0
        cy = 0
        ch = 0
        cw = 0
        ct = 0
        if pd.isna(t)==False: 
            t_tok = nltk.word_tokenize(t)
            if len(t_tok)==0: # ie si pas de texte de critique
                res_i.append(None)
                res_you.append(None)
                res_he.append(None)
                res_we.append(None)
                res_they.append(None)
            else: 
                t_tag = nltk.pos_tag(t_tok)
                for x in t_tag: 
                    if x[1]=='PRP' or x[1]=='PRP$':
                        if x[0] in i: 
                            ci = ci+1
                        if x[0] in you: 
                            cy = cy+1
                        if x[0] in he: 
                            ch = ch+1
                        if x[0] in we: 
                            cw = cw+1
                        if x[0] in they: 
                            ct = ct+1
                res_i.append(ci)
                res_you.append(cy)
                res_he.append(ch)
                res_we.append(cw)
                res_they.append(ct)
        else: 
            res_i.append(None)
            res_you.append(None)
            res_he.append(None)
            res_we.append(None)
            res_they.append(None)
    res = pd.DataFrame({'ID':range(len(df.axes[0])),
                        '1e_pers_s':res_i,
                        '2e_pers':res_you,
                        '3e_pers_s':res_he,
                        '1e_pers_p':res_we,
                        '3e_pers_p':res_they})
    return res

# ----------------------------------------------------------------------------

# CORPUS 1 = Theater Record 2010
C1 = pd.read_csv('20201106_C1.csv')
# CORPUS 2 = Blogs (chaque annee)
C2_2020 = pd.read_csv('20201106_C2_2020.csv')
C2_2019 = pd.read_csv('20201106_C2_2019.csv')
C2_2018 = pd.read_csv('20201106_C2_2018.csv')
C2_2017 = pd.read_csv('20201106_C2_2017.csv')
C2_2016 = pd.read_csv('20201106_C2_2016.csv')
C2_2015 = pd.read_csv('20201106_C2_2015.csv')
C2_2014 = pd.read_csv('20201106_C2_2014.csv')
C2_2013 = pd.read_csv('20201106_C2_2013.csv')
C2_2012 = pd.read_csv('20201106_C2_2012.csv')
C2_2011 = pd.read_csv('20201106_C2_2011.csv')
C2_2010 = pd.read_csv('20201106_C2_2010.csv')

# Pour chaque question, on va comparer C1 et chacun des C2. 

########################################
# 1. Syntaxe - Ponctuation interne     #
########################################
# % de virgules / point-virgules / tirets dans la critique

C1_PI = int_punct(C1)
C2_2020_PI = int_punct(C2_2020)
C2_2019_PI = int_punct(C2_2019)
C2_2018_PI = int_punct(C2_2018)
C2_2017_PI = int_punct(C2_2017)
C2_2016_PI = int_punct(C2_2016)
C2_2015_PI = int_punct(C2_2015)
C2_2014_PI = int_punct(C2_2014)
C2_2013_PI = int_punct(C2_2013)
C2_2012_PI = int_punct(C2_2012)
C2_2011_PI = int_punct(C2_2011)
C2_2010_PI = int_punct(C2_2010)

# Stockage: 
C1_PI.to_csv('stockage_df/20201106_C1_PI.csv',na_rep='NA')
C2_2020_PI.to_csv('stockage_df/20201106_C2_2020_PI.csv',na_rep='NA')
C2_2019_PI.to_csv('stockage_df/20201106_C2_2019_PI.csv',na_rep='NA')
C2_2018_PI.to_csv('stockage_df/20201106_C2_2018_PI.csv',na_rep='NA')
C2_2017_PI.to_csv('stockage_df/20201106_C2_2017_PI.csv',na_rep='NA')
C2_2016_PI.to_csv('stockage_df/20201106_C2_2016_PI.csv',na_rep='NA')
C2_2015_PI.to_csv('stockage_df/20201106_C2_2015_PI.csv',na_rep='NA')
C2_2014_PI.to_csv('stockage_df/20201106_C2_2014_PI.csv',na_rep='NA')
C2_2013_PI.to_csv('stockage_df/20201106_C2_2013_PI.csv',na_rep='NA')
C2_2012_PI.to_csv('stockage_df/20201106_C2_2012_PI.csv',na_rep='NA')
C2_2011_PI.to_csv('stockage_df/20201106_C2_2011_PI.csv',na_rep='NA')
C2_2010_PI.to_csv('stockage_df/20201106_C2_2010_PI.csv',na_rep='NA')

# Data-viz: 
# ...

########################################
# 2. Syntaxe - Ponctuation externe     #
########################################
# % phrases declaratives / interrogatives / exclamatives dans la critique

C1_PE = ext_punct(C1)
C2_2020_PE = ext_punct(C2_2020)
C2_2019_PE = ext_punct(C2_2019)
C2_2018_PE = ext_punct(C2_2018)
C2_2017_PE = ext_punct(C2_2017)
C2_2016_PE = ext_punct(C2_2016)
C2_2015_PE = ext_punct(C2_2015)
C2_2014_PE = ext_punct(C2_2014)
C2_2013_PE = ext_punct(C2_2013)
C2_2012_PE = ext_punct(C2_2012)
C2_2011_PE = ext_punct(C2_2011)
C2_2010_PE = ext_punct(C2_2010)

# Stockage: 
C1_PE.to_csv('stockage_df/20201106_C1_PE.csv',na_rep='NA')
C2_2020_PE.to_csv('stockage_df/20201106_C2_2020_PE.csv',na_rep='NA')
C2_2019_PE.to_csv('stockage_df/20201106_C2_2019_PE.csv',na_rep='NA')
C2_2018_PE.to_csv('stockage_df/20201106_C2_2018_PE.csv',na_rep='NA')
C2_2017_PE.to_csv('stockage_df/20201106_C2_2017_PE.csv',na_rep='NA')
C2_2016_PE.to_csv('stockage_df/20201106_C2_2016_PE.csv',na_rep='NA')
C2_2015_PE.to_csv('stockage_df/20201106_C2_2015_PE.csv',na_rep='NA')
C2_2014_PE.to_csv('stockage_df/20201106_C2_2014_PE.csv',na_rep='NA')
C2_2013_PE.to_csv('stockage_df/20201106_C2_2013_PE.csv',na_rep='NA')
C2_2012_PE.to_csv('stockage_df/20201106_C2_2012_PE.csv',na_rep='NA')
C2_2011_PE.to_csv('stockage_df/20201106_C2_2011_PE.csv',na_rep='NA')
C2_2010_PE.to_csv('stockage_df/20201106_C2_2010_PE.csv',na_rep='NA')

# Data-viz: 
# ... 

########################################
# 3. Lexique - Types de mots           #
########################################
# % adverbes, noms, verbes, adjectifs qualificatifs, superlatifs

# adverbs = RB, RBR, RBS
# nouns = NN, NNS, NNP, NNPS
# verbs = VB, VBD, VBG, VBN, VBP, VBZ
# adjectives (quali) = JJ, JJR (on met les adjectifs & comparatifs ensemble)
# adjectives (super) = JJS

C1_type = type_mots(C1)
C2_2020_type = type_mots(C2_2020)
C2_2019_type = type_mots(C2_2019)
C2_2018_type = type_mots(C2_2018)
C2_2017_type = type_mots(C2_2017)
C2_2016_type = type_mots(C2_2016)
C2_2015_type = type_mots(C2_2015)
C2_2014_type = type_mots(C2_2014)
C2_2013_type = type_mots(C2_2013)
C2_2012_type = type_mots(C2_2012)
C2_2011_type = type_mots(C2_2011)
C2_2010_type = type_mots(C2_2010)

# Stockage: 
C1_type.to_csv('stockage_df/20201112_C1_typemots.csv',na_rep='NA')
C2_2020_type.to_csv('stockage_df/20201112_C2_2020_typemots.csv',na_rep='NA')
C2_2019_type.to_csv('stockage_df/20201112_C2_2019_typemots.csv',na_rep='NA')
C2_2018_type.to_csv('stockage_df/20201112_C2_2018_typemots.csv',na_rep='NA')
C2_2017_type.to_csv('stockage_df/20201112_C2_2017_typemots.csv',na_rep='NA')
C2_2016_type.to_csv('stockage_df/20201112_C2_2016_typemots.csv',na_rep='NA')
C2_2015_type.to_csv('stockage_df/20201112_C2_2015_typemots.csv',na_rep='NA')
C2_2014_type.to_csv('stockage_df/20201112_C2_2014_typemots.csv',na_rep='NA')
C2_2013_type.to_csv('stockage_df/20201112_C2_2013_typemots.csv',na_rep='NA')
C2_2012_type.to_csv('stockage_df/20201112_C2_2012_typemots.csv',na_rep='NA')
C2_2011_type.to_csv('stockage_df/20201112_C2_2011_typemots.csv',na_rep='NA')
C2_2010_type.to_csv('stockage_df/20201112_C2_2010_typemots.csv',na_rep='NA')

# Data-viz:
# ...

########################################
# 4. Lexique - Mots les plus frequents #
########################################
# 15 adverbes / noms / verbes / adjectifs / superlatifs les + frequents

top_15(C1,'adv')
# ['now','still','rather','never','then','well','here','as','only','just',
# 'even','too','also','more','so']
top_15(C1,'nom')
#['London','years','theatre','audience','characters','way','story','time',
# 'performance','man','life','stage','show','production','play']
top_15(C1,'vb')
#['set','see','make','being','seems','had','do','does','makes','been',
# 'was','have','be','has','are']
top_15(C1,'adj')
# ['real','many','last','such','old','little','great','good','much','first',
# 'young','other','own','more','new']
top_15(C1,'sup')
#['oldest','forest','youngest','earnest','strongest','funniest','worst','honest',
# 'finest','west','biggest','greatest','latest','least','most']

top_15(C2_2020,'adv')
top_15(C2_2020,'nom')
top_15(C2_2020,'vb')
top_15(C2_2020,'adj')
top_15(C2_2020,'sup')

top_15(C2_2019,'adv')
top_15(C2_2019,'nom')
top_15(C2_2019,'vb')
top_15(C2_2019,'adj')
top_15(C2_2019,'sup')

top_15(C2_2018,'adv')
top_15(C2_2018,'nom')
top_15(C2_2018,'vb')
top_15(C2_2018,'adj')
top_15(C2_2018,'sup')

top_15(C2_2017,'adv')
top_15(C2_2017,'nom')
top_15(C2_2017,'vb')
top_15(C2_2017,'adj')
top_15(C2_2017,'sup')

top_15(C2_2016,'adv')
top_15(C2_2016,'nom')
top_15(C2_2016,'vb')
top_15(C2_2016,'adj')
top_15(C2_2016,'sup')

top_15(C2_2015,'adv')
top_15(C2_2015,'nom')
top_15(C2_2015,'vb')
top_15(C2_2015,'adj')
top_15(C2_2015,'sup')

top_15(C2_2014,'adv')
top_15(C2_2014,'nom')
top_15(C2_2014,'vb')
top_15(C2_2014,'adj')
top_15(C2_2014,'sup')

top_15(C2_2013,'adv')
top_15(C2_2013,'nom')
top_15(C2_2013,'vb')
top_15(C2_2013,'adj')
top_15(C2_2013,'sup')

top_15(C2_2012,'adv')
top_15(C2_2012,'nom')
top_15(C2_2012,'vb')
top_15(C2_2012,'adj')
top_15(C2_2012,'sup')

top_15(C2_2011,'adv')
top_15(C2_2011,'nom')
top_15(C2_2011,'vb')
top_15(C2_2011,'adj')
top_15(C2_2011,'sup')

top_15(C2_2010,'adv')
top_15(C2_2010,'nom')
top_15(C2_2010,'vb')
top_15(C2_2010,'adj')
top_15(C2_2010,'sup')

########################################
# 5. Syntaxe - Temps des verbes        #
########################################
# % verbes passe / present / futur
# present = VBP, VBZ, VBG, VB sans 'will' devant
# passe = VBD, VBN
# futur = VB avec 'will' devant

C1_vb = tps_vb(C1)
C2_2020_vb = tps_vb(C2_2020)
C2_2019_vb = tps_vb(C2_2019)
C2_2018_vb = tps_vb(C2_2018)
C2_2017_vb = tps_vb(C2_2017)
C2_2016_vb = tps_vb(C2_2016)
C2_2015_vb = tps_vb(C2_2015)
C2_2014_vb = tps_vb(C2_2014)
C2_2013_vb = tps_vb(C2_2013)
C2_2012_vb = tps_vb(C2_2012)
C2_2011_vb = tps_vb(C2_2011)
C2_2010_vb = tps_vb(C2_2010)

# Stockage: 
C1_vb.to_csv('stockage_df/20201112_C1_vb.csv',na_rep='NA')
C2_2020_vb.to_csv('stockage_df/20201112_C2_2020_vb.csv',na_rep='NA')
C2_2019_vb.to_csv('stockage_df/20201112_C2_2019_vb.csv',na_rep='NA')
C2_2018_vb.to_csv('stockage_df/20201112_C2_2018_vb.csv',na_rep='NA')
C2_2017_vb.to_csv('stockage_df/20201112_C2_2017_vb.csv',na_rep='NA')
C2_2016_vb.to_csv('stockage_df/20201112_C2_2016_vb.csv',na_rep='NA')
C2_2015_vb.to_csv('stockage_df/20201112_C2_2015_vb.csv',na_rep='NA')
C2_2014_vb.to_csv('stockage_df/20201112_C2_2014_vb.csv',na_rep='NA')
C2_2013_vb.to_csv('stockage_df/20201112_C2_2013_vb.csv',na_rep='NA')
C2_2012_vb.to_csv('stockage_df/20201112_C2_2012_vb.csv',na_rep='NA')
C2_2011_vb.to_csv('stockage_df/20201112_C2_2011_vb.csv',na_rep='NA')
C2_2010_vb.to_csv('stockage_df/20201112_C2_2010_vb.csv',na_rep='NA')

# Data-viz: 
# ...

########################################
# 6. Lexique - Emotions                #
########################################
# Quels pronoms personnels sont utilises ?
# PRP = personal pronouns ; PRP$ = possessive personal pronouns 

C1_pp = pronoms_perso(C1)
C2_2020_pp = pronoms_perso(C2_2020)
C2_2019_pp = pronoms_perso(C2_2019)
C2_2018_pp = pronoms_perso(C2_2018)
C2_2017_pp = pronoms_perso(C2_2017)
C2_2016_pp = pronoms_perso(C2_2016)
C2_2015_pp = pronoms_perso(C2_2015)
C2_2014_pp = pronoms_perso(C2_2014)
C2_2013_pp = pronoms_perso(C2_2013)
C2_2012_pp = pronoms_perso(C2_2012)
C2_2011_pp = pronoms_perso(C2_2011)
C2_2010_pp = pronoms_perso(C2_2010)

# Stockage: 
C1_pp.to_csv('stockage_df/20201112_C1_pp.csv',na_rep='NA')
C2_2020_pp.to_csv('stockage_df/20201112_C2_2020_pp.csv',na_rep='NA')
C2_2019_pp.to_csv('stockage_df/20201112_C2_2019_pp.csv',na_rep='NA')
C2_2018_pp.to_csv('stockage_df/20201112_C2_2018_pp.csv',na_rep='NA')
C2_2017_pp.to_csv('stockage_df/20201112_C2_2017_pp.csv',na_rep='NA')
C2_2016_pp.to_csv('stockage_df/20201112_C2_2016_pp.csv',na_rep='NA')
C2_2015_pp.to_csv('stockage_df/20201112_C2_2015_pp.csv',na_rep='NA')
C2_2014_pp.to_csv('stockage_df/20201112_C2_2014_pp.csv',na_rep='NA')
C2_2013_pp.to_csv('stockage_df/20201112_C2_2013_pp.csv',na_rep='NA')
C2_2012_pp.to_csv('stockage_df/20201112_C2_2012_pp.csv',na_rep='NA')
C2_2011_pp.to_csv('stockage_df/20201112_C2_2011_pp.csv',na_rep='NA')
C2_2010_pp.to_csv('stockage_df/20201112_C2_2010_pp.csv',na_rep='NA')

# Data-viz: 
# ...

########################################
# 7. Autres                            #
########################################

# ...