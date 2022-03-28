### Hi there 👋 

## Data Engineering Approach 

<img align="center" src="https://cdn-icons-png.flaticon.com/512/2861/2861645.png" width="100">

* In the dataset "zoning_final.xlsx" we are looking at 50 reviews divided in the following zones

<table>
  <tr>
    <th>Color</th>
    <th>Signification</th>
  </tr>
  <tr>
    <td>Purple</td>
    <td>Introduction</td>
  </tr>
  <tr>
    <td>Red</td>
    <td>Plot</td>
  </tr>
  <tr>
    <td>Blue</td>
    <td>General Analysis of the Play</td>
  </tr>
  <tr>
    <td>Green</td>
    <td>Visual, Auditory and Audible Details</td>
  </tr>
  <tr>
    <td>Orange</td>
    <td>Actors’ Performances</td>
  </tr>
  <tr>
    <td>Brown</td>
    <td>Remarks on the Structure of the Play</td>
  </tr>
  <tr>
    <td>Yellow</td>
    <td>Conclusion</td>
  </tr>
</table>


* There are two types of variables the "Reviews" one that are describing the review in its completness and the "Zones" one that are only describing the precise zone of the row

* The target variable is "Target_Variable_Sentence_Type"

* More than 177 variables have been created

* The variables are the following one:

['Zone - Main_Sentence_x', 'Zone - Women/Feminism', 'Zone - LGBT / Queer / Sexuality / Gender', 'Zone - Disabled', 'Zone - Politics', 'Zone - Supernatural', 'Zone - Science', 'Zone - Body', 'Zone - Childhood', 'Zone - Cultural difference / Race', 'Zone - Death', 'Zone - Education', 'Zone - Family', 'Zone - Environment', 'Zone - Friendship', 'Zone - Love', 'Zone - Identity', 'Zone - Memory', 'Zone - Relationships', 'Zone - Religion', 'Zone - Violence', 'Zone - Spectator_Cat', 'Zone - Audience_Cat', 'Zone - Show_Cat', 'Zone - Story_Cat', 'Zone - Decor_Cat', 'Zone - Theatre_Cat', 'Zone - Production_Cat', 'Zone - Character_Cat', 'Zone - End_Cat', 'Zone - Beginning_Cat', 'Zone - Max_Value', 'Zone - Cluster', 'Zone - Neg', 'Zone - Neu', 'Zone - Pos', 'Zone - Compound', 'Zone - Polarity', 'Zone - Subjectivity', 'Zone - Emotion', 'Zone - Angry', 'Zone - Fear', 'Zone - Happy', 'Zone - Sad', 'Zone - Surpise', 'Zone - Top_10_Keywords', 'Zone - Count_Words', 'Zone - Question_Mark', 'Zone - Exclamation_Mark', 'Zone - Word_Average', 'Zone - Main_Sentence_y', 'Zone - Sentence_Type', 'Zone - Type_word_k', 'Zone - Type_word_v', 'Zone - ID', 'Zone - Virgules_pct', 'Zone - Point_virgules_pct', 'Zone - Tirets_pct', 'Zone - ID', 'Zone - Declaratives_pct', 'Zone - Interrogatives_pct', 'Zone - Exclamatives_pct', 'Zone - ID', 'Zone - Nb_mots', 'Zone - Adverbes', 'Zone - Noms', 'Zone - Verbes', 'Zone - Adjectifs', 'Zone - Superlatifs', 'Zone - ID', 'Zone - Passe', 'Zone - Present', 'Zone - Futur', 'Zone - ID', 'Zone - 1e_pers_s', 'Zone - 2e_pers', 'Zone - 3e_pers_s', 'Zone - 1e_pers_p', 'Zone - 3e_pers_p', 'Review - Blog', 'Review - Publication Date', 'Review - Reviewer', 'Review - Title of the Play', 'Review - Playwright', 'Review - Theatre  ', 'Review - Review with Zoning', 'Review - Intro-Purple', 'Review - Red-Plot', 'Review - Blue-General', 'Review - Green-Details', 'Review - Orange-Perf', 'Review - Borwn-Structure', 'Review - Yellow-Conclusion', 'Review - Latitude', 'Review - Longitude', 'Review - Women/Feminism', 'Review - LGBT / Queer / Sexuality / Gender', 'Review - Disabled', 'Review - Politics', 'Review - Supernatural', 'Review - Science', 'Review - Body', 'Review - Childhood', 'Review - Cultural difference / Race', 'Review - Death', 'Review - Education', 'Review - Family', 'Review - Environment', 'Review - Friendship', 'Review - Love', 'Review - Identity', 'Review - Memory', 'Review - Relationships', 'Review - Religion', 'Review - Violence', 'Review - Spectator_Cat', 'Review - Audience_Cat', 'Review - Show_Cat', 'Review - Story_Cat', 'Review - Decor_Cat', 'Review - Theatre_Cat', 'Review - Production_Cat', 'Review - Character_Cat', 'Review - End_Cat', 'Review - Beginning_Cat', 'Review - Max_Value', 'Review - Cluster', 'Review - Neg', 'Review - Neu', 'Review - Pos', 'Review - Compound', 'Review - Polarity', 'Review - Subjectivity', 'Review - Angry', 'Review - Fear', 'Review - Happy', 'Review - Sad', 'Review - Surpise', 'Review - Top_10_Keywords', 'Review - Start_Review_1', 'Review - End_Review_1', 'Review - Start_Review_2', 'Review - End_Review_2', 'Review - Start_Review_3', 'Review - End_Review_3', 'Review - Nbr_of_Sentences', 'Review - Length_Start_Count_Words_1', 'Review - Length_End_Count_Words_1', 'Review - Length_Start_Count_Words_2', 'Review - Length_End_Count_Words_2', 'Review - Length_Start_Count_Words_3', 'Review - Length_End_Count_Words_3', 'Review - Question_Mark_Start_1', 'Review - Question_Mark_End_1', 'Review - Question_Mark_Start_2', 'Review - Question_Mark_End_2', 'Review - Question_Mark_Start_3', 'Review - Question_Mark_End_3', 'Review - Exclamation_Mark_Start_1', 'Review - Exclamation_Mark_End_1', 'Review - Exclamation_Mark_Start_2', 'Review - Exclamation_Mark_End_2', 'Review - Exclamation_Mark_Start_3', 'Review - Exclamation_Mark_End_3', 'Review - Word_Average_Start_1', 'Review - Word_Average_End_1', 'Review - Word_Average_Start_2', 'Review - Word_Average_End_2', 'Review - Word_Average_Start_3', 'Review - Word_Average_End_3', 'Review - Main_Sentence', 'Target_Variable_Sentence_Type', 'Zone - Start Index', 'Zone - Count Character', 'Review - Count Character', 'Zone - Count Word', 'Review - Count Word']








