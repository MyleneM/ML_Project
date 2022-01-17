## NLP App Theatre Reviews

### Import packages

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
from PIL import Image, ImageDraw, ImageFont
import textwrap
from nltk.corpus import stopwords
from pathlib import Path
# Perform standard imports
import spacy
nlp = spacy.load('en_core_web_sm')
from spacy import displacy
from annotated_text import annotated_text
import sumy
import spacy_streamlit
from spacy_streamlit import visualize_tokens
from textblob import TextBlob
from gensim.summarization import summarize
import streamlit.components.v1 as components

# Sumy Summary Pkg
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent, px
from htbuilder.funcs import rgba, rgb

from pathlib import Path
import base64
import time

import text2emotion as te

import seaborn as sns
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


st.set_page_config(
    page_title="Drama Critiques Playground", layout="wide", page_icon="./images/flask.png"
)

def sumy_summarizer(docx):
	parser = PlaintextParser.from_string(docx,Tokenizer("english"))
	lex_summarizer = LexRankSummarizer()
	summary = lex_summarizer(parser.document,3)
	summary_list = [str(sentence) for sentence in summary]
	result = ' '.join(summary_list)
	return result

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded



def main():
    def _max_width_():
        max_width_str = f"max-width: 1000px;"
        st.markdown(
            f"""
        <style>
        .reportview-container .main .block-container{{
            {max_width_str}
        }}
        </style>
        """,
            unsafe_allow_html=True,
        )


    # Hide the Streamlit header and footer
    def hide_header_footer():
        hide_streamlit_style = """
                    <style>
                    footer {visibility: hidden;}
                    </style>
                    """
        st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    # increases the width of the text and tables/figures
    _max_width_()

    # hide the footer
    hide_header_footer()

    images = Image.open('images/binary.png')
    st.image(images, width=400)

    st.markdown("# Behind the Machine 🔍 🖥")
    st.subheader(
        """
        This is a place where you can get familiar with nlp models  directly from your browser 🧪
        """
    )
    st.markdown("     ")

    st.markdown("---")

    st.sidebar.title("Configurations")
    st.sidebar.markdown("---")
    st.sidebar.header("Select Dataset")
    selected_indices = []
    master_review = st.sidebar.text_area("🎭 Paste Review 🙂", "Enter Text")
    if st.sidebar.button("Analyze"):
        print(f"Your copied review:{master_review}")

    st.sidebar.markdown("OR")
    def file_select(folder='./datasets'):
        filelist = os.listdir(folder)
        st.sidebar.markdown("OR")
        selectedfile = st.sidebar.selectbox('', filelist)
        return os.path.join(folder, selectedfile)




    if st.sidebar.button('Upload Data'):
        data = st.sidebar.file_uploader('', type=['CSV'])

        if data is not None:
            df = pd.read_csv(data)
    else:
        filename = file_select()
        st.sidebar.info('You selected {}'.format(filename))
        if filename is not None:
            df = pd.read_csv(filename)
    st.sidebar.markdown("---")
    st.sidebar.header("Select Process Step")
    nlp_steps = st.sidebar.selectbox('', ['00 - Show  Dataset','01 - Show  Review','02 - Basic Information','03 - Tokenization','04 - Lemmatization','05 - Name Entity Recognition','06 - Part of Speech',"07 - Sentiment Analysis","08 - Text Summarization","09 - Zoning","10 - Mapping"])
    index_review = st.sidebar.number_input("Pick an Index of Review", 0, 100)
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        [<img src='data:image/png;base64,{}' class='img-fluid' width=25 height=25>](https://github.com/gaetanbrison/nlp) <small> NLP 4 Critics 1.0.0 | November 2021</small>""".format(
            img_to_bytes("./images/github.png")
        ),
        unsafe_allow_html=True,
    )

    if nlp_steps == "00 - Show  Dataset":
        if master_review == "Enter Text":
            num = st.number_input('No. of Rows', 5, 10)
            head = st.radio('View from top (head) or bottom (tail)', ('Head', 'Tail'))

            if head == 'Head':
                st.dataframe(df.head(num))
                #

            else:
                st.dataframe(df.tail(num))
            st.markdown("---")
        else:
            st.write(f"#### Your copied review: '{master_review}'")
            st.markdown("---")

        snippet = f"""

        >>> import pandas as pd
        >>> import numpy as  as np

        >>> df.head(5)
        #Or
        >>> df.tail(5)

        """
        code_header_placeholder = st.empty()
        snippet_placeholder = st.empty()
        code_header_placeholder.header(f"**Code for the step: 00 - Show  Dataset**")
        snippet_placeholder.code(snippet)


    elif nlp_steps == '01 - Show  Review':
        if master_review == "Enter Text":
            st.markdown("The review you selected is the following one: ")
            st.write('"' + df["Review"][index_review] + '"')
            st.markdown("---")
        else:
            st.markdown("The review you selected is the following one: ")
            st.write('"' + master_review + '"')
            st.markdown("---")

        snippet = f"""

        >>> import pandas as pd
        >>> import numpy as  as np

        >>> df["Review"][0]


        """
        code_header_placeholder = st.empty()
        snippet_placeholder = st.empty()
        code_header_placeholder.header(f"**Code for the step: 01 - Show  Review**")
        snippet_placeholder.code(snippet)

    elif nlp_steps == '02 - Basic Information':
        if master_review == "Enter Text":
            st.markdown("Number of rows and columns helps us to determine how large the dataset is.")
            st.text('* Review number of words')
            st.write(len(df["Review"][index_review].split(" ")))
            st.text('* Review number of characters')
            st.write(len(df["Review"][index_review]))
            st.markdown("---")
        else:
            st.markdown("Number of rows and columns helps us to determine how large the dataset is.")
            st.text('* Review number of words')
            st.write(len(master_review.split(" ")))
            st.text('* Review number of characters')
            st.write(len(master_review))
            st.markdown("---")

        snippet = f"""

        >>> import pandas as pd
        >>> import numpy as  as np

        # Review number of words
        >>> len(df["Review"][0].split(" "))

        # &
        
        # Review number of characters
        >>> len(df["Review"][0])


        """
        code_header_placeholder = st.empty()
        snippet_placeholder = st.empty()
        code_header_placeholder.header(f"**Code for the step: 02 - Basic Information**")
        snippet_placeholder.code(snippet)

    elif nlp_steps == '03 - Tokenization':
        if master_review == "Enter Text":
            st.markdown("This is a view of the text split in tokens for the computer to better understand: ")
            doc = nlp(df["Review"][index_review])
            text = df["Review"][index_review].replace(" "," | ")
            st.write(text)
            st.markdown("---")
        else:
            st.markdown("This is a view of the text split in tokens for the computer to better understand: ")
            doc = nlp(master_review)
            text = master_review.replace(" "," | ")
            st.write(text)
            st.markdown("---")
        snippet = f"""

        >>> import pandas as pd
        >>> import numpy as  as np
        >>> import nltk

        >>> doc = nlp(df["Review"][0])
        >>> text = df["Review"][0].replace(" "," | ")
        >>> print(text)

        """
        code_header_placeholder = st.empty()
        snippet_placeholder = st.empty()
        code_header_placeholder.header(f"**Code for the step: 03 - Tokenization**")
        snippet_placeholder.code(snippet)

    elif nlp_steps == '04 - Lemmatization':
        if master_review == "Enter Text":
            st.markdown(
                "Missing values are known as null or NaN values. Missing data tends to **introduce bias that leads to misleading results.**")

            doc = nlp(df["Review"][index_review])
            list_text = []
            list_pos = []
            list_lemma = []
            list_lemma_ = []
            for token in doc:
                list_text.append(token.text)
                list_pos.append(token.pos_)
                list_lemma.append(token.lemma)
                list_lemma_.append(token.lemma_)
            df_lemmatization = pd.DataFrame(
                {'Text': list_text, 'Position': list_pos, 'Unique Code': list_lemma, 'Lemma': list_lemma_, })
            st.dataframe(df_lemmatization)
            st.markdown("---")
        else:
            st.markdown(
                "Missing values are known as null or NaN values. Missing data tends to **introduce bias that leads to misleading results.**")

            doc = nlp(master_review)
            list_text = []
            list_pos = []
            list_lemma = []
            list_lemma_ = []
            for token in doc:
                list_text.append(token.text)
                list_pos.append(token.pos_)
                list_lemma.append(token.lemma)
                list_lemma_.append(token.lemma_)
            df_lemmatization = pd.DataFrame(
                {'Text': list_text, 'Position': list_pos, 'Unique Code': list_lemma, 'Lemma': list_lemma_, })
            st.dataframe(df_lemmatization)
            st.markdown("---")


        snippet = f"""

        >>> import pandas as pd
        >>> import numpy as  as np
        >>> import nltk

        >>> doc = nlp(df["Review"][0])
        >>> list_text = []
        >>> list_pos = []
        >>> list_lemma = []
        >>> list_lemma_ = []
        >>> for token in doc:
            >>> list_text.append(token.text)
            >>> list_pos.append(token.pos_)
            >>> list_lemma.append(token.lemma)
            >>> list_lemma_.append(token.lemma_)
        >>> df_lemmatization = pd.DataFrame('Text': list_text, 'Position': list_pos, 'Unique Code': list_lemma)
        >>> df_lemmatization

        """
        code_header_placeholder = st.empty()
        snippet_placeholder = st.empty()
        code_header_placeholder.header(f"**Code for the step: 04 - Lemmatization**")
        snippet_placeholder.code(snippet)


    elif nlp_steps == '05 - Name Entity Recognition':
        if master_review == "Enter Text":
            HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""
            st.markdown("This part assign a tag to each name and entity in a review: ")
            #html = displacy.render(doc, style='ent', jupyter=True)
            #html = html.replace("\n\n", "\n")
            #st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)
            docx = nlp(df["Review"][0])
            html = displacy.render(docx, style="ent")
            html = html.replace("\n\n", "\n")
            st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)
            st.markdown("---")
        else:
            HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""
            st.markdown("This part assign a tag to each name and entity in a review: ")
            #html = displacy.render(doc, style='ent', jupyter=True)
            #html = html.replace("\n\n", "\n")
            #st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)
            docx = nlp(master_review)
            html = displacy.render(docx, style="ent")
            html = html.replace("\n\n", "\n")
            st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)
            st.markdown("---")
        snippet = f"""

        >>> import pandas as pd
        >>> import numpy as  as np
        >>> import nltk
        >>> import spacy
        >>> import htbuilder

        >>> HTML_WRAPPER = ""<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem"></div>""

        >>> docx = nlp(df["Review"][0])
        >>> html = displacy.render(docx, style="ent")
        >>> html = html.replace("\n\n", "\n")
        >>> st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)


        """
        code_header_placeholder = st.empty()
        snippet_placeholder = st.empty()
        code_header_placeholder.header(f"**Code for the step: 05 - Name Entity Recognition**")
        snippet_placeholder.code(snippet)

    # check dupication rate
    elif nlp_steps == '06 - Part of Speech':
        if master_review == "Enter Text":
            st.markdown(" Duplication rate is defined as the ratio of  number of duplicates to total records in dataset.")
            doc = nlp(df["Review"][index_review])
            models = ["en_core_web_sm", "/path/to/model"]
            default_text = df["Review"][index_review]
            visualizers = ["ner", "textcat"]
            spacy_streamlit.visualize(models, default_text, visualizers)
            st.markdown("---")
        else :
            st.markdown(" Duplication rate is defined as the ratio of  number of duplicates to total records in dataset.")
            doc = nlp(master_review)
            models = ["en_core_web_sm", "/path/to/model"]
            default_text = master_review
            visualizers = ["ner", "textcat"]
            spacy_streamlit.visualize(models, default_text, visualizers)
            st.markdown("---")
        snippet = f"""

        >>> import pandas as pd
        >>> import numpy as  as np
        >>> import nltk
        >>> import spacy
        >>> import htbuilder

        >>> doc = nlp(df["Review"][0])
        >>> models = ["en_core_web_sm", "/path/to/model"]
        >>> default_text = df["Review"][0]
        >>> visualizers = ["ner", "textcat"]
        >>> spacy_streamlit.visualize(models, default_text, visualizers)


        """
        code_header_placeholder = st.empty()
        snippet_placeholder = st.empty()
        code_header_placeholder.header(f"**Code for the step: 06 - Part of Speech**")
        snippet_placeholder.code(snippet)

    # Sentiment Analysis
    elif nlp_steps == "07 - Sentiment Analysis":
        if master_review == "Enter Text":
            st.subheader("Analyse Your Text")

            message = st.text_area("Enter Text", df["Review"][index_review])
            if st.button("Run Sentiment Analysis"):
                blob = TextBlob(message)
                result_sentiment = blob.sentiment
                result_sentiment_2 = te.get_emotion(message)
                nltk.download('vader_lexicon')

                # configure size of heatmap
                sns.set(rc={'figure.figsize': (35, 3)})

                # function to visualize
                def visualize_sentiments(data):
                    sns.heatmap(pd.DataFrame(data).set_index("Sentence").T, center=0, annot=True, cmap="PiYG")

                # text
                sentence = "To inspire and guide entrepreneurs is where I get my joy of contribution"

                # sentiment analysis
                sid = SentimentIntensityAnalyzer()

                # call method
                st.success(sid.polarity_scores(sentence))

                # heatmap

                st.success(result_sentiment)
                st.success(result_sentiment_2)
            st.markdown("---")
        else:
            st.subheader("Analyse Your Text")

            message = st.text_area("Enter Text", master_review)
            if st.button("Run Sentiment Analysis"):
                blob = TextBlob(message)
                result_sentiment = blob.sentiment
                result_sentiment_2 = te.get_emotion(message)
                nltk.download('vader_lexicon')

                # configure size of heatmap
                sns.set(rc={'figure.figsize': (35, 3)})

                # function to visualize
                def visualize_sentiments(data):
                    sns.heatmap(pd.DataFrame(data).set_index("Sentence").T, center=0, annot=True, cmap="PiYG")

                # text
                sentence = "To inspire and guide entrepreneurs is where I get my joy of contribution"

                # sentiment analysis
                sid = SentimentIntensityAnalyzer()

                # call method
                st.success(sid.polarity_scores(sentence))

                # heatmap

                st.success(result_sentiment)
                st.success(result_sentiment_2)
            st.markdown("---")

        snippet = f"""

        >>> import pandas as pd
        >>> import numpy as  as np
        >>> import nltk
        >>> import spacy

        >>> sid = SentimentIntensityAnalyzer()
        >>> st.success(sid.polarity_scores(sentence))
        >>> blob = TextBlob(message)
        >>> result_sentiment = blob.sentiment
        >>> st.success(result_sentiment)


        """
        code_header_placeholder = st.empty()
        snippet_placeholder = st.empty()
        code_header_placeholder.header(f"**Code for the step: 07 - Sentiment Analysis**")
        snippet_placeholder.code(snippet)


    # Summarization
    elif nlp_steps == "08 - Text Summarization":
        if master_review == "Enter Text":
            st.subheader("Summarize Your Text")

            message2 = st.text_area("Review",df["Review"][index_review])
            summary_options = st.selectbox("Choose Summarizer", ['sumy', 'gensim'])
            if st.button("Summarize"):
                if summary_options == 'sumy':
                    st.text("Using Sumy Summarizer ..")
                    summary_result = sumy_summarizer(message2)
                elif summary_options == 'gensim':
                    st.text("Using Gensim Summarizer ..")
                    summary_result = summarize(message2)
                else:
                    st.warning("Using Default Summarizer")
                    st.text("Using Gensim Summarizer ..")
                    summary_result = summarize(message2)
                st.success(summary_result)
            st.markdown("---")
        else:
            st.subheader("Summarize Your Text")

            message2 = st.text_area("Review",master_review)
            summary_options = st.selectbox("Choose Summarizer", ['sumy', 'gensim'])
            if st.button("Summarize"):
                if summary_options == 'sumy':
                    st.text("Using Sumy Summarizer ..")
                    summary_result = sumy_summarizer(message2)
                elif summary_options == 'gensim':
                    st.text("Using Gensim Summarizer ..")
                    summary_result = summarize(message2)
                else:
                    st.warning("Using Default Summarizer")
                    st.text("Using Gensim Summarizer ..")
                    summary_result = summarize(message2)
                st.success(summary_result)
            st.markdown("---")



        snippet = f"""

        >>> import pandas as pd
        >>> import numpy as  as np
        >>> import nltk
        >>> import sumy

        >>> summary_result = summarize(message)

        """
        code_header_placeholder = st.empty()
        snippet_placeholder = st.empty()
        code_header_placeholder.header(f"**Code for the step: 08 - Text Summarization**")
        snippet_placeholder.code(snippet)


    # Summarization
    elif nlp_steps == "09 - Zoning":
        if master_review == "Enter Text":
            st.subheader("Creation of the zoning")
            images = Image.open('images/zoning.png')
            st.image(images, width=None)
            st.markdown("---")

        else:
            st.subheader("Creation of the zoning")
            images = Image.open('images/zoning.png')
            st.image(images, width=None)
            st.markdown("---")
        snippet = f"""

        >>> import pandas as pd
        >>> import numpy as  as np
        >>> import nltk


        >>> work in progress

        """
        code_header_placeholder = st.empty()
        snippet_placeholder = st.empty()
        code_header_placeholder.header(f"**Code for the step: 09 - Zoning**")
        snippet_placeholder.code(snippet)


    elif nlp_steps == "10 - Mapping":
        st.header("Map reviews")
        HtmlFile = open("corpus_I_map_v2.html", 'r', encoding='utf-8')
        source_code = HtmlFile.read()
        print(source_code)
        components.html(source_code, height = 600)
        st.markdown("---")




if __name__=='__main__':
    main()

st.markdown(" ")
st.markdown("### ** 👨🏼‍💻 App Contributors: **")
st.image(['images/mylene.png','images/gaetan.png'], width=100,caption=["Mylène","Gaëtan"])

st.markdown(f"####  Link to Project Website [here]({'https://dramacritiques.com/fr/accueil/'}) 🚀 ")



def image(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))


def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)


def layout(*args):

    style = """
    <style>
      # MainMenu {visibility: hidden;}
      footer {visibility: hidden;background - color: white}
     .stApp { bottom: 80px; }
    </style>
    """
    style_div = styles(
        position="fixed",
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        color="black",
        text_align="center",
        height="auto",
        opacity=1,

    )

    style_hr = styles(
        display="block",
        margin=px(8, 8, "auto", "auto"),
        border_style="inset",
        border_width=px(2)
    )

    body = p()
    foot = div(
        style=style_div
    )(
        hr(
            style=style_hr
        ),
        body
    )

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)

        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)

def footer2():
    myargs = [
        " Made by ",
        link("https://odhn.ens.psl.eu/en/newsroom/dans-les-coulisses-des-humanites-numeriques", "Mylène & Gaëtan"),
        " 👩🏼‍💻 👨🏼‍💻"
    ]
    layout(*myargs)


if __name__ == "__main__":
    footer2()



