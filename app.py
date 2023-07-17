import streamlit as st #Streamlit import
import pandas as pd #Pandas import
import matplotlib.pyplot as plt
import nltk 
nltk.download('punkt')
from nltk.corpus import stopwords
from wordcloud import WordCloud
import neattext as nt
import neattext.functions as nfx
import gensim as gn
import spacy
from spacy import displacy
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
#nlp = English()
# Create a blank Tokenizer with just the English vocab
tokenizer = Tokenizer(nltk.vocab)

#nlp = spacy.load('en_core_web_sm')

#text Downloader
import base64
import time
timestr = time.strftime("%Y%m%d-%H%M%S")

                      
                      
HTML_WRAPPER = """<div style="overflow-x: auto; border: 2px solid #e6e9ef </div> """

def text_downloader(raw_text):
    b64= base64.b64encode(raw_text.encode()).decode()
    new_file_name = "Clean_text_result_{}_.txt".format(timestr)
    st.markdown('### Download File ###')
    href = f'<a href="data:file/txt;base64,{b64}" download="{new_file_name}">click here!</a>'
    st.markdown(href, unsafe_allow_html=True)

def make_downloadable(data):
    csvfile = data.to_csv(index=False)
    b64 = base64.b64.encode(csvfile.encode()).decode()
    new_filename = "nlp_result_{}_.csv".format(timestr)
    st.markdown("### ** Download CSV File ** ###")
    href = f'<a href="data:file/csv;base64,{b64}" download="{new_filemane}">Click Here!</a>'
    st.markdown(href, unsafe_allow_html=True)

    
def text_analysis(my_text):
    docx = nlp(my_text)
    #token = nlp.tokenizer(my_text)
    alldata = [(token.text, token.pos_, token.tag_, token.lemma_, token.is_alpha, token.is_stopword_)]
    df = pd.DataFrame(alldata, columns=['Token', 'PoS', 'Lemma', 'IsAlpha', 'IsStopWord'])
    return df

def get_entities(my_text):
                      docx = nlp(my_text)
                      entities = [(entity.text, entity.label_) for entity in doc.ents]
                      return entities


def wordcloud_plot(my_text):
        wordcloud_t = WordCloud().generate(my_text)
        fig = plt.figure()
        plt.imshow(wordcloud_t, interpolation='bilinear')
        plt.axiss("off")
        st.pyplot(fig)
                             
     

def main():
    st.title("Text Cleaning App")
    menu = ['Text File', 'CSV File']
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == 'Text File':
        #normalize_case = st.sidebar.checkbox("Normalize Case")
        text_file = st.file_uploader("Upload Txt File", type=['txt'])
        normalize_case = st.sidebar.checkbox("Normalized Case (Lower Case)")
        clean_stopwords = st.sidebar.checkbox("Stopwords Removal")
        clean_punctuations = st.sidebar.checkbox("Punctuations Removal")
        clean_special_char = st.sidebar.checkbox("Special Characters Removal")
        clean_numbers = st.sidebar.checkbox("Numbers Removal")
        clean_url = st.sidebar.checkbox("URL Removal")
        if text_file is not None:
            
            file_details = {"Filename":text_file.name, 
                            "Filesize":text_file.size, 
                            "Filetype":text_file.type}
            st.write(file_details)
            raw_text = text_file.read().decode('utf-8')
            col1, col2 = st.columns(2)
        
            
            with col1:
                with st.expander("Original Text"):
                    #raw_text = text_file.read().decode('utf-8')
                    #file_details= {"Filename":text_file.name,
                                   #"Filesize":text_file.size, 
                                  #"Filetype": text_file.type}
                    st.write(raw_text)
                    
                    
                    
            with col2:
                with st.expander("Processed Text"):
                    if normalize_case:
                        raw_text = raw_text.lower()
                    
                    if clean_stopwords:
                        raw_text = nfx.remove_stopwords(raw_text)
                    
                    if clean_numbers:
                        raw_text = nfx.remove_numbers(raw_text)
                    
                    if clean_url:
                        raw_text = nfx.remove_urls(raw_text)
                    if clean_punctuations:
                        raw_text = nfx.remove_punctuations(raw_text)
                        
                    if clean_special_char:
                        raw_text = nfx.remove_special_characters(raw_text)
                    st.write(raw_text)
                    
                    text_downloader(raw_text)
              
                  
            with st.expander("Text Analysis"):
                token_result_df = text_analysis(raw_text)
                st.dataframe(token_result_df)
                make_downloadable(token_result_df)
              
               
                  
            with st.expander("WordCloud Plot"):
                wordcloud_plot(raw_text)
              
               
            with st.expander("Parts of Speech Tags"):
                fig = plt.figure()
                sns.countplot(token_result_df['PoS'])
                plt.xticks(rotation=45)
                st.pyplot(fig)
                  
                    
                    
                    
                    
        
                    
                    
    


    else:
        csv_file = st.file_uploader("Upload csv file", type=['csv'])








if __name__ == '__main__':
    main()

