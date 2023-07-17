import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud

nltk.download('stopwords')
nltk.download('punkt')

# Text Downloader
import base64
import time

timestr = time.strftime("%Y%m%d-%H%M%S")


def text_downloader(raw_text):
    b64 = base64.b64encode(raw_text.encode()).decode()
    new_file_name = f"Clean_text_result_{timestr}.txt"
    st.markdown('### Download File ###')
    href = f'<a href="data:file/txt;base64,{b64}" download="{new_file_name}">click here!</a>'
    st.markdown(href, unsafe_allow_html=True)


def make_downloadable(data):
    csvfile = data.to_csv(index=False)
    b64 = base64.b64encode(csvfile.encode()).decode()
    new_filename = f"nlp_result_{timestr}.csv"
    st.markdown("### ** Download CSV File ** ###")
    href = f'<a href="data:file/csv;base64,{b64}" download="{new_filename}">Click Here!</a>'
    st.markdown(href, unsafe_allow_html=True)


def text_analysis(my_text):
    tokens = word_tokenize(my_text)
    stopwords_list = set(stopwords.words('english'))
    stopwords_removed = [token for token in tokens if token.lower() not in stopwords_list]
    pos_tags = nltk.pos_tag(stopwords_removed)
    df = pd.DataFrame(pos_tags, columns=['Token', 'PoS'])
    return df


def wordcloud_plot(my_text):
    wordcloud_t = WordCloud().generate(my_text)
    fig = plt.figure()
    plt.imshow(wordcloud_t, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(fig)


def main():
    st.title("Text Cleaning App")
    menu = ['Text File', 'CSV File']
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == 'Text File':
        text_file = st.file_uploader("Upload Txt File", type=['txt'])
        normalize_case = st.sidebar.checkbox("Normalized Case (Lower Case)")
        clean_stopwords = st.sidebar.checkbox("Stopwords Removal")
        clean_punctuations = st.sidebar.checkbox("Punctuations Removal")
        clean_special_char = st.sidebar.checkbox("Special Characters Removal")
        clean_numbers = st.sidebar.checkbox("Numbers Removal")
        clean_url = st.sidebar.checkbox("URL Removal")

        if text_file is not None:
            file_details = {"Filename": text_file.name,
                            "Filesize": text_file.size,
                            "Filetype": text_file.type}
            st.write(file_details)

            raw_text = text_file.read().decode('utf-8')
            col1, col2 = st.columns(2)

            with col1:
                with st.expander("Original Text"):
                    st.write(raw_text)

            with col2:
                with st.expander("Processed Text"):
                    if normalize_case:
                        raw_text = raw_text.lower()

                    if clean_stopwords:
                        tokens = word_tokenize(raw_text)
                        stopwords_list = set(stopwords.words('english'))
                        stopwords_removed = [token for token in tokens if token.lower() not in stopwords_list]
                        raw_text = " ".join(stopwords_removed)

                    if clean_numbers:
                        raw_text = ''.join(c for c in raw_text if not c.isdigit())

                    if clean_url:
                        raw_text = raw_text.replace("http://", "").replace("https://", "")

                    if clean_punctuations:
                        raw_text = ''.join(c for c in raw_text if c.isalnum() or c.isspace())

                    if clean_special_char:
                        raw_text = ''.join(c for c in raw_text if c.isalnum() or c.isspace())

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
                token_result_df['PoS'].value_counts().plot(kind='bar')
                plt.xticks(rotation=45)
                st.pyplot(fig)

    else:
        csv_file = st.file_uploader("Upload CSV file", type=['csv'])

        if csv_file is not None:
            file_details = {"Filename": csv_file.name,
                            "Filesize": csv_file.size,
                            "Filetype": csv_file.type}
            st.write(file_details)

            df = pd.read_csv(csv_file)
            st.dataframe(df)

            preprocess_columns = st.multiselect("Select Columns to Preprocess", df.columns)
            normalize_case = st.checkbox("Normalize Case (Lower Case)")
            clean_stopwords = st.checkbox("Stopwords Removal")
            clean_punctuations = st.checkbox("Punctuations Removal")
            clean_special_char = st.checkbox("Special Characters Removal")
            clean_numbers = st.checkbox("Numbers Removal")
            clean_url = st.checkbox("URL Removal")

            new_columns = []
            for column in preprocess_columns:
                new_column = f"preprocessed_{column}"
                new_columns.append(new_column)

                processed_data = df[column].astype(str)
                if normalize_case:
                    processed_data = processed_data.str.lower()

                if clean_stopwords:
                    tokens = processed_data.apply(word_tokenize)
                    stopwords_list = set(stopwords.words('english'))
                    processed_data = tokens.apply(lambda x: " ".join([token for token in x if token.lower() not in stopwords_list]))

                if clean_numbers:
                    processed_data = processed_data.str.replace(r'\d+', '')

                if clean_url:
                    processed_data = processed_data.str.replace("http://", "").replace("https://", "")

                if clean_punctuations or clean_special_char:
                    processed_data = processed_data.str.replace(r'[^\w\s]+', '')

                df[new_column] = processed_data

            st.dataframe(df[new_columns])

            make_downloadable(df[new_columns])


if __name__ == '__main__':
    main()
