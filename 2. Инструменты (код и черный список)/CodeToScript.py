def main():

    # download&import NLP libraries and pandas
    import os

    os.system("pip install csv")
    import csv

    os.system("pip install requests")
    import requests

    os.system("pip install tkinter")
    from tkinter import filedialog

    os.system("pip install spacy")
    import spacy

    os.system("spacy download ru_core_news_sm")
    os.system("python -m spacy download ru_core_news_sm")

    os.system("pip install natasha")

    from natasha import (Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger,
                         NewsSyntaxParser, NewsNERTagger, Doc)

    segmenter = Segmenter()
    morph_vocab = MorphVocab()

    emb = NewsEmbedding()
    morph_tagger = NewsMorphTagger(emb)
    syntax_parser = NewsSyntaxParser(emb)
    ner_tagger = NewsNERTagger(emb)

    os.system("pip install pandas")
    import pandas as pd

    currentdirectory = os.getcwd()
    textFolder = filedialog.askdirectory(initialdir=currentdirectory,
                                         title='Choose folder with txt files')
    print('this is textFolder: ', textFolder)

    filesForWork = os.listdir(textFolder)
    print('and this is filesForWork: ', filesForWork)

    filesInDirectory = [f for f in os.listdir(textFolder)]
    print('these are files in dir: ', filesInDirectory)

    nlp = spacy.load("ru_core_news_sm")
    nlp.max_length = 20000000

    for file_name in filesInDirectory:
        with open(f'{textFolder}/{file_name}', 'r', encoding='utf-8') as text:

            book = text.read()

            doc_spacy = nlp(book)

            header = ['spacy', 'start']
            rows = []

            for ent in doc_spacy.ents:
                if ent.label_ == 'LOC':
                    one_row = []
                    one_row = [ent.lemma_, ent.start_char]
                    rows.append(one_row)

            with open('results_spacy.csv', 'w',
                      encoding='utf-8') as file_spacy:
                writer = csv.writer(file_spacy)
                writer.writerow(header)
                writer.writerows(rows)

            doc_natasha = Doc(book)
            doc_natasha.segment(segmenter)
            doc_natasha.tag_morph(morph_tagger)
            doc_natasha.parse_syntax(syntax_parser)
            doc_natasha.tag_ner(ner_tagger)
            locations = []
            for span in doc_natasha.spans:
                if span.type == 'LOC':
                    locations.append(span)
            for span in locations:
                span.normalize(morph_vocab)

            header_natasha = ['natasha', 'start']
            rows_natasha = []
            for span in locations:
                one_row = []
                one_row = [span.normal, span.start]
                rows_natasha.append(one_row)
            with open('results_natasha.csv', 'w', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(header_natasha)
                writer.writerows(rows_natasha)

            natasha_table = pd.read_csv("results_natasha.csv")
            spacy_table = pd.read_csv("results_spacy.csv")
            merged_tables_outer = pd.merge(natasha_table, spacy_table,
                                           on="start", how="outer")
            column_names = ["start", "natasha", "spacy"]
            outer_table = merged_tables_outer.reindex(columns=column_names)
            outer_table.to_csv("outer_table.csv", index=False)
            merged_tables_inner = pd.merge(natasha_table, spacy_table,
                                           on="start", how="inner")
            inner_table = merged_tables_inner.reindex(columns=column_names)
            inner_table.to_csv("inner_table.csv", index=False)

            response = requests.get('https://github.com/Ml-Gn/MOSCOW-TEXT/bl'
                                    'ob/main/2.'
                                    '%20%D0%98%D0%BD%D1%81%D1%82%D1%80%D1%83%D'
                                    '0%BC%D0%B5%D0%BD%D1%82%D1%8B%2'
                                    '0(%D0%BA%D0%BE%D0%B4%20%D0%B8%20%D1%87%D'
                                    '0%B5%D1%80%D0%BD%D1%8B%D0%B9%20%D1%8'
                                    '1%D0%BF%D0%B8%D1%81%D'
                                    '0%BE%D0%BA)/black_list.csv')
            read_obj = response.text
            b_list = csv.reader(read_obj)
            list_of_csv = list(b_list)
            black_list = []
            for i in list_of_csv:
                loc = ''.join(i).lower().split()
                loc = ''.join(loc)
                black_list.append(loc)
            header_clean_spacy = ['spacy', 'start']
            rows_clean_spacy = []

            for ent in doc_spacy.ents:
                if ent.label_ == 'LOC':
                    if ent.lemma_ not in black_list:
                        one_row = []
                        one_row = [ent.lemma_, ent.start_char]
                        rows_clean_spacy.append(one_row)
            with open('results_spacy_clean.csv',
                      'w', encoding='utf-8') as file_spacy_clean:
                writer = csv.writer(file_spacy_clean)
                writer.writerow(header_clean_spacy)
                writer.writerows(rows_clean_spacy)
            header_clean_natasha = ['natasha', 'start']
            rows_clean_natasha = []

            for span in locations:
                if span.normal.lower() not in black_list:
                    one_row = []
                    one_row = [span.normal, span.start]
                    rows_clean_natasha.append(one_row)

            with open('results_natasha_clean.csv',
                      'w', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(header_clean_natasha)
                writer.writerows(rows_clean_natasha)

            natasha_table_clean = pd.read_csv("results_natasha_clean.csv")
            spacy_table_clean = pd.read_csv("results_spacy_clean.csv")
            clean_tables_inner = pd.merge(natasha_table_clean,
                                          spacy_table_clean,
                                          on="start", how="inner")
            clean_tables_inner.to_csv("clean_results.csv", index=False)
            merge_new = pd.merge(inner_table, clean_tables_inner,
                                 on='start', how='outer')
            tables = merge_new.rename(columns={'natasha_x': 'natasha',
                                               'spacy_x': 'spacy',
                                               'natasha_y': 'natasha_clean',
                                               'spacy_y': 'spacy_clean'})
            tables.to_csv("both_results.csv", index=False, encoding='utf-8')
    os.remove("results_spacy.csv")
    os.remove("results_natasha.csv")
    os.remove("inner_table.csv")
    os.remove("outer_table.csv")
    os.remove("results_spacy_clean.csv")
    os.remove("results_natasha_clean.csv")
    os.remove("both_results.csv")
    print('Complete')


if __name__ == '__main__':
    main()
