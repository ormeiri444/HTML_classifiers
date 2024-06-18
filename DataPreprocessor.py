import pandas as pd
import re


class DataPreprocessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = pd.read_csv(filepath)

    @staticmethod
    def clean_text(text):
        text = str(text)
        cleaned_text = re.sub(r'(?<=\b\w)-|-(?=\w\b)|[^\w\s-]', '', text)
        return cleaned_text

    @staticmethod
    def normalize_text(text):
        date_patterns = [
            r'\b\d{1,2}[\/.-]\d{1,2}[\/.-]\d{2,4}\b',
            r'\b\d{4}[\/.-]\d{1,2}[\/.-]\d{1,2}\b',
            r'\b\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b',
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},'
            r'?\s+\d{4}\b',
            r'\b\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s'
            r'+\d{4}\b',
        ]
        date_regex = '|'.join(date_patterns)

        text = re.sub(date_regex, '<date>', text)
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '<url>',
                      text)
        text = re.sub(r'\b[\w.-]+?@\w+?\.\w{2,4}\b', '<email>', text)
        text = re.sub(r'\$\d+(?:\.\d{2})?', '<money>', text)
        text = re.sub(r'\b\d{3}-\d{4}\b', '<phone>', text)

        replacements = {
            '“': '"', '”': '"', '‘': "'", '’': "'",
            'â\x80\x9c': '"', 'â\x80\x9d': '"', 'â\x80\x99': "'"
        }
        for bad_char, good_char in replacements.items():
            text = text.replace(bad_char, good_char)

        text = text.replace('\n', ' ').replace('\r', ' ')
        text = text.lower()

        punctuation = ['\n', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<',
                       '=',
                       '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']
        for symbol in punctuation:
            text = text.replace(symbol, '')

        return text

    def preprocess(self):
        self.df['cleaned_text'] = self.df['extracted_text'].apply(self.clean_text)
        X = self.df[self.df['cleaned_text'] != 'nan']
        X['cleaned_text'] = X['cleaned_text'].apply(self.normalize_text)
        combined_df = X.dropna(subset=['quality_mean', 'accessibility_mean'])
        combined_df = combined_df[['cleaned_text', 'quality_mean', 'accessibility_mean']]
        return combined_df
