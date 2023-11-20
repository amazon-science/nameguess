import json
import re
import nltk
import wordninja
from typing import List, Tuple, Dict
from sklearn.metrics import classification_report
from nltk.stem.wordnet import WordNetLemmatizer

class CrypticIdentifier:
    """Module to identify any cryptic forms in a column header.
    Example usage: 
        identifier = CrypticIdentifier(vocab_file)
        identifier.iscryptic("newyorkcitytotalpopulation") --> False
        identifier.iscryptic("tot_revq4") --> True
    """
    def __init__(self, vocab_file=None, word_rank_file=None, k_whole=4, k_split=2):
        """
        Args:
            vocab_file (str, optional): json file containing the vocabulary. Defaults to None.
            k_whole (int, optional): length threshold for a whole string to be considered non-cryptic if it fails the first round of check (i.e.
            _iscryptic returns True). Defaults to 4.
            k_split (int, optional): length threshold for each word split (wordninja.split()) from the string to be considered non-cryptic, if the pre-split string fails the first round of check (i.e.
            _iscryptic returns True). Defaults to 2.
        """
        if vocab_file is not None:
            with open(vocab_file, "r") as fi:
                self.vocab = json.load(fi)
                print("#vocab={}".format(len(self.vocab)))
        else:
            self.vocab = None

        self.k_whole = k_whole
        self.k_split = k_split
        if word_rank_file is None:
            self.splitter = wordninja
        else:
            self.splitter = wordninja.LanguageModel(word_rank_file)
        self.lem = WordNetLemmatizer()
        

    def split_rm_punc(self, text: str) -> list:
        return re.sub(r'[^\w\s]', ' ', text).split()

    def separate_camel_case(self, text: str) -> list:
        return re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', text))

    def convert2base(self, text: str) -> str:
        return self.lem.lemmatize(text)

    def _split(self, text: str) -> list:
        text = text.replace('_', ' ')
        words = self.split_rm_punc(self.separate_camel_case(text))
        return words

    def _iscryptic(self, text: str) -> bool:
        words = self._split(text)
        if all([word.isnumeric() for word in words]):
            return True
        if self.vocab is None:
            self.vocab = nltk.corpus.wordnet.words('english')

        return any([self.convert2base(w).lower() not in self.vocab for w in words])



    def doublecheck_cryptic(self, text: str) -> Tuple[bool, List[str]]:
        """Double-check whether a column header contains cryptic terms. For example in some cases where neither 
        delimiters between tokens nor camelcases is available

        Args:
            text (str): column header

        Returns:
            Tuple[
                    bool: whether header is cryptic
                    List[str]: splitted tokens from the header
                ]
        """

        #stopwords = nltk.corpus.stopwords.words('english')

        def split_check(words: List[str]) -> Tuple[bool, List[str]]:
            l_cryptic = []
            for ele in words:
                if ele.isdigit():
                    l_cryptic.append(False)
                ## Cornercases includes stopwords like "I", "for", etc.
                elif len(ele) < self.k_split: # and ele.lower() not in stopwords:
                    l_cryptic.append(True)
                ## Second round check
                else:
                    l_cryptic.append(self._iscryptic(ele))
            return any(l_cryptic), words
            
        if len(text) >= self.k_whole:
            if self._iscryptic(text):
                split = self.splitter.split(text)
                return split_check(split)            
            else:
                # return (False, self.splitter.split(text))
                return (False, self._split(text))
        else:
            return (True, [text])

    def iscryptic(self, text: str) -> bool:
        return self.doublecheck_cryptic(text)[0]
    
    def split_results(self, text: str) -> List[str]:
        return self.doublecheck_cryptic(text)[1]


def read_gt_names(file_name):
    names = {}
    with open(file_name, 'r') as f:
        for line in f:
            word = line.strip('\n').lower()
            names[word] = True
    return names


import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, required=True)
    parser.add_argument('--vocab_path', type=str, default="./lookups/wordnet.json")
    parser.add_argument('--word_rank_path', type=str, default="./lookups/wordninja_words_alpha.txt.gz")
    args = parser.parse_args()

    identifier = CrypticIdentifier(args.vocab_path, args.word_rank_path)
    is_cryptic, split_res = identifier.doublecheck_cryptic(args.text)
    print(f"Logical name: {not is_cryptic}")
    print(f"Split result: {split_res}")
    