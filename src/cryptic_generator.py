import random 
import re
import numpy as np
import nltk
from typing import Dict

nltk.download('stopwords')

class CrypticNameGenerator:
    def __init__(self, per_tok_target_len, lookup_abbreviation,
                 p_filter_acronym, lookup_acronym,
                 pr_keep_k, pr_remove_vowels, pr_logic,
                 pm_as_is, pm_lookup, pm_selected_rule, 
                 seed):
        """_summary_
        Class for automatic cryptic name generation from table column headers

        Args:
            per_tok_target_len (int): the target length when abbreviating each token through rules
            lookup_abbreviation (dict): a lookup tables containing (expansion, abbreviation) pairs
            lookup_acronym (dict): a lookup tables containing (expansion, acronym) pairs
            p_filter_acronym (float): probability of filtering and replacing the subsequence by an acronym from the acronym lookup dictionary.
            pr_keep_k (float): for rules, the probability of choosing rule 1: keep the first k characters
            pr_remove_vowels (float): for rules, the probability of choosing rule 2: remove all non-leading vowels
            pr_logic (float): for rules, the probability of choosing rule 3: logic from https://docs.tibco.com/pub/enterprise-runtime-for-R/4.1.1/doc/html/Language_Reference/base/abbreviate.html
            pm_as_is (float): for token-level methods, the probability of choosing token-level method 1: keep the token as-is
            pm_lookup (float): for token-level methods, the probability of choosing token-level method 2: generate abbreviation through lookup table
            pm_selected_rule (float): or token-level methods, the probability of choosing token-level method 3: use rules selected from (pr_keep_k, pr_remove_vowels, pr_logic)
        """
        self.per_tok_target_len = per_tok_target_len
        self.lookup_abbreviation = lookup_abbreviation
        self.lookup_acronym = lookup_acronym
        self.p_filter_acronym = p_filter_acronym
        self.pr_keep_k = pr_keep_k
        self.pr_remove_vowels = pr_remove_vowels
        self.pr_logic = pr_logic
        self.pm_as_is = pm_as_is
        self.pm_lookup = pm_lookup
        self.pm_selected_rule = pm_selected_rule
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        self.stemmer = nltk.stem.PorterStemmer()
        self.seed = seed
        random.seed(self.seed)
        
    def rule_keep_k(self, query: str) -> str:
        """
        Rule 1: Keep first k characters in a word
        """
        return query[:self.per_tok_target_len] if len(query) > self.per_tok_target_len else query

    def rule_remove_vowels(self, query: str) -> str:
        """
        Rule 2: Keep removing non-leading vowels until the threshold or all non-leading vowels have been removed
        """
        start, elems = query[0], list(query)[1:]
        
        vow_idx = [i for i, val in enumerate(elems) if val in ('a', 'e', 'i', 'o', 'u')]
        counter_vow = len(vow_idx)
        counter_truncate = len(query)
        if len(query) > self.per_tok_target_len and vow_idx: 
            while counter_truncate >= self.per_tok_target_len and counter_vow > 0:
                elems[vow_idx[counter_vow-1]] = ""
                counter_vow -= 1
                counter_truncate -= 1

        return start + "".join(elems)

    def rule_logic(self, query: str) -> str:
        """ 
        Rule 3:
        Code contributed by Nicholas Hespe @nahespe

        The abbreviation algorithm does not simply truncate. 
        It has a threshold, according to which it will drop, in order:

            1. duplicate values next to eachother
            2. lower case vowels.
            3. lower case consonants and punctuation.
            4. upper case letters and special characters.  
        
        exits if target_len <= 2
        
        """
        start, elems = query[0], list(query)[1:]
        
        ## exit early if not valid
        if len(elems) < self.per_tok_target_len: 
            return start + "".join(elems)
        
        counter = len(elems)
        while counter >= self.per_tok_target_len:
            counter -= 1
            
            ## remove duplicates next to eachother
            candidates = [i for i in range(len(elems[:-1])) if (elems[i] and elems[i]==elems[i+1])]
            if candidates:
                choice = random.choice(candidates)
                elems[choice] = ""
                continue
                
            ## search for vowels and remove right to left
            candidates = [i for i, val in enumerate(elems) if val in ('a', 'e', 'i', 'o', 'u')]
            if candidates:
                choice = random.choice(candidates)
                elems[choice] = ""
                continue
            
            ## Search for  lower case consonants and remove randomly
            candidates = [i for i, val in enumerate(elems) if (val and not val in ('a', 'e', 'i', 'o', 'u'))]
            if candidates:
                choice = random.choice(candidates)
                elems[choice] = ""
            
        return start + "".join(elems)

    def select_from_probs(self, probs: list, epsilon: float=1e-8) -> int:
        """
        Make random selection based on the probabilities of each index
        """
        assert abs(np.sum(probs) - 1) < epsilon, 'Sampling probabilities must add up tp 1.'
       
        rand = random.uniform(0, 1)

        def cum_sum(l):
            sum = 0
            new_l = [0]
            for ele in l:
                sum += ele
                new_l.append(sum) 
            return new_l

        probs_cum = cum_sum(probs)
        for i, this_level in enumerate(probs_cum[:-1]):
            next_level = probs_cum[i + 1]
            if this_level <= rand < next_level:
                return i
            else:
                pass

    def tokenize(self, text: str, 
                   keep_punc: bool=True, 
                   keep_stopwords: bool=True,
                   split_camelcase: bool=True,
                   use_stem: bool=False) -> list:
        """_summary_
        Split the text into words and punctuations

        Args:
            text (str): input string
            keep_punc (bool, optional): whether to keep non-alphanumeric symbols. Defaults to True.
            keep_stopwords (bool, optional): whether to keep stop words. Defaults to False.
            split_camelcase (bool, optional): whether to split camelCased words (i.e. "camelCase" -> "camel Case"). Defaults to True.
            use_stem (bool, optional): whether to use stemmer
        Returns:
            list: a list of tokens
        """
        def split_with_punc(text: str) -> list:
            return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

        def separate_camel_case(text: str) -> list:
            return re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', text))

        text = text.replace('_', ' ')
        if split_camelcase:
            text = separate_camel_case(text)
        if keep_punc:
            res = split_with_punc(text)
        else:
            res = text.split()
        if not keep_stopwords:
            res = [ele for ele in res if ele not in self.stopwords]
        
        ## Each tokenized words are stemmed
        return [self.stemmer.stem(ele) if use_stem else ele for ele in res]

    ## Methods
    def select_rule(self, query: str) -> str:
        """
        Method 3: Randomly select a rule from all the pre-defined rules and apply on the string
        """
        ## Rule not applied on numericals
        if query.isdigit():
            return query 

        rule_choices = [(self.pr_keep_k, self.rule_keep_k), 
                        (self.pr_remove_vowels, self.rule_remove_vowels), 
                        (self.pr_logic, self.rule_logic)]
        ## Probabilities of choosing each of the rule when the method seleted is rule-based.
        rule_probs = [choice[0] for choice in rule_choices]
        selected_rule_idx = self.select_from_probs(rule_probs)
        selected_rule = rule_choices[selected_rule_idx][-1]
        if len(query) > 10:
            orig_thres = self.per_tok_target_len
            self.per_tok_target_len = len(query) // 2
            res = selected_rule(query)
            self.per_tok_target_len = orig_thres
        else:
            res = selected_rule(query)
        return res

    def as_is(self, query: str) -> str:
        """
        Method 1: Keep the word as is.s
        """
        return query
    
    def lookup(self, query: str) -> str: 
        """
        Method 2: Find corresponding abbreviation from a lookup table
        """
        ## TODO if returns multiple values, current solution is to randomly pick one, but need to later figure out a soln to cache one value for future use in the same table, or some similar tables
        if query in self.lookup_abbreviation:
            values_raw = self.lookup_abbreviation[query]
            if values_raw is not None:
                weights = [ele["upvotes"] for ele in values_raw.values()]
                if sum(weights) > 0:
                    abbrev = random.choices(list(values_raw.keys()),
                        weights=weights, k=1)[0]
                    return abbrev
                
        return self.select_rule(query)

    def select_method(self, query: str) -> str:
        """
        Select one of the token-level processing method
        """
        method_choices = [(self.pm_as_is, self.as_is), 
                          (self.pm_lookup, self.lookup), 
                          (self.pm_selected_rule, self.select_rule)]
        method_probs = [choice[0] for choice in method_choices]
        selected_method_idx = self.select_from_probs(method_probs)
        selected_method = method_choices[selected_method_idx][-1]
        return selected_method(query)

    def combine(self, toks: list, p_camel=.333, p_underscore=.333) -> str:
        """
        Combine the abbreviated tokens into the cryptic name by either camelCase or underscore_name
        """
        def preprocess(toks: list) -> list:
            new_toks = []
            for tok in toks:
                if isinstance(tok, list):
                    new_toks.extend(tok)
                else:
                    new_toks.append(tok)
            return new_toks

        def combine_underscore(toks: list) -> str:
            res = ""
            for i, tok in enumerate(toks):
                if tok.isalnum() and i < len(toks) - 1:
                    res += tok
                    if toks[i+1].isalnum():
                        res += "_"
                else:
                    res += tok
            return res
        
        def combine_camel(toks: list) -> str:
            if len(toks) > 1:
                camel_case = "".join([toks[0]] + [tok[0].upper() + tok[1:] if len(tok) > 1 else tok.upper() for tok in toks[1:]])
                return camel_case
            else:
                return "".join(toks)

        def combine_simple(toks: list) -> str: 
            return "".join(toks)

        toks = preprocess(toks)
        rand = random.uniform(0, 1)
        if 0 < rand < p_camel:
            return combine_camel(toks)
        elif p_camel <= rand < p_camel + p_underscore:
            return combine_underscore(toks)
        else:
            return combine_simple(toks)
    
    def span2plus(self, lst):
        res = []
        for i in range(2, len(lst) + 1):
            for t in range(len(lst) - i + 1):
                res.append((lst[t:t+i], t, t+i))
        return res

    def filter_acronyms(self, words, lookup):
    
        combs = self.span2plus(words)
        for comb, l_end, r_start in combs:
            comb_string = " ".join(comb)
            if comb_string in lookup:
                acronym_cands = lookup[comb_string]
                weights = [ele["upvotes"] for ele in acronym_cands.values()]
                if sum(weights) > 0:
                    acronym = random.choices(list(acronym_cands.keys()), weights=weights, k=1)[0]
                    left, right = words[:l_end], words[r_start:]
                    return [acronym], l_end, r_start
        return [], -1, -1

    def generate(self, text: str) -> str:
        """
        Generate cryptic name from column header
        """
        toks = self.tokenize(text)
        if len(toks) < 10:
            # The time complexity for acronym matching is O(N(N-1)/2) ~ O(N^2), where N is the number of tokens in the column header.
            # It is possible to encounter very long headers like a small paragraph and we should avoid matching acronyms for very long headers.
            # Threshold set to 10 tokens.
            acronym, acronym_start_idx, acronym_end_idx = self.filter_acronyms([tok.lower() for tok in toks], self.lookup_acronym)
            rand = random.uniform(0, 1)

            ## Case where there exist matching span(s) from the acronym lookup dictionary and generator selected to replace acronyms
            if acronym_start_idx >= 0 and rand < self.p_filter_acronym:
                left = [self.select_method(tok) for tok in toks[:acronym_start_idx]]
                right = [self.select_method(tok) for tok in toks[acronym_end_idx:]]
                return left + acronym + right

        return [self.select_method(tok) for tok in toks]

import json
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, required=True)
    parser.add_argument('--seed', type=int, default=22)
    parser.add_argument('-k', '--per_tok_target_len', type=int, default=5)
    parser.add_argument('--config_path', type=str, default='./src/cryptifier_config.json')
    args = parser.parse_args()

    with open(args.config_path, 'r') as fi:
        params = json.load(fi)

    lookup_abbreviation_path = './lookups/abbreviation_samples.json'
    lookup_acronym_path = './lookups/acronym_samples.json'

    ## Lookup tables
    lookup_abbreviation = json.load(open(lookup_abbreviation_path, "r"))
    lookup_acronym = json.load(open(lookup_acronym_path, "r"))

    generator = CrypticNameGenerator(lookup_abbreviation=lookup_abbreviation,
                                    lookup_acronym=lookup_acronym, seed=args.seed, **params)
                                     
    print(generator.combine(generator.generate(args.text)))
