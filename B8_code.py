import numpy as np
import pandas as pd 
import re
import math
from collections import Counter, defaultdict, OrderedDict, deque
from pprint import pprint

from nltk.stem import PorterStemmer
import random
from sklearn.metrics import classification_report, confusion_matrix

from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
from scipy.sparse import dok_matrix

from operator import neg
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB, CategoricalNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDClassifier

import csv

import scipy
import string
import sklearn
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import ttest_ind
from sympy import bernoulli



stopwords = "ttds_stopwords.txt"
sentiment_train_path = "train-sentiment.txt"
sentiment_test_path = "ttds_2023_cw2_test_final.txt"


system_results = "ttdssystemresults.csv"
query_eval = "qrels.csv"


# Part 1

DECIMAL_PLACES = 3

class Eval:
    def __init__(self, system_results, true_results):
        self.system_number = 3
        self.system_results = self.parse_results(system_results)
        self.true_results = self.parse_true_results(true_results)
    
    def set_system_number(self, system_number):
        self.system_number = system_number
    
    def parse_true_results(self, true_results):
        result_mapping = defaultdict()
        with open(true_results, "r") as f:
            for line in f.readlines()[1:]:
                line_items = line.strip().split(',')
                query_id = int(line_items[0])
                if query_id not in result_mapping:
                    result_mapping[query_id] = []
                result_mapping[query_id].append(tuple(map(int, line_items[1:])))
    
        # Sorting each list by the last element of the tuples
        for query_id in result_mapping:
            result_mapping[query_id].sort(key=lambda x: x[-1])
        return result_mapping
    
    def parse_results(self, result_document):
        result_mapping = defaultdict(lambda: defaultdict(list))
        with open(result_document, "r") as f:
            for line in f.readlines()[1:]:
                line_items = line.strip().split(',')
                line_items = list(map(int, line_items[:-1])) + [float(line_items[-1])]
                tuple_items = tuple(line_items)

                query_key = tuple_items[1]
                system_key = tuple_items[0]
                value_tuple = tuple_items[2:5]

                result_mapping[query_key][system_key].append(value_tuple)

        # Sort each inner list by the last element (x[4])
        for query_key, system_dict in result_mapping.items():
            for query_key in system_dict:
                system_dict[query_key].sort(key=lambda x: x[-1], reverse=True)

        return result_mapping
    
    def p_at_N(self, query_number, N=10):
        query_docs = self.system_results[query_number][self.system_number][:N]
        query_doc_nums = set([doc_num for (doc_num, _, _) in query_docs])
        #pprint(query_docs)
                
        true_docs = self.true_results[query_number][:N]
        true_doc_nums = set([doc_num for (doc_num, _) in true_docs])
        #pprint(true_docs)
        
        total_results = true_doc_nums.intersection(query_doc_nums)
        #pprint(total_results)
        
        #TODO might need to be N
        result = round(len(total_results) / len(query_doc_nums), DECIMAL_PLACES)
        return result
    
    def r_at_N(self, query_number, N=50):
        query_docs = self.system_results[query_number][self.system_number][:N]
        query_doc_nums = set([doc_num for (doc_num, _, _) in query_docs])
                
        true_docs = self.true_results[query_number][:N]
        true_doc_nums = set([doc_num for (doc_num, _) in true_docs])
        
        total_results = true_doc_nums.intersection(query_doc_nums)
        
        result = round(len(total_results) / len(true_docs), DECIMAL_PLACES)
        return result
    
    def r_precision(self, query_number):
        rank = len(self.true_results[query_number])
        return round(self.p_at_N(query_number, rank), DECIMAL_PLACES)
    
    def a_precision(self, query_number):
        true_docs = self.true_results[query_number]
        true_doc_nums = set([doc_num for (doc_num, _) in true_docs])
        
        query_docs = self.system_results[query_number][self.system_number]
        query_doc_nums = set([doc_num for (doc_num, _, _) in query_docs])
        
        precision_values = []
        seen = 1
        relevant = 0
        
        for doc in query_doc_nums:
            if doc in true_doc_nums:
                relevant += 1
                precision_values.append(relevant / seen)
            seen += 1
            if relevant == len(true_doc_nums):
                break
        
        # TODO rounding issues here. 
        return round(sum(precision_values) / len(true_doc_nums), DECIMAL_PLACES)
        
    def dcg_at_K(self, query_number, K=10):
        query_docs = self.system_results[query_number][self.system_number][:K]
        query_doc_nums = [doc_num for (doc_num, _, _) in query_docs]
        
        true_docs = self.true_results[query_number]
        true_doc_mapping = dict(true_docs)
        
        doc_scores = []
        
        idx = 1
        for doc in query_doc_nums:
            if doc in true_doc_mapping:
                doc_scores.append((idx, true_doc_mapping[doc]))
                idx += 1
                continue
            idx += 1
            

        if doc_scores and doc_scores[0][0] == 1:
            return doc_scores[0][1] + sum([(score / math.log2(idx)) for idx, score in doc_scores[1:]])
        else:
            return sum([(score / math.log2(idx)) for idx, score in doc_scores])
        
    def idcg_at_K(self, query_number, K=10):
        true_docs = self.true_results[query_number][:K]
        if not true_docs:
            return 0
        result_cache = true_docs[0][1]
        
        if len(true_docs) == 1:
            return result_cache

        for i in range(1, len(true_docs)):
            score = true_docs[i][1]
            result_cache += (score / math.log2(i+1))
            
        return result_cache 
    
    def ndcg_at_K(self, query_number, K=10):
        return round(self.dcg_at_K(query_number, K=K) / self.idcg_at_K(query_number, K=K), DECIMAL_PLACES)
    
    def evaluate_system(self, filepath):
        ir_eval_df = pd.read_csv(filepath)

        # filter out row with first value = 'mean'
        ir_eval_df = ir_eval_df[ir_eval_df['query_number'] != 'mean']

        # Initialize a dictionary for results
        results = {}
        
        # Calculate mean scores for each system
        mean_scores = ir_eval_df.groupby('system_number').mean()

        # Iterate through each metric to find the best and second best systems
        for metric in ['P@10', 'R@50', 'r-precision', 'AP', 'nDCG@10', 'nDCG@20']:
            sorted_scores = mean_scores.sort_values(by=metric, ascending=False)
            best_system = sorted_scores.index[0]
            second_best_system = sorted_scores.index[1]

            # Perform t-test
            t_stat, p_value = ttest_ind(
                ir_eval_df[ir_eval_df['system_number'] == best_system][metric],
                ir_eval_df[ir_eval_df['system_number'] == second_best_system][metric]
            )

            # Store results
            results[metric] = {
                'Best System': best_system,
                'Second Best System': second_best_system,
                'T-Statistic': t_stat,
                'P-Value': p_value,
                'Significantly Better': p_value < 0.05
            }

        return results
    
    
def run_eval(system_results, query_eval):
    eval = Eval(system_results, query_eval)
    
    output_file = "test_ir_eval.csv"
    
    results = []
    mean_scores = {}

    # Collect scores for each system and query
    for sysnumber in range(1, 7):
        sys_scores = {metric: [] for metric in ['P@10', 'R@50', 'r-precision', 'AP', 'nDCG@10', 'nDCG@20']}
        for querynumber in range(1, 11):
            eval.set_system_number(sysnumber)
            pAt10 = eval.p_at_N(querynumber, N=10)
            rAt50 = eval.r_at_N(querynumber, N=50)
            rPrecision = eval.r_precision(querynumber)
            aPrecision = eval.a_precision(querynumber)
            ndcgAt10 = eval.ndcg_at_K(querynumber, K=10)
            ndcgAt20 = eval.ndcg_at_K(querynumber, K=20)
            results.append((sysnumber, querynumber, pAt10, rAt50, rPrecision, aPrecision, ndcgAt10, ndcgAt20))
            
            # Collecting scores for mean calculation
            sys_scores['P@10'].append(pAt10)
            sys_scores['R@50'].append(rAt50)
            sys_scores['r-precision'].append(rPrecision)
            sys_scores['AP'].append(aPrecision)
            sys_scores['nDCG@10'].append(ndcgAt10)
            sys_scores['nDCG@20'].append(ndcgAt20)

        # Calculate mean scores for each system
        mean_scores[sysnumber] = {metric: round(sum(scores) / len(scores), DECIMAL_PLACES) for metric, scores in sys_scores.items()}
        
        results.append((sysnumber, 'mean', mean_scores[sysnumber]['P@10'], mean_scores[sysnumber]['R@50'], 
                        mean_scores[sysnumber]['r-precision'], mean_scores[sysnumber]['AP'], 
                        mean_scores[sysnumber]['nDCG@10'], mean_scores[sysnumber]['nDCG@20']))

    # Write results to CSV file
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['system_number', 'query_number', 'P@10', 'R@50', 'r-precision', 'AP', 'nDCG@10', 'nDCG@20'])
        for row in results:
            row = [f"{item:.3f}" if i in range(2,7) else item for i, item in enumerate(row)]
            writer.writerow(row)



#run_eval(system_results, query_eval)



# Part 2


bible_quran = "train_and_dev.tsv"
bible_quran = pd.read_csv(bible_quran, sep="\t", header=None, index_col=False, names=['doc', 'text'])


with open(stopwords, 'r') as f:
    stopwords = set(map(str.strip, f.readlines()))


quran_text = bible_quran[bible_quran['doc'] == 'Quran']['text'].values.tolist()
nt_text = bible_quran[bible_quran['doc'] == 'NT']['text'].values.tolist()
ot_text = bible_quran[bible_quran['doc'] == 'OT']['text'].values.tolist()


class TokeniserP2:
   def __init__(self, preserved_patterns=[], lowercase=True, exclude_numbers=False, min_token_len=1, drop_patterns=[]):
       # We can specify a list of regex patterns to not split on.
       self.preserved_patterns = [re.compile(p) for p in preserved_patterns]
       self.lowercase = lowercase
       self.exclude_numbers = exclude_numbers
       self.min_token_len = min_token_len
       #self.splitter = re.compile(r'[^\w]+', re.ASCII) 
       self.splitter = re.compile(r'[^A-Za-z]+')
       self.drop_patterns = [re.compile(p) for p in drop_patterns]
        
       self.token_cache = {} # Cache for memoization

   def tokenise(self, text):
        
       # We have not used enumerate because of the sub-token splitting below.
       for token in text.split():
           if token in self.token_cache:
               yield self.token_cache[token]
            
           # Check if the token matches any preserved pattern
           if any(pattern.fullmatch(token) for pattern in self.preserved_patterns):
               yield token
           elif any(pattern.fullmatch(token) for pattern in self.drop_patterns):
               continue
           else:
               for sub_token in self.splitter.split(token):

                   # Apply transformations and checks based on the provided controls
                   sub_token = sub_token.lower() if self.lowercase else sub_token
                    
                   # Check if sub_token matches conditions given to init function above.
                   if (not sub_token.isdigit() or not self.exclude_numbers) and len(sub_token) >= self.min_token_len:
                       yield sub_token




tokeniser = TokeniserP2()
stemmer = PorterStemmer()

def process_text(text, enable_stem=True, enable_stopping=True, limit=None):
   if not limit:
       limit = len(text)
   results = []
   for line in text[:limit]:
       cache = []
       tokens = tokeniser.tokenise(line)
       for token in tokens:
           if enable_stopping and token in stopwords:
               continue
           if enable_stem:
               token = stemmer.stem(token)
           cache.append(token)
       results.append(cache)
   return results


def generate_counts(text):
   c = Counter()
   for line in text:
       c.update(line)
   return c

def calculate_counts(word, corpus_one, corpus_two, corpus_three):
   target_corpus = corpus_one
   target_corpus_length = len(target_corpus)
   other_corpus = corpus_two + corpus_three
   other_corpus_length = len(other_corpus)
   N = target_corpus_length + other_corpus_length
   N11 = 0
   for verse in target_corpus:
       if word in verse:
           N11 += 1
   N01 = target_corpus_length - N11
   N10 = 0
   for verse in other_corpus:
       if word in verse:
           N10 += 1
   N00 = other_corpus_length - N10
   return N, N11, N01, N10, N00

def compute_mi_chi(N, N11, N01, N10, N00):
   mi = 0
   N0x, N1x = N01 + N00, N10 + N11
   Nx0, Nx1 = N00 + N10, N01 + N11

   if (N1x * Nx1) > 0 and (N * N11) > 0:
       mi += (N11 / N) * math.log2((N * N11) / (N1x * Nx1))
   if N0x * Nx1 > 0 and (N * N01) > 0:
       mi += (N01 / N) * math.log2((N * N01) / (N0x * Nx1))
   if N1x * Nx0 > 0 and (N * N10) > 0:
       mi += (N10 / N) * math.log2((N * N10) / (N1x * Nx0))
   if N0x * Nx0 > 0 and (N * N00) > 0:
       mi += (N00 / N) * math.log2((N * N00) / (N0x * Nx0))

   expected_N11 = (N1x * Nx1) / N
   expected_N01 = (N0x * Nx1) / N
   expected_N10 = (N1x * Nx0) / N
   expected_N00 = (N0x * Nx0) / N
   chi_squared = ((N11 - expected_N11) ** 2 / expected_N11 if expected_N11 else 0) + \
                 ((N01 - expected_N01) ** 2 / expected_N01 if expected_N01 else 0) + \
                 ((N10 - expected_N10) ** 2 / expected_N10 if expected_N10 else 0) + \
                 ((N00 - expected_N00) ** 2 / expected_N00 if expected_N00 else 0)
   return mi, chi_squared



quran_text_processed = process_text(quran_text)
ot_text_processed = process_text(ot_text)
nt_text_processed = process_text(nt_text)


quran_counts = generate_counts(quran_text_processed)
ot_counts = generate_counts(ot_text_processed)
nt_counts = generate_counts(nt_text_processed)


common_vocab = set()
common_vocab.update(quran_counts.keys())
common_vocab.update(ot_counts.keys())
common_vocab.update(nt_counts.keys())


#quran first
quran_mi_chi = []
for word in common_vocab:
   N, N11, N01, N10, N00 = calculate_counts(word, quran_text_processed, ot_text_processed, nt_text_processed)
   mi, chi_squared = compute_mi_chi(N, N11, N01, N10, N00)
   quran_mi_chi.append((word, mi, chi_squared))
    


# ot first
ot_mi_chi = []
for word in common_vocab:
   N, N11, N01, N10, N00 = calculate_counts(word, ot_text_processed, quran_text_processed,  nt_text_processed)
   mi, chi_squared = compute_mi_chi(N, N11, N01, N10, N00)
   ot_mi_chi.append((word, mi, chi_squared))
    


# nt first
nt_mi_chi = []
for word in common_vocab:
   N, N11, N01, N10, N00 = calculate_counts(word, nt_text_processed, quran_text_processed, ot_text_processed)
   mi, chi_squared = compute_mi_chi(N, N11, N01, N10, N00)
   nt_mi_chi.append((word, mi, chi_squared))


quran_top_10_mi = sorted(quran_mi_chi, key=lambda x: x[1], reverse=True)[:10]
print("Quran top 10 MI")
pprint(quran_top_10_mi)

quran_top_10_chi = sorted(quran_mi_chi, key=lambda x: x[2], reverse=True)[:10]
print("Quran top 10 Chi")
pprint(quran_top_10_chi)

ot_top_10_mi = sorted(ot_mi_chi, key=lambda x: x[1], reverse=True)[:10]
print("OT top 10 MI")
pprint(ot_top_10_mi)

ot_top_10_chi = sorted(ot_mi_chi, key=lambda x: x[2], reverse=True)[:10]
print("OT top 10 Chi")
pprint(ot_top_10_chi)

nt_top_10_mi = sorted(nt_mi_chi, key=lambda x: x[1], reverse=True)[:10]
print("NT top 10 MI")
pprint(nt_top_10_mi)

nt_top_10_chi = sorted(nt_mi_chi, key=lambda x: x[2], reverse=True)[:10]
print("NT top 10 Chi")
pprint(nt_top_10_chi)




all_corpora = quran_text_processed + ot_text_processed + nt_text_processed
quran_range = len(quran_text_processed)
ot_range = quran_range + len(ot_text_processed)
nt_range = ot_range + len(nt_text_processed)

dictionary = Dictionary(all_corpora)
encoded_verses = [dictionary.doc2bow(verse) for verse in all_corpora]



lda = LdaModel(encoded_verses, id2word=dictionary, num_topics=20)


quran_topic_scores = defaultdict(float)
ot_topic_scores = defaultdict(float)
nt_topic_scores = defaultdict(float)

for idx, verse in enumerate(encoded_verses):
   topic_scores = lda.get_document_topics(verse)
   for (topic, score) in topic_scores:        
       if idx < quran_range:
           quran_topic_scores[topic] += score
       elif idx < ot_range:
           ot_topic_scores[topic] += score
       else:
           nt_topic_scores[topic] += score

quran_topic_scores = {k: v / len(quran_text_processed) for k, v in quran_topic_scores.items()}
ot_topic_scores = {k: v / len(ot_text_processed) for k, v in ot_topic_scores.items()}
nt_topic_scores = {k: v / len(nt_text_processed) for k, v in nt_topic_scores.items()}

def top_n_topics(topic_scores, n):
   return {k: v for k, v in sorted(topic_scores.items(), key=lambda item: item[1], reverse=True)[:n]}

n = 3
qtop_n_topics = top_n_topics(quran_topic_scores, n)
otop_n_topics = top_n_topics(ot_topic_scores, n)
ntop_n_topics = top_n_topics(nt_topic_scores, n)



print(qtop_n_topics)
print(otop_n_topics)
print(ntop_n_topics)


for topic in lda.print_topics(num_topics=20, num_words=10):
   print(topic)
    
    
    
    
    























    
def join_to_string(data_list):
    # Flatten the list of lists into a single list
    flattened_list = ["\t".join(map(str, item)) for item in data_list]
    
    # Join all items in the list into a single string
    joined_string = '\n'.join(map(str, flattened_list))
    return joined_string

    
sentiment_train = pd.read_csv(sentiment_train_path, sep='\t').values.tolist()
sentiment_test = pd.read_csv(sentiment_test_path, sep='\t').values.tolist()
sentiment_test_final = pd.read_csv("ttds_2023_cw2_test_final.txt", sep='\t').values.tolist()

random.shuffle(sentiment_train)
random.shuffle(sentiment_test)

train_ratio = 0.8
test_ratio = 0.1
dev_ratio = 0.1

train_section_end =  int(len(sentiment_train) * train_ratio)
test_section_end = train_section_end + int(len(sentiment_train) * test_ratio)
dev_section_start = test_section_end

train_set = sentiment_train[:train_section_end]
test_set = sentiment_train[train_section_end:test_section_end]
dev_set = sentiment_train[test_section_end:]


class Tokeniser:
    def __init__(self, preserved_patterns=[], drop_patterns=[], lowercase=False, exclude_numbers=False, min_token_len=1):
        # We can specify a list of regex patterns to not split on.
        self.preserved_patterns = preserved_patterns
        self.drop_patterns = drop_patterns
        self.lowercase = lowercase
        self.exclude_numbers = exclude_numbers
        self.min_token_len = min_token_len
        #self.splitter = re.compile(r'[^\w]+', re.ASCII) 
        self.splitter = re.compile(r'[^A-Za-z]+')
        
        self.token_cache = {} # Cache for memoization

    def tokenise(self, text):
        token_queue = deque()
        token_queue.extend(text.split())
        while token_queue:
            token = token_queue.popleft()
            if token in self.token_cache:
                yield self.token_cache[token]
            
            # Check if the token matches any preserved pattern
            if any(pattern.search(token) for pattern in self.drop_patterns):
                # only dropping one pattern for now
                other_tokens = self.drop_patterns[0].sub(' ', token)
                token_queue.extendleft(other_tokens.split())
                continue
            elif any(pattern.search(token) for pattern in self.preserved_patterns):
                token_cache = token
                match_tokens = []
                for idx, pattern in enumerate(self.preserved_patterns):
                    if pattern.search(token):
                        match_tokens.extend(pattern.findall(token))
                        token_cache = pattern.sub('', token_cache)
                token_queue.extendleft(token_cache.split())
                for match_token in match_tokens:
                    if type(match_token) == tuple:
                        # only occurs for contractions
                        match_token = "'".join(match_token)
                    yield match_token
            else:
                for sub_token in self.splitter.split(token):
                    # Apply transformations and checks based on the provided controls
                    sub_token = sub_token.lower() if self.lowercase else sub_token
                    
                    # Check if sub_token matches conditions given to init function above.
                    if (not sub_token.isdigit() or not self.exclude_numbers) and len(sub_token) >= self.min_token_len:
                        yield sub_token

EMOJI_REGEX = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+')
URL_REGEX = re.compile(r'https?://\S+')
VOWEL_REGEX = re.compile(r'[aeiouAEIOU]')
WORD_REGEX = re.compile(r'\w+')

# will match if we have a char and digit next to each other
DIGIT_REGEX = re.compile(r'\d+')
MIXED_DIGIT_REGEX = re.compile(r'^[\D\s]+[\d]+[\D\s]+$')
CONTRACTION_REGEX = re.compile(r"(\b\w+)['\"](\w+\b)")#r"'\w+")
ELONGATED_INFIX_REGEX = re.compile(r'(\w)(\1{2,})')
ELONGATED_SUFFIX_REGEX = re.compile(r'(\w)(\1{1,})$')
ELONGATED_PREFIX_REGEX = re.compile(r'^(\w)\1{1,}')

# matches if contains more than one uppercase letter
UPPERCASE_REGEX = re.compile(r'.*[A-Z]{2,}.*')
HASHTAG_REGEX = re.compile(r'#\w+')
MENTION_REGEX = re.compile(r'@\w+')

preserve_patterns = [HASHTAG_REGEX, MENTION_REGEX, EMOJI_REGEX, CONTRACTION_REGEX]
drop_patterns = [URL_REGEX]


class TokenNormaliser:
    def __init__(self, stemmer=None, stopwords=None, lowercase=False, min_token_len=1, stopping=False, stemming=False):
        self.stemmer = stemmer
        self.stopwords = stopwords
        self.cached_tokens = defaultdict(set)   
        self.lowercase = lowercase
        self.min_token_len = min_token_len
        self.stopping = stopping
        self.stemming = stemming
        
    def normalise(self, tokens, stopping=False, stemming=False, lowercase=False, min_token_len=1):
        cache = []
        mentions = []
        hashtags = []
        emojis = []
        uppercase = []
        for token in tokens:
            if token in self.cached_tokens:
                cache.extend(self.cached_tokens[token])
                continue
            if token.startswith('@'):
                mentions.append(token)
                continue
            if token.startswith('#'):
                hashtags.append(token)
                continue
            
            if EMOJI_REGEX.search(token):
                splits = set(filter(None, EMOJI_REGEX.split(token)))
                get_emojis = set(filter(None, EMOJI_REGEX.findall(token)))
                
                cache.extend(list(splits))
                emojis.extend(list(get_emojis))
                continue
            
            if re.search(UPPERCASE_REGEX, token):
                uppercase.append(token)
                cache.append(token)
                
            if re.search(CONTRACTION_REGEX, token):
                splits = set(filter(None, CONTRACTION_REGEX.split(token)))
                get_joined_strings = set(CONTRACTION_REGEX.sub(r'\1\2', token))
                all_tokens = splits.union(get_joined_strings)
                self.cached_tokens[token].update(all_tokens)
                
                cache.extend(list(all_tokens))
                continue
                
            if re.search(DIGIT_REGEX, token):
                splits = set(filter(None, DIGIT_REGEX.split(token)))
                get_digits = set(filter(None, DIGIT_REGEX.findall(token)))
                get_joined_strings = set(filter(None, DIGIT_REGEX.sub("",token).split()))
                all_tokens = splits.union(get_digits).union(get_joined_strings)
                self.cached_tokens[token].update(all_tokens)
                
                cache.extend(list(all_tokens))
                
            if re.search(ELONGATED_PREFIX_REGEX, token):
                words = WORD_REGEX.findall(token)
                replacements_single =  set(filter(None, map(lambda x: ELONGATED_PREFIX_REGEX.sub(r'\1\1', x), words)))
                replacements_multi =  set(filter(None, map(lambda x: ELONGATED_PREFIX_REGEX.sub(r'\1', x), words)))
                all_tokens = replacements_single.union(replacements_multi)
                self.cached_tokens[token].update(all_tokens)
                
                cache.extend(list(all_tokens))
                    
            if re.search(ELONGATED_INFIX_REGEX, token):
                words = WORD_REGEX.findall(token)
                replacements_single =  set(filter(None, map(lambda x: ELONGATED_INFIX_REGEX.sub(r'\1\1', x), words)))
                replacements_multi =  set(filter(None, map(lambda x: ELONGATED_INFIX_REGEX.sub(r'\1', x), words)))
                all_tokens = replacements_single.union(replacements_multi)
                self.cached_tokens[token].update(all_tokens)
                
                cache.extend(list(all_tokens))
                
            if re.search(ELONGATED_SUFFIX_REGEX, token):
                words = WORD_REGEX.findall(token)
                replacements_single =  set(filter(None, map(lambda x: ELONGATED_SUFFIX_REGEX.sub(r'\1\1', x), words)))
                replacements_multi =  set(filter(None, map(lambda x: ELONGATED_SUFFIX_REGEX.sub(r'\1', x), words)))
                all_tokens = replacements_single.union(replacements_multi)
                self.cached_tokens[token].update(all_tokens)
                
                cache.extend(list(all_tokens))
                
            cache.append(token)

        tweet_tokens = []
        for token in cache:
            token = token.lower() if self.lowercase else token
            if self.stemming and self.stemmer:
                token = self.stemmer.stem(token)
            if len(token) <= min_token_len:
                continue
            if stopping:
                if token not in self.stopwords:
                    tweet_tokens.append(token)
            else:
                tweet_tokens.append(token)

        tweet_data = defaultdict(list)
        tweet_data['tokens'] = map(str.lower, tweet_tokens) if lowercase else tweet_tokens
        tweet_data['mentions'] = map(str.lower, mentions) if lowercase else mentions
        tweet_data['hashtags'] = map(str.lower, hashtags) if lowercase else hashtags
        tweet_data['emojis'] = emojis
        tweet_data['uppercase'] = uppercase
        
        return tweet_data
    
    def replace_suffix(self, match):
        char = match.group(1)
        return char * 2 if char.lower() in 'aeiou' else char
        
    def __repr__(self):
        return (f"<Tokens: {self.cached_tokens}>\n")
    
    
        

tokeniser = Tokeniser(preserved_patterns=preserve_patterns, drop_patterns=drop_patterns, lowercase=True)
stemmer = PorterStemmer()


class WordIndexNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.index_id = 0
        self.word_frequency = 0
        self.word_probability = 0.0
        
    def __repr__(self):
        return f"<{self.children}>"



class WordIndex:
    def __init__(self):
        self.root = WordIndexNode()
        self.index_size = 0
        self.total_word_frequency = 0

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = WordIndexNode()
            node = node.children[char]
            
        node.is_end_of_word = True
        node.index_id = self.index_size
        
        node.word_frequency += 1
        self.total_word_frequency += 1
        self.index_size += 1
        
        node.word_probability = node.word_frequency / self.total_word_frequency
        
    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return None
            node = node.children[char]
        return (word, node.index_id, node.word_frequency, node.word_probability) if node.is_end_of_word else None
    
    def prefix_search(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
            
        words_with_prefix = []
        for word, _, node in self.traverse(node, prefix):
            if node.is_end_of_word:
                # Update probability for the current word
                node.word_probability = node.word_frequency / self.total_word_frequency
                words_with_prefix.append((word, node.index_id, node.word_frequency, node.word_probability))
            
        return words_with_prefix
    
    def traverse(self, node, prefix=""):
        stack = [(node, prefix)]
        while stack:
            current_node, current_word = stack.pop()
            for char, child_node in current_node.children.items():
                stack.append((child_node, current_word + char))
            yield current_word, current_node.index_id, current_node
    
    def match_word_lazy(self, long_word, disable_overlap=True, limit=10, min_length=1):
        def lazy_search(start_index):
            node = self.root
            current_word = ""
            scan_index = start_index
            words_found = defaultdict(list)
            
            while scan_index < len(long_word):
                char = long_word[scan_index]
                if char in node.children:
                    node = node.children[char]
                    current_word += char
                    if node.is_end_of_word:
                        word_data = tuple((current_word, scan_index - len(current_word) + 1, scan_index + 1, node.word_probability))
                        words_found[scan_index].append(word_data)
                    scan_index += 1
                else:
                    node = self.root
                    current_word = ""
                    if char not in node.children:
                        scan_index += 1
            return words_found

        start_index = 0
        word_objects = OrderedDict()
        word_prob_mapping = {}
        words_found = lazy_search(start_index)
        keys = sorted(words_found.keys())
        
        for key in keys:
            word_objects[key] = words_found[key]
            
        def recurse(index, current_sequence, all_sequences):
            if index >= len(keys):
                all_sequences.append([pack[0] for pack in current_sequence])
                return
            
            key_value = keys[index]
            for pack in word_objects[key_value]:
                word, start, end, prob = pack
                word_prob_mapping[word] = prob
                if not current_sequence or current_sequence[-1][2] <= start:
                    recurse(index + 1, current_sequence + [pack], all_sequences)
            
            recurse(index + 1, current_sequence, all_sequences)
            
        all_sequences = []
        recurse(0, [], all_sequences)

        token_counter = Counter()
        for sequence in all_sequences:
            token_counter.update([word for word in sequence])
            
        # sorted by length
        return sorted(token_counter.keys(), key=len, reverse=True)[:limit]
    
    
    def match_word_greedy(self, long_word, allow_overlap=False):
        def find_word_greedy(start_index):
            node = self.root
            current_word = ""
            longest_word = ""
            probability = 0

            for i in range(start_index, len(long_word)):
                char = long_word[i]
                if char in node.children:
                    node = node.children[char]
                    current_word += char
                    if node.is_end_of_word:
                        longest_word = current_word  # Update longest word
                        probability = node.word_probability
                else:
                    break   

            return longest_word, probability

        index = 0
        words_with_probabilities = OrderedDict()
        non_overlapping_words = set()
        
        while index < len(long_word):
            longest_word, probability = find_word_greedy(index)
            if longest_word:
                words_with_probabilities[longest_word] = probability
                if not allow_overlap:
                    non_overlapping_words.add(longest_word)
                    index += len(longest_word)  # Move to the next character if overlaps are not allowed
                else:
                    index += 1  # Move to the next character if overlaps are allowed
            else:
                index += 1  # Move to the next character if no word is found
        if not allow_overlap:
            return list(non_overlapping_words)
        return words_with_probabilities.keys()
    
    def __str__(self, limit=10):
        word_list = [] 

        for word, _, node in self.traverse(self.root):
            if node.is_end_of_word and len(word_list) < limit:
                # Update probability for the current word
                node.word_probability = node.word_frequency / self.total_word_frequency
                word_list.append((word, node.index_id, node.word_frequency, node.word_probability))

        return str(word_list) 
    
    def __len__(self):
        return self.index_size
    
    def __contains__(self, word):
        return self.search(word) is not None
    
    def __getitem__(self, word):
        return self.search(word)

    def __repr__(self):
        return str(self)





class DataLoader:
    def __init__(self):
        self.data = []
        self.stopwords = set()
        self.train_data = []
        self.dev_data = []
        self.test_data = []
        self.size = 0

    def set_stopwords(self, stopwords_file):
        with open(stopwords_file, 'r') as f:
            self.stopwords = set(map(str.strip, f.readlines()))

    def load_data(self, file_path):
        with open(file_path, 'r') as f:
            self.data = [line.strip().split('\t') for line in f.readlines()][1:]
            self.size = len(self.data)

    def shuffle_data(self):
        random.shuffle(self.data)

    def split_data(self, train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1):
        total_len = len(self.data)
        train_end = int(total_len * train_ratio)
        dev_end = int(total_len * (train_ratio + dev_ratio))

        self.train_data = self.data[:train_end]
        self.dev_data = self.data[train_end:dev_end]
        self.test_data = self.data[dev_end:]

    def get_sample(self, data_type, sample_size=5):
        data = getattr(self, f"{data_type}_data", "data")
        if data:
            id, labels, texts = zip(*data)
            return random.sample(list(zip(labels, texts)), sample_size)
        else:
            return []

    def get_labels_and_data(self, data_type):
        data = getattr(self, f"{data_type}_data", "data")
        if data:
            id, labels, texts = zip(*data)
            return labels, texts
        else:
            return [], []




class TweetProcessor:
    def __init__(self):
        self.stopping = False
        self.stemming = False
        self.lowercase = True
        self.calulate_mi_chi = False
        self.min_token_len = 1
        self.token_weght = 1.0
        self.add_hashtags = True
        self.hashtag_threshold = 3
        self.bow_vocabulary_size = 3000
        
        self.index = WordIndex()
        self.tokeniser = None
        self.token_normaliser = None
        self.miChiIndex = None
        self.vocabulary = set()
        self.bow_vocabulary = set()
        self.cat2id = { 'neutral': 0, 'positive': 1 , 'negative': 2 }
        self.word2index = {}
        self.index2word = {}
        self.bowId2index = {}
        self.index2bowId = {}
        self.tweet2tokens = defaultdict(list)
        self.token_normalisation = {}
        self.oov_index = len(self.vocabulary)
        
        self.word_counter = Counter()
        self.pos_counter = Counter()
        self.neg_counter = Counter()
        self.neu_counter = Counter()

        self.all_tweets = []
        self.pos_tweets = []
        self.neg_tweets = []
        self.neu_tweets = []
        
        self.process_queue = deque()
        self.tweet_object_cache = {}
        
        self.all_counts = defaultdict(lambda: (0,0,0,0))

    
    def setTokeniser(self, tokeniser):
        self.tokeniser = tokeniser
        
    def setTokenNormaliser(self, token_normaliser):
        self.token_normaliser = token_normaliser
        
    def setTweetMiChiIndex(self, miChiIndex):
        self.miChiIndex = miChiIndex
        self.updateBowVocabulary()
        
    def updateBowVocabularySize(self, bow_vocabulary_size):
        self.bow_vocabulary_size = bow_vocabulary_size
        self.updateBowVocabulary()
        
    def updateBowVocabulary(self):
        # other things here. 
        #self.bow_vocabulary = set(sorted(self.miChiIndex.mi_chi_mapping, key=lambda x: self.miChiIndex.mi_chi_mapping[x][0][1], reverse=True)[:self.bow_vocabulary_size + 1])
        self.bow_vocabulary = set(set([x[0] for x in tweetProcessor.word_counter.most_common(self.bow_vocabulary_size)]))
        self.bow_vocabulary.add("OOV")
        self.oov_index = len(self.bow_vocabulary)
        for idx, word in enumerate(self.bow_vocabulary):
            self.bowId2index[idx] = word
            self.index2bowId[word] = idx
        
    def updateParams(self, stopping=False, stemming=False, lowercase=False, min_token_len=1, token_weight=1.0, calulate_mi_chi=False, bow_vocabulary_size=3000):
        self.stopping = stopping
        self.stemming = stemming
        self.lowercase = lowercase
        self.min_token_len = min_token_len
        self.token_weght = token_weight
        self.calulate_mi_chi = calulate_mi_chi
        self.bow_vocabulary_size = bow_vocabulary_size
    
    def index_tweets(self, dataloader_texts, dataloader_labels):
        for idx, tweet in enumerate(dataloader_texts):
            # apply any other normalsiation here
            token_cache = []
            tweetObject = self.token_normaliser.normalise(self.tokeniser.tokenise(tweet))
            if tweetObject['hashtags']:
                self.tweet_object_cache[idx] = tweetObject
                self.process_queue.append(idx)
            if tweetObject['emoji']:
                emoji_set = set(tweetObject['emoji'])
                token_cache.extend(list(emoji_set))
                self.vocabulary.update(emoji_set)
            if tweetObject['mentions']:
                token_cache.append("<MENTION>")
                token_cache.extend(map(str.lower, tweetObject['mentions']))
            
            for token in tweetObject['tokens']:
                token_cache.append(token)
                self.word2index[token] = self.index.index_size
                self.index2word[self.index.index_size] = token
                self.index.insert(token)
                self.vocabulary.add(token)
                
            self.tweet2tokens[idx] = token_cache
            
        if self.add_hashtags:
            while self.process_queue:
                idx = self.process_queue.popleft()
                tweetObject = self.tweet_object_cache[idx]
                for hashtag in tweetObject['hashtags']:
                    hashtag = hashtag.lower() if self.lowercase else hashtag
                    greedy_match = set(self.index.match_word_greedy(hashtag[1:]))
                    lazy_match = set(self.index.match_word_lazy(hashtag[1:]))
                    hashtag_tokens = [word for word in greedy_match.union(lazy_match) if len(word) >= self.hashtag_threshold]
                    self.token_normalisation[hashtag] = hashtag_tokens
                    self.tweet2tokens[idx].extend(hashtag_tokens)
        
        for idx in self.tweet2tokens:
            self.word_counter.update(self.tweet2tokens[idx])
            if dataloader_labels[idx] == "neutral":
                self.neu_tweets.append(self.tweet2tokens[idx])
            elif dataloader_labels[idx] == "positive":
                self.pos_tweets.append(self.tweet2tokens[idx])
            elif dataloader_labels[idx] == "negative":
                self.neg_tweets.append(self.tweet2tokens[idx])
        # update mi chi index
        for word in self.word_counter:
            self.all_counts[word] = (self.word_counter[word], self.neu_counter[word], self.pos_counter[word], self.neg_counter[word])

    def convert_to_bow(self, texts, weight=1):
        #if not self.bow_vocabulary or not self.miChiIndex:
        #    print("Mi Chi index not set")
        #    return
        
        matrix_size = (len(texts), len(self.bow_vocabulary) + 1)
        oov_index = self.bow_vocabulary_size
        bow = dok_matrix(matrix_size, dtype=np.float32)
        
        for i, tweet in enumerate(texts):
            token_cache = []
            tweetObject = self.token_normaliser.normalise(self.tokeniser.tokenise(tweet))
            # apply any other normalsiation here
            if tweetObject['hashtags']:
                for hashtag in tweetObject['hashtags']:
                    hashtag = hashtag.lower() if self.lowercase else hashtag
                    hashtag_tokens = self.token_normalisation.get(hashtag[1:], [])
                    token_cache.extend(hashtag_tokens)
            if tweetObject['emoji']:
                emoji_set = set(tweetObject['emoji'])
                token_cache.extend(list(emoji_set))
            if tweetObject['mentions']:
                token_cache.append("<MENTION>")
            token_cache.extend(tweetObject['tokens'])
            for token in token_cache:
                bow[i, self.index2bowId.get(token, oov_index)] += self.get_feature_weight(token, weight)
                # maybe consider adding some scaling facter here.
        return bow
                
    def get_features(self, texts):
        features = []
        for tweet in texts:
            tweetObject = self.token_normaliser.normalise(self.tokeniser.tokenise(tweet))
            feature = []
            for token in tweetObject['tokens']:
                if token not in self.bow_vocabulary:
                    feature.append(self.oov_index)
                    continue
                feature.append(self.word2index[token])
            features.append(feature)
        return features
    
    def get_labels(self, labels):
        return [self.cat2id[label] for label in labels]
    
    def get_feature_weight(self, word, alpha=1):
        if word not in self.bow_vocabulary or not self.miChiIndex:
            return 1
        # sum of square of differences of mi_chi scores across classes
        neu_chi, _ = self.miChiIndex.mi_chi_mapping[word][1]
        pos_chi, _ = self.miChiIndex.mi_chi_mapping[word][2]
        neg_chi, _ = self.miChiIndex.mi_chi_mapping[word][3]
        res = (neu_chi - pos_chi) ** 2 + (neu_chi - neg_chi) ** 2 + (pos_chi - neg_chi) ** 2
        inv_log_normalized_res = math.exp(res) / (1 + alpha)
        return inv_log_normalized_res
    
    
    def compute_accuracy(self, model, x_test, y_test):
        y_pred = model.predict(x_test)
        acc = np.mean(y_pred == y_test)
        print(f"Accuracy: {acc}")
        
    def __repr__(self) -> str:
        attributes = f"stopping={self.stopping}, stemming={self.stemming}, lowercase={self.lowercase}, min_token_len={self.min_token_len}, token_weight={self.token_weght}, calulate_mi_chi={self.calulate_mi_chi}"
        stats = f"Total Tweets: {len(self.all_tweets)}, Positive Tweets: {len(self.pos_tweets)}, Negative Tweets: {len(self.neg_tweets)}, Neutral Tweets: {len(self.neu_tweets)}"
        index_stats = f"Vocabulary Size: {len(self.vocabulary)}, BOW Vocabulary Size: {len(self.bow_vocabulary)}, Index Size: {len(self.index)}"
        return f"<TweetProcessor({attributes})>\n{stats}\n{index_stats}"
        


class TweetMiChiIndex:
    def __init__(self, vocabulary, pos_tweets, neg_tweets, neu_tweets):
        self.pos_tweets = pos_tweets
        self.neg_tweets = neg_tweets
        self.neu_tweets = neu_tweets
        self.vocabulary = vocabulary
        
        self.mi_chi_mapping = defaultdict(lambda: ((0,0), (0,0), (0,0), (0,0)))
        self.pos_mi_chi = []
        self.neg_mi_chi = []
        self.neu_mi_chi = []
        
        self.build()
    
    def build(self):
        count = 0
        for word in self.vocabulary:
            if count % 1000 == 0:
                print(f"Processed {count} words")
            count += 1
            N, N11, N01, N10, N00 = self.calculate_counts(word, self.pos_tweets, self.neg_tweets, self.neu_tweets)
            mi, chi_squared = self.compute_mi_chi(N, N11, N01, N10, N00)
            self.pos_mi_chi.append((word, mi, chi_squared))
            pos_mi, pos_chi = (mi, chi_squared)
            
            N, N11, N01, N10, N00 = self.calculate_counts(word, self.neg_tweets, self.pos_tweets, self.neu_tweets)
            mi, chi_squared = self.compute_mi_chi(N, N11, N01, N10, N00)
            self.neg_mi_chi.append((word, mi, chi_squared))
            neg_mi, neg_chi = (mi, chi_squared)
            
            N, N11, N01, N10, N00 = self.calculate_counts(word, self.neu_tweets, self.neg_tweets, self.pos_tweets)
            mi, chi_squared = self.compute_mi_chi(N, N11, N01, N10, N00)
            self.neu_mi_chi.append((word, mi, chi_squared))
            neu_mi, neu_chi = (mi, chi_squared)
            
            word_total_mi = pos_mi + neg_mi + neu_mi
            word_total_chi = pos_chi + neg_chi + neu_chi
            self.mi_chi_mapping[word] = ((word_total_mi, word_total_chi), (neu_mi, neu_chi), (pos_mi, pos_chi), (neg_mi, neg_chi))
        
    def calculate_counts(self, word, corpus_one, corpus_two, corpus_three):
        target_corpus = corpus_one
        target_corpus_length = len(target_corpus)
        other_corpus = corpus_two + corpus_three
        other_corpus_length = len(other_corpus)
        N = target_corpus_length + other_corpus_length
        N11 = 0
        for verse in target_corpus:
            if word in verse:
                N11 += 1
        N01 = target_corpus_length - N11
        N10 = 0
        for verse in other_corpus:
            if word in verse:
                N10 += 1
        N00 = other_corpus_length - N10
        return N, N11, N01, N10, N00
    
    
    def compute_mi_chi(self, N, N11, N01, N10, N00):
        mi = 0
        N0x, N1x = N01 + N00, N10 + N11
        Nx0, Nx1 = N00 + N10, N01 + N11

        if (N1x * Nx1) > 0 and (N * N11) > 0:
            mi += (N11 / N) * math.log2((N * N11) / (N1x * Nx1))
        if N0x * Nx1 > 0 and (N * N01) > 0:
            mi += (N01 / N) * math.log2((N * N01) / (N0x * Nx1))
        if N1x * Nx0 > 0 and (N * N10) > 0:
            mi += (N10 / N) * math.log2((N * N10) / (N1x * Nx0))
        if N0x * Nx0 > 0 and (N * N00) > 0:
            mi += (N00 / N) * math.log2((N * N00) / (N0x * Nx0))

        expected_N11 = (N1x * Nx1) / N
        expected_N01 = (N0x * Nx1) / N
        expected_N10 = (N1x * Nx0) / N
        expected_N00 = (N0x * Nx0) / N
        chi_squared = ((N11 - expected_N11) ** 2 / expected_N11 if expected_N11 else 0) + \
                    ((N01 - expected_N01) ** 2 / expected_N01 if expected_N01 else 0) + \
                    ((N10 - expected_N10) ** 2 / expected_N10 if expected_N10 else 0) + \
                    ((N00 - expected_N00) ** 2 / expected_N00 if expected_N00 else 0)
        return mi, chi_squared



    
    
stopwords = "ttds_stopwords.txt"
sentiment_train_path = "train-sentiment.txt"
sentiment_test_path = "ttds_2023_cw2_test_sentiment.txt"

data_loader = DataLoader()
data_loader.set_stopwords("ttds_stopwords.txt")
data_loader.load_data("train-sentiment.txt")
data_loader.shuffle_data()
data_loader.split_data()

sample_data = data_loader.get_sample("train", sample_size=5)
train_labels, train_texts = data_loader.get_labels_and_data("train")
dev_labels, dev_texts = data_loader.get_labels_and_data("dev")
test_labels, test_texts = data_loader.get_labels_and_data("test")





# Baseline model code import time
def convert_to_bow_matrix(preprocessed_data, word2id):
    stemmer = PorterStemmer()
    # matrix size is number of docs x vocab size + 1 (for OOV)
    matrix_size = (len(preprocessed_data),len(word2id)+1)
    oov_index = len(word2id)
    # matrix indexed by [doc_id, token_id]
    X = scipy.sparse.dok_matrix(matrix_size)

    # iterate through all documents in the dataset
    for doc_id,doc in enumerate(preprocessed_data):
        for word in doc:
            #word = word.lower()
            # default is 0, so just add to the count for this word in this doc
            # if the word is oov, increment the oov_index
            X[doc_id,word2id.get(word,oov_index)] += 1
    
    return X




def preprocess_data_baseline(data):
    
    chars_to_remove = re.compile(f'[{string.punctuation}]')
    
    documents = []
    categories = []
    vocab = set([])
    
    lines = data.split('\n')
    
    for line in lines:
        # make a dictionary for each document
        # word_id -> count (could also be tf-idf score, etc.)
        line = line.strip()
        if line:
            # split on tabs, we have 3 columns in this tsv format file
            tweet_id, category, tweet = line.split('\t')

            # process the words
            words = chars_to_remove.sub('',tweet).lower().split()
            for word in words:
                vocab.add(word)
            # add the list of words to the documents list
            documents.append(words)
            # add the category to the categories list
            categories.append(category)
            
    return documents, categories, vocab


def preprocess_data(data, tokeniser):
    
    documents = []
    categories = []
    vocab = set([])
    
    for tweet_id, category, tweet in data:
        # make a dictionary for each document
        # filter stopwords.
        # tweet_tokens = list(filter(lambda x: x not in stopwords, tokeniser.tokenise(tweet)))
        # tweet_tokens = list(tokeniser.tokenise(tweet))
        tweet_tokens = list(map(str.lower, tokeniser.tokenise(tweet)))
        # replace words starting with @ with <user>
        #tweet_tokens = ['<USER>' if token.startswith('@') else token for token in tweet_tokens]
        tweet_tokens = [token[1:] if token.startswith('@') else token for token in tweet_tokens]  
        # replace words starting with # with <hashtag>
        # tweet_tokens = ['<HASHTAG>' if token.startswith('#') else token for token in tweet_tokens]
        
        # just replace the removing the hashtag symbol
        tweet_tokens = [token[1:] if token.startswith('#') else token for token in tweet_tokens]    
        
        
        # tweet_tokens = list(map(str.lower, filter(lambda x: x not in stopwords, tokeniser.tokenise(tweet))))
        # tweet_tokens = list(map(stemmer.stem, tokeniser.tokenise(tweet)))
        
        for token in tweet_tokens:
            vocab.add(token)
        
        documents.append(tweet_tokens)
        categories.append(category)
        
    return documents, categories, vocab

def compute_accuracy(predictions, true_values):
    num_correct = 0
    num_total = len(predictions)
    for predicted,true in zip(predictions,true_values):
        if predicted==true:
            num_correct += 1
    return num_correct / num_total





def join_train_set(data):
    return "\n".join(map(lambda x: "\t".join(map(str, x)),data))

train_set_data_baseline, train_set_categories_baseline, train_set_vocab_baseline = preprocess_data_baseline(join_train_set(train_set))
test_set_data_baseline, test_set_categories_baseline, test_set_vocab = preprocess_data_baseline(join_train_set(test_set))
dev_set_data_baseline, dev_set_categories_baseline, dev_set_vocab_baseline = preprocess_data_baseline(join_train_set(dev_set))


final_test_data_baseline, final_test_categories_baseline, final_test_vocab_baseline = preprocess_data_baseline(join_train_set(sentiment_test_final))

word2id = {}
id2word = {}

for word_id, word in enumerate(train_set_vocab_baseline):
    word2id[word] = word_id
    id2word[word_id] = word
    
cat2id = {}
for cat_id, cat in enumerate(set(train_set_categories_baseline)):
    cat2id[cat] = cat_id

X_train_baseline = convert_to_bow_matrix(train_set_data_baseline, word2id)
y_train_baseline = [cat2id[cat] for cat in train_set_categories_baseline]

X_dev_baseline = convert_to_bow_matrix(dev_set_data_baseline, word2id)
y_dev_baseline = [cat2id[cat] for cat in dev_set_categories_baseline]

X_test_baseline = convert_to_bow_matrix(test_set_data_baseline, word2id)
y_test_baseline = [cat2id[cat] for cat in test_set_categories_baseline]

X_final_test_baseline = convert_to_bow_matrix(final_test_data_baseline, word2id)
y_final_test_baseline = [cat2id[cat] for cat in final_test_categories_baseline]

final_train_baseline = convert_to_bow_matrix(train_set_data_baseline + dev_set_data_baseline, word2id)
final_train_y_baseline = [cat2id[cat] for cat in train_set_categories_baseline + dev_set_categories_baseline]

y_train_improved = [cat2id[cat] for cat in train_labels]
y_dev_improved = [cat2id[cat] for cat in dev_labels]
y_test_improved = [cat2id[cat] for cat in test_labels]



# SVM model
svm_model_baseline = SVC(C=1000)
svm_model_baseline.fit(X_train_baseline, y_train_baseline)
y_train_pred_baseline = svm_model_baseline.predict(X_train_baseline)
y_dev_pred_baseline = svm_model_baseline.predict(X_dev_baseline)
y_test_pred_baseline = svm_model_baseline.predict(X_final_test_baseline)

print("SVM model")
print("Training accuracy: ", compute_accuracy(y_train_pred_baseline, y_train_baseline))
print("Dev accuracy: ", compute_accuracy(y_dev_pred_baseline, y_dev_baseline))
print("Test accuracy: ", compute_accuracy(y_test_pred_baseline, y_final_test_baseline))
print()


tokeniser = Tokeniser(preserved_patterns=preserve_patterns, drop_patterns=drop_patterns, lowercase=False)
stemmer = PorterStemmer()
tokenNormaliser = TokenNormaliser(stemmer=stemmer, stopwords=data_loader.stopwords, stopping=False, stemming=True, lowercase=False, min_token_len=1)
tweetProcessor = TweetProcessor()
tweetProcessor.setTokeniser(tokeniser)
tweetProcessor.setTokenNormaliser(tokenNormaliser)

limit = len(train_texts)
tweetProcessor.index_tweets(train_texts[:limit],train_labels[:limit])

tweetProcessor.updateBowVocabularySize(len(tweetProcessor.vocabulary))

final_test_data = DataLoader()
final_test_data.load_data(sentiment_test_path)


x_data, y_data = [], []

for (_, category, tweet) in final_test_data.data:
    x_data.append(tweet)
    y_data.append(category)

y_data = [cat2id[cat] for cat in y_data]



x_train_improved = tweetProcessor.convert_to_bow(train_texts[:limit], weight=1) #convert_to_bow_matrix(train_texts, word2id) #tweetProcessor.convert_to_bow(train_texts[:limit], weight=1)
x_dev_improved = tweetProcessor.convert_to_bow(dev_texts[:limit], weight=1) #convert_to_bow_matrix(dev_texts, word2id) #
x_test_improved = tweetProcessor.convert_to_bow(test_texts[:limit], weight=1) #convert_to_bow_matrix(test_texts, word2id) #
x_test_final_improved = tweetProcessor.convert_to_bow(x_data, weight=1)

# Logistic Regression model
logistic_model_improved = LogisticRegression(solver='liblinear',class_weight='balanced', penalty='l1', max_iter=10000)  #, multi_class='auto')

logistic_model_improved.fit(x_train_improved.A, y_train_improved)
print("Logistic Regression model improved")

y_train_pred_improved = logistic_model_improved.predict(x_train_improved)
print("Training accuracy: ", compute_accuracy(y_train_pred_improved, y_train_improved))

y_dev_pred_improved = logistic_model_improved.predict(x_dev_improved)
print("Dev accuracy: ", compute_accuracy(y_dev_pred_improved, y_dev_improved))

y_final_pred_improved = logistic_model_improved.predict(x_test_final_improved)
print("Final accuracy: ", compute_accuracy(y_final_pred_improved, y_data))


def save_classification_report_to_csv(y_true_train, y_true_dev, y_true_test, 
                                    y_pred_train_baseline, y_pred_dev_baseline, y_pred_test_baseline,
                                    y_pred_train_improved, y_pred_dev_improved, y_pred_test_improved, y_dev_improved, csv_filename):
    report_baseline_train = classification_report(y_true_train, y_pred_train_baseline, output_dict=True)
    report_baseline_dev = classification_report(y_true_dev, y_pred_dev_baseline, output_dict=True)
    report_baseline_test = classification_report(y_true_test, y_pred_test_baseline, output_dict=True)
    report_improved_train = classification_report(y_true_train, y_pred_train_improved, output_dict=True)
    report_improved_dev = classification_report(y_dev_improved, y_pred_dev_improved, output_dict=True)
    reprot_improved_test = classification_report(y_true_test, y_pred_test_improved, output_dict=True)

    # clear csv file 
    open(csv_filename, 'w').close()
    
    #pprint(report_baseline)
    
    # Open CSV file for writing
    with open(csv_filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # system,split,p-pos,r-pos,f-pos,p-neg,r-neg,f-neg,p-neu,r-neu,f-neu,p-macro,r-macro,f-macro
        header = ["system", "split", "p-pos", "r-pos", "f-pos", "p-neg", "r-neg", "f-neg", "p-neu", "r-neu", "f-neu", "p-macro", "r-macro", "f-macro"]
        writer.writerow(header)
        
        # Function to write row data
        def write_row(system, name, report):
            row = [
                system, name,
                report.get('1', {}).get('precision', 'NA'),
                report.get('1', {}).get('recall', 'NA'),
                report.get('1', {}).get('f1-score', 'NA'),
                report.get('0', {}).get('precision', 'NA'),
                report.get('0', {}).get('recall', 'NA'),
                report.get('0', {}).get('f1-score', 'NA'),
                report.get('2', {}).get('precision', 'NA'),
                report.get('2', {}).get('recall', 'NA'),
                report.get('2', {}).get('f1-score', 'NA'),
                report['macro avg']['precision'],
                report['macro avg']['recall'],
                report['macro avg']['f1-score']]
            writer.writerow(row)

        # Write the classification report to CSV
        write_row("baseline", "train", report_baseline_train)
        write_row("baseline", "dev", report_baseline_dev)
        write_row("baseline", "test", report_baseline_test)
        write_row("improved", "train",report_improved_train)
        write_row("improved", "dev", report_improved_dev)
        write_row("improved", "test",reprot_improved_test)

    
save_classification_report_to_csv(y_train_baseline
                                , y_dev_baseline
                                , y_final_test_baseline
                                , y_train_pred_baseline
                                , y_dev_pred_baseline
                                , y_test_pred_baseline
                                , y_train_pred_improved
                                , y_dev_pred_improved
                                , y_final_pred_improved
                                ,y_dev_improved
                                ,"classification.csv")
