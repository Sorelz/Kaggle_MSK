# -*- coding: utf-8 -*-

####### This will be an attempt to process the train data, and build a classifier on the Leaks data
####### The step op processing the data:


import os
import pandas as pd
import re
import numpy as np
import string ## string has properties of digits, letters, punctuation etc.
from nltk import PorterStemmer

from segtok.segmenter import split_single, split_multi #seems to be the better choice than nltk despite lower speed
from nltk.tokenize import sent_tokenize, word_tokenize
import pickle
from scipy import sparse
from sklearn import metrics, model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import log_loss
from sklearn.preprocessing import OneHotEncoder

import matplotlib.pyplot as plt
import seaborn as sns
#from ConcordanceIndexClass import ConcordanceIndex2
#from sklearn.naive_bayes import MultinomialNB
#from collections import OrderedDict
import xgboost as xgb
#### for loading 
import first_text_processing  
import imp
imp.reload(first_text_processing)

from first_text_processing import find_sub, find_sub_noText, variation_regex
from first_text_processing import get_sentences_sub, get_sentences_sub_noText, sentences_to_string

#import seaborn as sns
#import matplotlib.pyplot as plt


pd.options.mode.chained_assignment = None  # default='warn'

cwd = os.getcwd()
train = pd.read_csv(cwd + '\\data\\training_variants')
test = pd.read_csv(cwd + '\\data\\test_variants')

train_texts = pd.read_csv(cwd + '\\data\\training_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"], encoding = "utf-8")
test_texts = pd.read_csv(cwd + '\\data\\test_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"], encoding = "utf-8")

train = pd.merge(train, train_texts, how='left', on='ID')
test = pd.merge(test, test_texts, how='left', on='ID')


###### OR USE THIS IF U WANT TO USE THE LEAKS!!!!! (not necessary!) ##############################s
pickle_in = open(cwd + "\\data\\Leaks\\Leaks_all", "rb") #### OR use Leaks for the full leaks (no sub)
Leaks = pickle.load(pickle_in)

##### Rename test to Leaks, because we are now using only the Leaks for validation ##########
test = Leaks
test.index = test.ID
Class_leaks = test.Class
test = test.drop(['Class'], axis=1) # drop Class, to be used for validation later on 

#test['length_text'] = test['Text'].map(lambda x: len(str(x).split(' '))) 
#test.length_text [test.length_text<300]
# null for ID 
#train['length_text'] = train['Text'].map(lambda x: len(str(x).split(' '))) 
#train.length_text [train.length_text<300]
#len(Leaks[Leaks.Count>1]) # how many times it appears in text

#### process the train and test set together
data_all = pd.concat((train, test), axis=0, ignore_index=True)
data_all_backup = data_all[:] ##### We keep backup because we want dummy variables of Gene & Text 
# TODO maybe also use Variation function of Gene from a database, and other suggestions. Also can use Count_sub as feature

# data_all.Text[data_all.Text=='null'] = NaN ##### replace null with NaN?

###############################################################################################################################
############# Subtitutions (subs) processing of data set ############################

######### First find those that have the format of being a substitution in data
data_all['Substitutions_var'] = data_all.Variation.apply(lambda x: bool(re.search('^[A-Z]\\d+[A-Z*]$', x))*1) #multiplying by 1 converts True to 1, False to 0 => Maybe modify this later?
data_all['Stop_codon_var'] = data_all.Variation.apply(lambda x: bool(re.search('[*]', x))*1) #multiplying by 1 converts True to 1, False to 0

data_sub = data_all[data_all['Substitutions_var']==1] ### Now we know the index of where a substitution occurs - the data_sub

sub_in_text, sub_not_in_text = find_sub(data_sub)
sub_in_text_backup = sub_in_text[:] ## index gets changed by text_processing if we don't make a copy

##### INVESTIGATION: Why do some subs don't appear in Text?: Try to automize this and find out
### Substitutions can appear as SAME_PREFIX - Other number - SAME_SUFFIX

sub_again_no_Text, sub_noText = find_sub_noText(sub_not_in_text) # 108 such cases out of 411 = nice improvement
sub_noText_backup = sub_noText[:]

#sub_again_no_Text['length_text'] = sub_again_no_Text['Text'].map(lambda x: len(str(x).split(' '))) 
#sub_again_no_Text.length_text [sub_again_no_Text.length_text<300]

# sum(sub_noText.Class.isnull()) => 30 from test set
#### TODO: change all substitutions to the same name in text => could catch more of the relation between sub and Text


##### The next takes a bit long, so we saved it into the pickle sub_sentences below
NLTK_sub_noText = [sent_tokenize(sub_noText.Text[i]) for i in sub_noText.index]
sub_noText_sentences = get_sentences_sub_noText(sub_noText, NLTK_sub_noText, window_left = 2, window_right = 2) # Retrieves sentences where subsitution mutation is included
sub_noText_sentences = [sorted(set(sentences), key = sentences.index) for sentences in sub_noText_sentences] # only use unique sentences

##################### 
NLTK_sub = [sent_tokenize(sub_in_text.Text[i]) for i in sub_in_text.index] # takes a long time to run tokenizer => use pickle to save
sub_sentences = get_sentences_sub(sub_in_text, NLTK_sub, window_left = 2, window_right = 2) 
# Retrieves sentences where subsitution mutation is included.
# window_left and window_right specify which sentences to keep at the left side or right side of the sub sentences.
# IMPORTANT: I used also placeholderMutation to replace the original sub mutations here

sub_sentences = [sorted(set(sentences), key = sentences.index) for sentences in sub_sentences]
# removes duplicates

############
pickle_in = open(cwd + "\\data\\Tokenized sentences\\NLTK_sub_noText", "rb")
NLTK_sub_noText = pickle.load(pickle_in) # pickle of all saved sentences

pickle_in = open(cwd + "\\data\\Tokenized sentences\\NLTK_sub", "rb")
NLTK_sub = pickle.load(pickle_in) # pickle of all saved sentences

pickle_in = open(cwd + "\\data\\Tokenized sentences\\sub_noText_sentences", "rb")
sub_noText_sentences = pickle.load(pickle_in) # pickle of saved sentences with sub

pickle_in = open(cwd + "\\data\\Tokenized sentences\\sub_sentences", "rb")
sub_sentences = pickle.load(pickle_in) # pickle of saved sentences with sub

#pickle_out = open(cwd + "\\data\\Tokenized sentences\\sub_noText_sentences", "wb")
#pickle.dump(sub_noText_sentences, pickle_out)
#pickle_out.close()

# sub_sentences_5 = [(sub_sentences[i], i) for i in range(len(sub_sentences)) if len(sub_sentences[i])==5]


sub_sentences_string = sentences_to_string(sub_sentences)
sub_noText_string = sentences_to_string(sub_noText_sentences)
#pickle_in = open(cwd + "\\data\\Tokenized sentences\\test_sentences", "rb")
#test_sentences = pickle.load(pickle_in) # dont use for Leaks, only for full train set in 2nd round

data_all.Text[sub_in_text_backup.index] = sub_sentences_string
data_all.Text[sub_noText_backup.index] = sub_noText_string

##############################################################################################################################
############## Non-subs preprocessing of data set #######################

#def find_mutation_type(row, pattern):  ##### TODO: make clearer by using a function instead of lambda
#    return bool(re.search('^fusion', row, re.IGNORECASE)) *1. Also for subs

####### Fusions : 'Fusions' ############
data_all['Fusion_var'] = data_all.Variation.apply(lambda x: bool(re.search('^fusion', x, re.IGNORECASE))*1) #multiplying by 1 converts True to 1, False to 0
new_fusion, new_data_all = variation_regex(data_all, '^fusion') 

###### Fusions: 'Gene-Gene fusion' ########
data_all['gene_fusion_var'] = new_data_all.Variation.apply(lambda x: bool(re.search('fusion', x, re.IGNORECASE))*1) 
_ , new_data_all = variation_regex(new_data_all, 'fusion') 
###### Notice that NaN introduced for places where splicing occured => replace after NaN with 0's when complete

####### Deletions: 'Deletions' ############
data_all['Deletion_var'] = new_data_all.Variation.apply(lambda x: bool(re.search('^del', x, re.IGNORECASE))*1) 
new_del, new_data_all = variation_regex(new_data_all, '^del') 

####### Deletions & Insertions wheteher together or seperately (doesn't make a big difference IMO)
data_all['del_or_ins_var'] = new_data_all.Variation.apply(lambda x: bool(re.search('del|ins', x, re.IGNORECASE))*1) 
# we could also later divide it into del, ins if we want to

###### Amplifications #########
data_all['Amplification_var'] = data_all.Variation.apply(lambda x: bool(re.search('ampl', x, re.IGNORECASE))*1) 

###### Truncations ########### Don't forget there are 'Truncating mutations' = 95 and '_trunc' = 4
data_all['Truncation_var'] = data_all.Variation.apply(lambda x: bool(re.search('trunc', x, re.IGNORECASE))*1) 

####### Exons #########
data_all['exon_var'] = data_all.Variation.apply(lambda x: bool(re.search('exon', x, re.IGNORECASE))*1) 

####### Frameshift mutations ########
data_all['frameshift_var'] = data_all.Variation.apply(lambda x: bool(re.search('fs', x, re.IGNORECASE))*1) 

####### Duplications ##############
data_all['dup_var'] = data_all.Variation.apply(lambda x: bool(re.search('dup', x, re.IGNORECASE))*1) 

data_all.fillna(0, inplace = True)

##### TODO: sentence tokenizer for the non-subs as well !!


################################################################################################################################

################### Building models on our data #############################################################################

################ The dummy variables for Gene and Text ##################
## TODO: also use dummy for Text? There are 135 shared Genes and 142 shared Text between train and Leaks!  len(set(train.Text) & set(Leaks.Text))
data_all_dummy = data_all_backup[['Gene', 'Text']] # drop those columns we don't need as dummy.
X_dummy = pd.get_dummies(data_all_dummy) # converts categorical variables into dummy variable. From len set => 269 genes + 2090 texts
X_dummy_train = X_dummy[:train.shape[0]]
X_dummy_test = X_dummy[train.shape[0]:]

dummy_names = X_dummy.columns.values #### To remember names if you want to check again what Gene or Text used
#X_dummy = X_dummy.values

###### Use the variation types 
variation_types = data_all.drop(['ID', 'Gene', 'Class', 'Text', 'Variation'], axis =1)
X_variation_train = variation_types[:train.shape[0]]
X_variation_test = variation_types[train.shape[0]:]

variation_names = variation_types.columns.values 

############### Process text into a bag-of-words matrix with values from tfidf ##################
# TODO: other formations like word2vec, genism. Maybe POS tagging or other syntax relations to derive semantics.

########## OPTIONAL: More processing of the text. Processing can delete unneeded words #############
####### Make the text lower case ###########
#data_all.Text = data_all.Text.apply(lambda words: words.lower()) # This is already done in tfidf option lowercase = True

######## Remove any digits (for now, the digits have no real meaning). And even words that contain digits. ################
data_all.Text = data_all.Text.apply(lambda words: ' '.join(word for word in words.split() if not any(letter.isdigit() for letter in word))) # taken from https://stackoverflow.com/questions/18082130/python-regex-to-remove-all-words-which-contains-number => also regex

##### Remove stopwords???? #######
##### Use Porter Stemmer for stemmming the sentences #######

######## Remove any punctuation in text ########## ☺☺☺☺☺☺☺☺☺☺☺
#remove_punctuation = str.maketrans('', '', string.punctuation)
#data_all.Text = data_all.Text.apply(lambda words: words.translate(remove_punctuation))
##### Remove stopwords???? #######
##### Use Porter Stemmer for stemmming the sentences #######
#data_all.Text = data_all.Text.apply(lambda words: ' '.join(PorterStemmer().stem(word) for word in words.split()))#

#stemmer = PorterStemmer()  ### define our own tokenizer for tfidf. Taken from https://stackoverflow.com/questions/26126442/combining-text-stemming-and-removal-of-punctuation-in-nltk-and-scikit-learn
#
#def stem_tokens(tokens, stemmer):
#    stemmed = []
#    for item in tokens:
#        stemmed.append(stemmer.stem(item))
#    return stemmed
#
#def tokenize(text):
#    tokens = word_tokenize(text)
#    tokens = [i for i in tokens if i not in string.punctuation]
#    stems = stem_tokens(tokens, stemmer)
#    return stems
##### 

tfidf_all = data_all.Text 
tfidf_train = tfidf_all[:train.shape[0]]
tfidf_test = tfidf_all[train.shape[0]:]

### TFIDF for term-document matrix and probabilities # TODO: optimize tfidf for better performance 
#### Optimization questions: how many ngrams, use_idf, max_features etc.
   
tfidf = TfidfVectorizer(
	min_df=10, max_features=300, lowercase =True, #tokenizer = tokenize,
	analyzer='word', token_pattern=r'\w+', ngram_range=(1, 3), use_idf=True, 
	smooth_idf=True, sublinear_tf=True, stop_words = 'english'
).fit(tfidf_train) #### does take about a minute or two

tfidf_names =tfidf.get_feature_names()

X_text_train = tfidf.transform(tfidf_train)
X_text_train_matrix = pd.DataFrame(X_text_train.toarray(), columns=tfidf_names) #can observe the matrix

X_text_test = tfidf.transform(tfidf_test)
X_text_test_matrix = pd.DataFrame(X_text_test.toarray(), columns=tfidf_names)



##### Stack all data together into a sparse matrix 
X_train = sparse.hstack((X_dummy_train, X_variation_train, X_text_train), format='csr')
X_test = sparse.hstack((X_dummy_test, X_variation_test, X_text_test), format='csr')

y_train = train.Class.values - 1 # we use 0 to 8 for classed instead of 1 to 9

feature_names = dummy_names.tolist() + variation_names.tolist() + tfidf_names ### feature names altogether 

####################################################################################################################################




########################## XGboost model #################################
dtrain = xgb.DMatrix(X_train, label = y_train) # add feature names to understand feature importance  
dtest  = xgb.DMatrix(X_test)


# TODO optimize XGboost parameters 
params = { # see http://xgboost.readthedocs.io/en/latest/parameter.html for all parameters
    'max_depth': 12,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'eta': 0.1,
    'gamma': 0,
    'min_child_weight': 1,
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'num_class': 9
}

####### xgboost.cv(params, dtrain, num_boost_round=10, nfold=3, stratified=False, folds=None, metrics=(), obj=None, feval=None, maximize=False, early_stopping_rounds=None, fpreproc=None, as_pandas=True, verbose_eval=None, show_stdv=True, seed=0, callbacks=None, shuffle=True)
####### http://xgboost.readthedocs.io/en/latest/python/python_api.html
xgb_cv = xgb.cv(params = params, dtrain = dtrain, early_stopping_rounds = 20, nfold = 5, num_boost_round=1000, verbose_eval = True, seed = 18)
best_iteration_round = len(xgb_cv) # because it only stores values of early stopping
watchlist = [(dtrain, 'train')]
xgb_model = xgb.train(params = params, dtrain = dtrain, evals = watchlist, num_boost_round = best_iteration_round, verbose_eval = True)

####### xgb model evaluation: important features, decision trees
#plt.rcParams['figure.figsize'] = (7.0, 7.0)
#xgb.plot_importance(xgb_model);  plt.show()

xgb.plot_tree(xgb_model) #need graphviz 
####### Prediction 
y_predict = xgb_model.predict(dtest)

##### Log loss compared to the given leaks
log_loss(Class_leaks-1, y_predict)  #### only received 0.7945 for this model here, 0.75689 in LB. Don't know why the difference between the two
###############################################################################################################################


######################### Logistic regression model  ################################
X = X_train
y = y_train

y_predict = np.zeros((X_test.shape[0], max(y_train)+1))

n_folds = 5

stratified_kfold = model_selection.StratifiedKFold(n_splits=n_folds, random_state=1, shuffle=True)

fold = 0

## Split into CV set to train on also and then only use on the test set

for train_index, test_index in stratified_kfold.split(X, y):

	fold += 1

	X_train_new, X_validation    = X[train_index], 	X[test_index]
	y_train_new, y_validation    = y[train_index],   y[test_index]

	print("Fold", fold, X_train_new.shape, X_validation.shape)

	log_model = LogisticRegression(C = 3 , max_iter = 100, solver = 'sag', multi_class= 'multinomial')

	log_model.fit(X_train_new, y_train_new)

	p_train = log_model.predict_proba(X_train_new)
	p_validation = log_model.predict_proba(X_validation)
	p_test = log_model.predict_proba(X_test)

	print("%f Log loss of train set" % metrics.log_loss(y_train_new, p_train))
	print("%f Log loss of CV set" % metrics.log_loss(y_validation, p_validation))
	
	y_predict += p_test/n_folds
	print(y_predict)

#### In logreg: lower C is more conservative. In my case I get worse prediction than with a higher C (for example 3 instead of 0.4)
##### Evaluation: log loss computation
log_loss(Class_leaks-1, y_predict)  #### A score of 0.67373 on LB, 0.690296 with my computation. 0.67374 the second time. (so results change in CV). 
    
###############################################################################################################################
#################### CNN model ##########################





######## Submission to Leaderboard ###############
submission = pd.read_csv(cwd + "\\data\\submission.csv")

class_probs = []

for i in range(1,10):
    class_probs.append(train[train['Class'] == i].shape[0]/3321.) ### Frequency of each class (total rows) divided by test set numbers
    
for i in range(9):
    submission['class'+str(i+1)] = class_probs[i]

y_predict = pd.DataFrame(y_predict)
submission.ix[Leaks.ID, 1:] = y_predict.values
submission.to_csv("my_own_submission", index=False)
submission.head()


######################### Support vector machine model ################################ 
######## TAKES AGES TO RUN! see https://stackoverflow.com/questions/31681373/making-svm-run-faster-in-python for techniques to make it faster
#X = X_train
#y = y_train
#
#y_predict = np.zeros((X_test.shape[0], max(y_train)+1))
#
#n_folds = 5
#
#stratified_kfold = model_selection.StratifiedKFold(n_splits=n_folds, random_state=1, shuffle=True)
#
#fold = 0
#
### Split into CV set to train on also and then only use on the test set
#
#for train_index, test_index in stratified_kfold.split(X, y):
#
#	fold += 1
#
#	X_train_new, X_validation    = X[train_index], 	X[test_index]
#	y_train_new, y_validation    = y[train_index],   y[test_index]
#
#	print("Fold", fold, X_train_new.shape, X_validation.shape)
#
#	svm_model = SVC(C=1.0, kernel = 'linear', probability = True, max_iter = 15, verbose = True)
#
#	svm_model.fit(X_train_new, y_train_new)
#
#	p_train = svm_model.predict_proba(X_train_new)
#	p_validation = svm_model.predict_proba(X_validation)
#	p_test = svm_model.predict_proba(X_test)
#
#	print("%f Log loss of train set" % metrics.log_loss(y_train_new, p_train))
#	print("%f Log loss of CV set" % metrics.log_loss(y_validation, p_validation))
#	
#	y_predict += p_test/n_folds
#	print(y_predict)
#
##### In logreg: lower C is more conservative. In my case I get worse prediction than with a higher C (for example 3 instead of 0.4)
###### Evaluation: log loss computation
#log_loss(Class_leaks-1, y_predict)  #### A score of 0.67373 on LB, 0.690296 with my computation. 0.67374 the second time. (so results change in CV). 
#    
##Some classification problems can exhibit a large imbalance in the distribution of the target classes:
#for instance there could be several times more negative samples than positive samples. 
#In such cases it is recommended to use stratified sampling 


