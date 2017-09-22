# -*- coding: utf-8 -*-
import re 

One_to_Three_AA = {'C': 'Cys', 'D': 'Asp', 'S': 'Ser', 'Q': 'Gln', 'K': 'Lys',
         'I': 'Ile', 'P': 'Pro', 'T': 'Thr', 'F': 'Phe', 'N': 'Asn', 
         'G': 'Gly', 'H': 'His', 'L': 'Leu', 'R': 'Arg', 'W': 'Trp', 
         'A': 'Ala', 'V': 'Val', 'E': 'Glu', 'Y': 'Tyr', 'M': 'Met'}

pattern = re.compile('|'.join(One_to_Three_AA.keys()))


##### Get variation types by using regex
def variation_regex(data, pattern): # if you want to not ignore cases, add extra argument to function
    Boolean = [not bool(re.search(pattern, i, re.IGNORECASE)) for i in data.Variation]
    data_no_regex = data[Boolean]  # 182 Fusions => 495 over 
    not_Boolean = [not i for i in Boolean]  
    data_regex = data[not_Boolean]
    
    return (data_regex, data_no_regex)



##### Function to find substitutions variations in text
##### Substitution mutations = one amino acid substituted by another
##### can be for example a missense, nonsense, synonymous, etc.
def find_sub(data):

    ##### The normal case is around 2080 out of the 2644
    
    
    Boolean = [data.Variation[i] in data.Text[i] or #normal case
               data.Variation[i][:-1] in data.Text[i] or #case 1.
               pattern.sub(lambda x: One_to_Three_AA[x.group()], data.Variation[i][:-1]) # case2
               in data.Text[i]  for i in data.index] ## because new indexing we use 
    
    #TODO could also match insensitive as a next step for more info.
    #Shorter Boolean below = the normal version
    
    #Boolean = [trainSub.Variation[i] in trainSub.Text[i] #normal case
    #           for i in trainSub.ID] ## because new indexing we use ID
    #           
            
    sub_in_text = data[Boolean]
    not_Boolean = [not i for i in Boolean]  

    sub_not_in_text = data[not_Boolean]
#    sub_in_text['Count'] = [sub_in_text.Text[i].count(sub_in_text.Variation[i][:-1])
#                    +sub_in_text.Text[i].count(pattern.sub(lambda x: One_to_Three_AA[x.group()], sub_in_text.Variation[i][:-1]))
#                    for i in sub_in_text.index]
    
    return sub_in_text, sub_not_in_text

##### For subs that are not find in text: use regex to account for a different number
##### TODO: things you can further try - with AA name replacement, searching for the number only etc.
def find_sub_noText(data):
    Booleans = []
    for i in data.index:
        split_variation = re.split('(\d+)', data.Variation[i]) # split based on a number
        first_Amino = re.escape(split_variation[0]) #re.escpae uses variable as regex
        last_Amino = re.escape(split_variation[-1])
        #first_number = re.escape(split_variation[1][0])
        #new_regex = r"[^a-zA-Z0-9]" + first_Amino + first_number
        new_regex  = first_Amino + r"\d+" + last_Amino
        Boolean = bool(re.search(new_regex, data.Text[i]))
        Booleans.append(Boolean)
    
    sub_number_in_text = data[Booleans]
    not_Boolean = [not i for i in Booleans]  

    sub_again_no_text = data[not_Boolean]
    return sub_again_no_text, sub_number_in_text


##### Next we use a window to extract sentences
def get_sentences_sub(data, splitted_sentences, window_left, window_right):
    #position_sentences = [[] for _ in range(len(data))]  #### currently not used
    data.index = range(len(data))
    sentences_with_sub = [[] for _ in range(len(data))]
    
    for i in range(len(splitted_sentences)):
        sentences = splitted_sentences[i]
        one_to_three_variation = pattern.sub(lambda x: One_to_Three_AA[x.group()], data.Variation[i][:-1])
        Variation = data.Variation[i][:-1]        
        for j in range(len(sentences)):                              
            if (Variation in sentences[j]) or (one_to_three_variation in sentences[j]):
                new_regex = re.escape(Variation) + r"[\S]*" ### Means no white space 0 or more
                sentences[j] = re.sub(new_regex, ' placeholderMutation', sentences[j]) #case 1
                new_regex = re.escape(one_to_three_variation) + r"[\S]*"
                sentences[j] = re.sub(new_regex, ' placeholderMutation', sentences[j]) #case 2
                sentences_with_sub[i].extend(sentences[j-window_left : j+1+window_right])
                
                ### We add space to ' placeholderMutation' because sometimes there are letters in front of it
                # position_sentences[i].append(j) # not used for the moment

    return sentences_with_sub   ### This might take a while because it's looping through all sentences

def get_sentences_sub_noText(data, splitted_sentences, window_left, window_right):
    #position_sentences = [[] for _ in range(len(data))]  #### currently not used
    data.index = range(len(data))
    sentences_with_sub = [[] for _ in range(len(data))]
    
    for i in range(len(splitted_sentences)):
        sentences = splitted_sentences[i] 
        for j in range(len(sentences)):
            split_variation = re.split('(\d+)', data.Variation[i]) # split based on a number
            first_Amino = re.escape(split_variation[0]) #re.escpae uses variable as regex
            last_Amino = re.escape(split_variation[-1])
            new_regex  = first_Amino + r"\d+" + last_Amino
            
            Boolean = bool(re.search(new_regex, sentences[j]))
            if Boolean:
                sentences[j] = re.sub(new_regex, ' placeholderMutation', sentences[j]) # Might help catch sy
                sentences_with_sub[i].extend(sentences[j-window_left : j+1+window_right])
                # position_sentences[i].append(j) # not used for the moment

    return sentences_with_sub   ### This might take a while because it's looping through all sentences
#### Converts list of sentences into one string of sentences for each document => to use for tfidf etc.
def sentences_to_string(sentences_list):
    sentence_strings = []
    for sentences in sentences_list:
        sentence_string =  ' '.join(str(sentence) for sentence in sentences)
        sentence_strings.append(sentence_string)
    
    return sentence_strings ### This doesn't take such a long time to run



#def get_sentences_with_sub(data, splitted_sentences):
#    #position_sentences = [[] for _ in range(len(data))]  #### currently not used
#    data.index = range(len(data))
#    sentences_with_sub = [[] for _ in range(len(data))]
#    
#    for i in range(len(splitted_sentences)):
#        sentences = splitted_sentences[i] 
#        for j in range(len(sentences)):
#            if (data.Variation[i][:-1] in sentences[j]) or (pattern.sub(lambda x: One_to_Three_AA[x.group()], data.Variation[i][:-1]) in sentences[j]):
#                sentences_with_sub[i].append(sentences[j-1:j+2])
#                
#               # position_sentences[i].append(j) # not used for the moment
#    
#    return sentences_with_sub   ### This might take a while because it's looping through all sentences

# txt = re.sub(u"\u2013", "-", txt)
    
    