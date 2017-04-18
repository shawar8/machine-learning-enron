import numpy as np
import pandas as pd
import os
import re
from nltk import FreqDist
import wordcloud
import nltk
import matplotlib.pyplot as plt
from nltk.stem.snowball import SnowballStemmer
import string
from nltk.corpus import stopwords
'''------------------------------------------------------------------------'''

'''The parseOutText function in the code above, converts all the words to
lowercase, removes punctuations, removes stop words or words that dont really 
contribute to the model as they are very common and also converts all the words
to its root word. It also removes common words that used often in emails'''



# Creating list of things I will need
'''------------------------------------------------------------------'''
stemmer = SnowballStemmer('english')  ## function to stem english words
names = os.listdir('./maildir') # creating list of the names of the executives

##creating a list of common words used in emails to remove them
common_email_words = ['subject', 'pm', 'am', 'cc', 'messag', 'thank',
                          'pleas', 'forward', 'inform', 'attach', 'request',
                          'sent', 'email', 'would', 'could', 'know', 'get',
                          'need', 'let', 'may', 'like', 'think', 'origin',
                          'look', 'make', 'im', 'see', 'call', 'meet'] 
                          
##creating a list of common english words which add no value
stopwords = stopwords.words('english')         
'''------------------------------------------------------------------'''
### Creating a function that will take out each word, check if it is a common 
### english word/ email word,and if not, will stem it and return the email text
def parseOutText(f):
    final = []
    """ given an opened email file f, parse out all text below the
        metadata block at the top.
               
        example use case:
        f = open("email_file_name.txt", "r")
        text = parseOutText(f)
        
        """

    
    ### The list above contains words that are very commonly used in writing emails
    f.seek(0)  
    all_text = f.read()
    
    ### split off metadata using regular expression
    content = all_text.split(re.search(r'X-File.+', all_text).group())
    
    #content = all_text.split('X-FileName:')
    words = ""
    if len(content) > 1:
        ### remove punctuation
        text_string = content[1].translate(string.maketrans("", ""),
        string.punctuation)
        
        words = text_string.split()  ## splitting each word of an email to 
                                     ### items in a list
        
        for word in words:
            lower_case = word.lower()
            stemmed = stemmer.stem(lower_case)
            if stemmed not in stopwords and stemmed not in common_email_words:
                    final.append(stemmed)
        ### split the text string into individual words, check if the word is a 
        ### common word (I, me, you, the, etc) or a common email word, and if
        ### not, append the stemmed word to final. 
        
    return ' '.join(final)  ### returning the email from items of a list,
                            ### back to a string                 
'''------------------------------------------------------------------------'''
    
                      
'''What I will be looking to do in this project is that I will be extracting 
only the sent mails of each executive, and after cleaning it, will try to 
create a model that will help me to identify whom the email came from. '''


email_text = []   ### an empty list to append all the sent emails to
executive_list = []  ###explained below


'''In the code below, if you want to take only a sample, comment out the 
code'''

exec_number = 0
count = 0
acount = 0
for name in names:
        #if count == 30:
            #break
        #else:
            #count += 1
        
            exec_number += 1
            
            folders = os.listdir('./maildir/{}'.format(name))
            for folder in folders:
                if folder in ['_sent_mail', 'sent', 'sent_items']:
                    
                    files = os.listdir('./maildir/{}/{}'.format(name, folder))
                    for email in files:
                        
                            print name  ## to see whose mail is being extracted
                            text = open('./maildir/{}/{}/{}'.format(name,
                                        folder, email), 'r')
                            full_email = (parseOutText(text))
                            email_text.append(full_email)
                            executive_list.append(exec_number)  
                    
'''The executive list contains the numbers that correspond to the executives
names contained in the names list. (1 stands for allen-p, 150 stands for 
zufferli-j)'''

                             


'''First I will create a wordcloud that will exhibit more common words in a 
colorful way'''


from wordcloud import WordCloud

text = ' '.join(email_text)


wordcloud = WordCloud().generate(text)
plt.imshow(wordcloud)
plt.axis("off")



'''Some of the words you can clearly see are Enron, energy, deal etc which 
makes sense considering the operations enron were into and the size of the 
company. 

Other words that would stand out for somebody who didn't have much idea about 
Enron would be California, Power, etc. Going deeper into this will tell you 
about the time where California had frequent power outages due to shortage in 
electricity supply caused by market manipulations, illegal shutdowns of 
pipelines and capped retail electricity prices.

A demand supply gap was created by energy companies, mainly Enron, to create 
an artificial shortage. Energy traders took power plants offline for 
maintenance in days of peak demand to increase the price. Traders were 
thus able to sell power at premium prices, sometimes up to a factor of 20 
times its normal value. 

Because the state government had a cap on retail 
electricity charges, this market manipulation squeezed the industry's revenue 
margins, causing the bankruptcy of Pacific Gas and Electric Company (PG&E) and 
near bankruptcy of Southern California Edison in early 2001.

This crisis was so massive that then governer Gray Davis was forced to declare
a state of emergency.'''

'''------------------------------------------------------------------------'''

'''Next I will use nltk word tokenize class and freqDist function to show the 
50 most commonly used words'''



words = nltk.tokenize.word_tokenize(text)
fdist = FreqDist(words)

fdist.most_common(50)



'''----------------------------------------------------------------------'''

###Using Tfid Vectorizer to transform the dataset to a sparse matrix

'''What the Tfid Vectorizer does is it shows us the weightage of each word in 
an email based on the number of times the word shows up in each document(tf) 
as well as in the number of documents we find that word in(idf). However, the 
larger the document size, the higher probability a word would show up more 
times. Therefore, tf is actually the ratio of the number of times a word shows 
up in a document to the total number of words in that document. The tfidf, in 
a sense, gives weightage to a word based on how common the word is'''

###defining the vectorizer, removing stopwords and transforming to matrix
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df = 50)
x = vectorizer.fit_transform(email_text).toarray()


### Splitting the dataset into a training and test set

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, executive_list,
                                                    test_size = 0.2,
                                                    random_state = 0)

'''------------------------------------------------------------------------'''

'''I decided to use two models and check which one would be better to perform
this text analysis. One method is by using the gaussian method of naive_bayes.
The second is a more linear model which is very good when working with texts 
as it is very good in dealing with sparse matrices such as this created by the
TfidVectorizer'''

##Naive_bayes

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)


from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score

print 'Precision -------->',(precision_score(y_test, y_pred))
print 'Recall----------> ',(recall_score(y_test, y_pred))
print 'Accuracy----------> ', accuracy_score(y_test, y_pred)

'''We get very good values of precision, recall and accuracy of 77%, 71% and
and 71% respectively'''


## linear model (SGDClassifier)

from sklearn.linear_model import SGDClassifier
clf = SGDClassifier()
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

print 'Precision -------->',(precision_score(y_test, y_pred))
print 'Recall----------> ',(recall_score(y_test, y_pred))
print 'Accuracy----------> ', accuracy_score(y_test, y_pred)

'''As we can see, the SGDClassifier which is very good with text data having
very high precision, recall and accuracy values of 90% each.'''


