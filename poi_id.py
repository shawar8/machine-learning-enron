import math
import sys
import pickle
sys.path.append("../tools/")
import pandas as pd
import seaborn as sns
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import matplotlib.pyplot as plt
import numpy as np

###Opening the file and creating a dictionary called data_dict
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
    
###creating a dataframe of the data for easy exploration and cleaning


df = (pd.DataFrame(data_dict)).transpose()

original_df = (pd.DataFrame(data_dict)).transpose()

### separating the features into financial and emails

financial_features = ['other', 'bonus', 'restricted_stock', 'shared_receipt_with_poi',
                      'exercised_stock_options', 'expenses', 'deferred_income', 'long_term_incentive', 'deferral_payments']
email_features = ['to_messages', 'from_messages', 'from_this_person_to_poi',
                  'from_poi_to_this_person']
                  
'''I planned on changing the Nans in the financial features to 0 except for the salaries.
It didnt make sense to me as to how some of the executives would have 0 as a salary
So I decided to change those values to values in and around the median. This is why
I have excluded salary from the financial_features list'''

### First, I decided to make a plot of the data to have a look at outliers.

sns.boxplot(data=df[financial_features],orient='h')
sns.plt.title('Financial Features Boxplot')
plt.show()

''' At first sight, I noticed that there were a few values way above the rest. This is more easily
noticable with the exercised_stock_options feature where there is a value in the range of $300000000.
I decided to take a closer look at the data frame'''

print df[df.exercised_stock_options > 300000000]



''' This made me realize that there was a row in the dataframe associated with
the total values called TOTAL. So I decided to remove this'''

df = df.drop('TOTAL')


'''Because of this, I decided to look closer at the row names in the dataframe
and realized there was a name - [The travel agency in the park]. So I got rid of that too.'''

df = df.drop('THE TRAVEL AGENCY IN THE PARK')

'''I also noticed there was one name which did not have any values at all (Lockhart Eugene E).
So I did away with that value too'''

df = df.drop('LOCKHART EUGENE E')


'''On plotting the same boxplot above again, I noticed some values that seemed to be negative
when all other values were positive. I decided to look closely at them'''

print df[df.restricted_stock < 0]


'''I will deal with this a little later. First, I wanted to get rid of columns that did not
have much data in them and were filled with mostly Nans. I also deleted the
email addresses column. Along with this, I also deleted the total_payments column because
it was dependent on other payments'''


df = df.drop(['email_address','loan_advances','director_fees','restricted_stock_deferred',
       'director_fees', 'total_payments'], axis = 1)

                      
df = df.replace('NaN', np.nan)  ###replacing Nans to a numpy format


''' I also noticed a guy named Belfer Robert who had negative values in the deferral payments
feature which didnt coincide with the value in the pdf. This is where I decided to take the 
absolute values for all the data points even though the deferred_income feature has negative values.
This is because I will be making a classifier model, so it didnt matter what the sign was 
as long as the absolute values remained the same.'''

print df.loc['BELFER ROBERT']

df = df.abs()

'''Now I decided to fill all the nans in the financial features list with 0'''

df[financial_features] = df[financial_features].fillna(0)


'''some of the values in the total_stock_value column had 0s, so I edited them to show the sum of 
the exercised_stock_options and restricted_stock. I was also thinking about converting those values
to ratios to total_stock_values, but in some cases, the value would be divided by 0 to create
infinity. So I added 1 to the total_stock_value to avoid this problem'''


df['total_stock_value'] = df['exercised_stock_options'] + df['restricted_stock'] + 1


df['exercised_stock_ratio'] = df['exercised_stock_options']/df['total_stock_value']
df['restricted_stock_ratio'] = df['restricted_stock']/df['total_stock_value']


'''Then I deleted the stock columns that were no longer required'''

df = df.drop(['exercised_stock_options', 'restricted_stock', 'total_stock_value'], axis = 1)

'''I didnt fill the emails sent with 0s as it didnt make sense to me that a person would not send
emails at all, so for the total to and from messages, I filled the nans with the median value'''

df.loc[pd.isnull(df.from_messages), 'from_messages'] = df.from_messages.median()              

df.loc[pd.isnull(df.to_messages), 'to_messages'] = df.to_messages.median() 


''' for the messages to and from the person of interest, I filled them with random values within a
+/-5 range from the median'''

to_poi_median = df['from_this_person_to_poi'].median()
number_nans_to_poi = len(df[pd.isnull(df['from_this_person_to_poi'])])

from_poi_median = df['from_poi_to_this_person'].median()
number_nans_from_poi = len(df[pd.isnull(df['from_poi_to_this_person'])])

df.loc[pd.isnull(df.from_this_person_to_poi), 'from_this_person_to_poi'] = np.random.randint(to_poi_median - 5, 
       to_poi_median + 5, number_nans_to_poi)

df.loc[pd.isnull(df.from_poi_to_this_person), 'from_poi_to_this_person'] = np.random.randint(from_poi_median - 5, 
       from_poi_median + 5, number_nans_from_poi)




'''for the salaries, I decided to fill in the nans with the median values'''


df.loc[pd.isnull(df.salary), 'salary'] = df['salary'].median()

'''Instead of keeping 4 columns for from and to messages as well as the messages to and from the
poi, I decided to create two new columns instead that took the ratio of messages sent/received
from the poi to the total number of messages sent/received'''


df['fraction_sent_to_poi'] = df.from_this_person_to_poi/df.from_messages

df['fraction_from_poi'] =df.from_poi_to_this_person/df.to_messages


'''Finally, I deleted the email features that were no longer required'''

df = df.drop(['from_this_person_to_poi', 'from_poi_to_this_person',
              'from_messages', 'to_messages'], axis = 1)



''' Next I converted the dataframe back into a dictionary and got it ready for creating my model'''

my_dataset = df.to_dict('index')



features_list = ['poi', 'salary','other', 'bonus', 'restricted_stock_ratio', 'shared_receipt_with_poi',
                      'exercised_stock_ratio', 'expenses', 'deferred_income',
                      'long_term_incentive', 'deferral_payments',
                      'fraction_sent_to_poi', 'fraction_from_poi']
                      
original_features_list = ['poi', 'bonus', 'deferral_payments','deferred_income',
                          'director_fees', 'exercised_stock_options', 
                          'expenses', 'from_messages', 'from_poi_to_this_person',
                          'from_this_person_to_poi', 'loan_advances' ,
                          'long_term_incentive', 'other', 'restricted_stock',
                          'restricted_stock_deferred', 'salary', 'to_messages',
                          'shared_receipt_with_poi', 'total_payments', 
                          'total_stock_value']
'''------------------------------------------------------------------------'''
### Trying to run the algorithm over original features

'''In this section, I will try to run the algorithm through a pipeline by using
all the original features without creating any changes'''

original_dataset = featureFormat(data_dict, original_features_list, sort_keys = True)
labels, features = targetFeatureSplit(original_dataset)

from sklearn.cross_validation import train_test_split
labels_train, labels_test, features_train, features_test = train_test_split(labels, features, test_size = 0.25, random_state = 42) 

selector = SelectKBest(f_classif)
clf = DecisionTreeClassifier()

pipeline = Pipeline([('feature_select', selector),
                       ('clf', clf)])


pipe_params =   dict(feature_select__k = range(6,9),
                   clf__splitter = ['best', 'random'], 
                   clf__max_depth = [None, 1, 2, 3, 4],
                   clf__min_samples_split = [2, 5, 10, 15, 20],
                   clf__random_state = [42])




grid_search = GridSearchCV(estimator = pipeline, param_grid = pipe_params,
                           cv = 3, n_jobs = -1)

gs = grid_search.fit(features_train, labels_train)

print gs.best_params_ 

labels_pred = gs.predict(features_test)

print 'Precision -------->',(precision_score(labels_test,labels_pred))
print 'Recall----------> ',(recall_score(labels_test,labels_pred))

'''As you can see, the precision and recall scores are now 0. Now let us 
compare these results to using the algorithm on the new list of features we
created'''

'''------------------------------------------------------------------------'''
                      
                      
#####Extracting features and labels from local dataset for testing

dataset = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(dataset)                     
                      
features
###splitting data into training and testing set
from sklearn.cross_validation import train_test_split
labels_train, labels_test, features_train, features_test = train_test_split(labels, features, test_size = 0.25, random_state = 42) 


'''The issue now is that we have way too many features to fit into our model. The problem this 
creates is that we may over estimate relationship of our features with the label and thus result
in overfitting. Therefore it is important to take only those features that result in max change in 
the labels'''

### using the SelectKBest method to find best features
from sklearn.feature_selection import SelectKBest, f_classif, SelectPercentile


selector = SelectKBest(f_classif, k = 12)

selector.fit_transform(features_train, labels_train)  


'''SelectKBest is a form of Univariate feature selection method works by selecting 
the best features based on univariate statistical tests.'''  

###Looking at the p - Values

p_values = selector.pvalues_  
selector.scores_    
print p_values  


selected_features = ['poi']  ### poi needs to be the first element of the list in order to split the data properly

''' Of all these values, we take the ones that have a p-value less than 0.1. I decided to write code to show which features correspond to which p-value. Also,
I will append those features to the selected_features list'''

for i in selector.get_support(indices = True):
    print features_list[i+1], '------->', p_values[i]
    if p_values[i] < 0.1:
        print features_list[i+1]
        selected_features.append(features_list[i+1])
        
        
'''from this, it is clear to see the features with p-values low enough to be considered '''

print selected_features   
       
###CLassifiers

from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
### creating new labels and features based on the selected features

new_dataset = featureFormat(my_dataset, selected_features, sort_keys = True)
labels, features = targetFeatureSplit(new_dataset)
   
##splitting the data to training and test set based on selected features                   
labels_train, labels_test, features_train, features_test = train_test_split(labels, features, test_size = 0.25, random_state = 42) 
                      
'''Normally at this point, we would like to scale the dataset if we were using a classifying model
that is based on  Euclidian distance. But I will be selecting one of the tree models that dont require
us to scale the data'''

'''The decision tree classifier has many parameters that we can tune in order to get the most
efficient classifier. I will be using the GridSearchCV method to play around with the parameters
in order to make sure I get the best one.'''

clf = DecisionTreeClassifier()   ### creating the classifier

parameters = [{ 'splitter' : ['best', 'random'],
                'max_depth' : [None, 1, 2, 3, 4],
                'min_samples_split' : [2, 5, 10, 15, 20],
                'random_state' : [42]}]                      


grid_search = GridSearchCV(estimator = clf, param_grid = parameters,
                           cv = 3, n_jobs = -1, scoring = 'precision')

gs = grid_search.fit(features_train, labels_train)  ###fitting the gridsearch model
 
best_parameters = gs.best_params_   ###getting the best parameters

print best_parameters

''' This shows us that the best parameters are -> {'max_depth': None,
 'min_samples_split': 20,
 'random_state': 42,
 'splitter': 'random'} '''
 
###Creating the classifier with these parameters

clf = DecisionTreeClassifier(max_depth = None, min_samples_split = 20, random_state = 42,
                             splitter = 'random')


###fitting the model and predicting the test values
clf.fit(features_train, labels_train)
labels_pred = clf.predict(features_test)

accuracy = accuracy_score(labels_test, labels_pred)
print accuracy


'''The accuracy may be high but it is not a great way of testing the validity of our model.
 This is because even if the model predicts all the people to be non POIs, we will still
have a high accuracy. A better way would be to look at the confusion matrix'''

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, precision_score, recall_score

confusion_matrix(labels_test, labels_pred)

'''The matrix shows us that our model identified 31 people that were not pois correctly and
1 person that was a poi. The main issue of this dataset is that there arent many pois to test 
our model properly on'''



print 'Precision --------> ',(precision_score(labels_test,labels_pred))
print 'Recall----------> ',(recall_score(labels_test,labels_pred))


''' The values show us that we have a decent precision value of 0.33 and a recall score of 0.33.
The precision and recall and better methods of testing the validity of our model rather than the
accuracy'''


'''The process above was done in a series of steps for better understanding. The whole method can be 
reduced to a few steps if we use a pipeline'''
#-------------------------------------------------------------------------------

dataset = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(dataset)  

labels_train, labels_test, features_train, features_test = train_test_split(labels, features, test_size = 0.25, random_state = 42) 
 
###same calculations using a pipeline

selector = SelectKBest()
clf = DecisionTreeClassifier()
from sklearn.pipeline import Pipeline

'''The pipeline method will help us perform the feature selection and choosing the classifier in one
step. Then we use GridSearchCV to test out all the parameters and return the optimum ones.'''


pipeline = Pipeline([('feature_select', selector),
                       ('clf', clf)])


pipe_params =   dict(feature_select__k = range(6,9),
                   clf__splitter = ['best', 'random'], 
                   clf__max_depth = [None, 1, 2, 3, 4],
                   clf__min_samples_split = [2, 5, 10, 15, 20],
                   clf__random_state = [42])




grid_search = GridSearchCV(estimator = pipeline, param_grid = pipe_params,
                           cv = 3, n_jobs = -1)

gs = grid_search.fit(features_train, labels_train)

print gs.best_params_   ### to get the best parameters



'''The best parameters are {'clf__max_depth': None,
 'clf__min_samples_split': 10,
 'clf__random_state': 42,
 'clf__splitter': 'random',
 'feature_select__k': 8}'''
 
###Predicting the values and checking 


labels_pred = gs.predict(features_test)

print 'Precision -------->',(precision_score(labels_test,labels_pred))
print 'Recall----------> ',(recall_score(labels_test,labels_pred))
accuracy_score(labels_test, labels_pred)
confusion_matrix(labels_test, labels_pred)

''' The precision is 0.1667 and the recall is 0.3333. The precision value is 
not very good whereas the recall value is the minimum that we require'''

#-------------------------------------------------------------------------------                 
### creating a model based on naive_bayes

dataset = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(dataset)  

labels_train, labels_test, features_train, features_test = train_test_split(labels, features, test_size = 0.25, random_state = 42) 
 
from sklearn.naive_bayes import GaussianNB

selector = SelectKBest(f_classif)
clf = GaussianNB()

pipe = Pipeline([('feature_select', selector), ('clf', clf)])

parameters = dict(feature_select__k = range(1, 7))



grid_search = GridSearchCV(estimator = pipe, param_grid = parameters,
                           cv = 3, n_jobs = -1)

gs = grid_search.fit(features_train, labels_train)

gs.best_params_

''' The optimum number of features we should select is 8'''

labels_pred = gs.predict(features_test)

print 'Precision -------->',(precision_score(labels_test,labels_pred))
print 'Recall----------> ',(recall_score(labels_test,labels_pred))
print 'Accuracy----------> ', accuracy_score(labels_test, labels_pred)

'''We get a very good precision score of 0.5 (considering the size of the 
dataset) and the minimum value of Recall that we require. The Naive Bayes 
method is a good algorithm to employ for this dataset.'''

#-----------------------------------------------------------------------------
#Using stratified shuffle split and NaiveBayes algorithm with best features


clf = GaussianNB()

features_list = ['poi', 'salary','other', 'bonus', 'restricted_stock_ratio', 'shared_receipt_with_poi',
                      'exercised_stock_ratio', 'expenses', 'deferred_income',
                      'long_term_incentive', 'deferral_payments',
                      'fraction_sent_to_poi', 'fraction_from_poi']

dataset = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(dataset)

labels_train, labels_test, features_train, features_test = train_test_split(labels, features, test_size = 0.25, random_state = 42) 

selector = SelectKBest(f_classif, k = 'all')
selector.fit_transform(features_train, labels_train) 
p_values = selector.pvalues_  
scores = selector.scores_

selected_features = ['poi']  

for i in selector.get_support(indices = True):
    print features_list[i+1], '------->', scores[i]
    if p_values[i] < 0.1:
        selected_features.append(features_list[i+1])
        
from sklearn.cross_validation import StratifiedShuffleSplit
        
cv = StratifiedShuffleSplit(labels, 1000, random_state = 42)
for train_idx, test_idx in cv: 
    features_train = []
    features_test  = []
    labels_train   = []
    labels_test    = []
    for ii in train_idx:
        features_train.append( features[ii] )
        labels_train.append( labels[ii] )
    for jj in test_idx:
        features_test.append( features[jj] )
        labels_test.append( labels[jj] )
   
clf.fit(features_train, labels_train)
labels_pred = clf.predict(features_test)


features_list = selected_features

pickle.dump(clf, open("my_classifier.pkl", "w") )
pickle.dump(my_dataset, open("my_dataset.pkl", "w") )
pickle.dump(features_list, open("my_feature_list.pkl", "w") )

dump_classifier_and_data(clf, my_dataset, features_list)

                 
'''Conclusion


The main issue with this dataset is that there arent many values of positive 
pois to create a good model on. A bigger training and test set would give us a 
lot of points to train the algorithm with and allow us to increase our 
precision and recall score.

The best model to tackle this question was to find the best parameters, 
use stratifiedshufflesplit to get a good training and test set, and use Naive 
Bayes algorithm to identify the pois in the test set.'''

