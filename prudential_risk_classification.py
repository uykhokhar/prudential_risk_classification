
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve, learning_curve
from sklearn.linear_model import LogisticRegression LogisticRegressionCV


# ## 4) qwk fn

# In[2]:

def qwk(y1, y2):
    #get number of unique items in y1, y2, -> k
    comb = np.concatenate((y1, y2))
    val = np.unique(comb)
    k = val[-1] + 1
    
    #make 3 k x k matrices
    s = (k, k)
    O = np.zeros(s)
    E = np.zeros(s)
    W = np.zeros(s)
    iv = np.zeros(k)
    jv = np.zeros(k)
    n = 0

    #iterate through nparray and increment the value of the index by 1
    for i in range(len(y1)):
        n += 1
        iv[y1[i]] += 1
        jv[y2[i]] += 1

    #calculate O array 
    for i in range(len(y1)):
        O[y1[i]][y2[i]] += 1
    O = O / n
    print("O Matrix:\n {}".format(O))

    #generate W matrix 
    for i in range(k):
        for j in range(k):
            W[i][j] = (i-j)**2
    print("W Matrix:\n {}".format(W))

    #generate E matrix
    for i in range(k):
        for j in range(k):
            E[i][j] = (iv[i] * jv[j])/n**2
    print("E Matrix:\n {}".format(E))
    
    kappa = 1 - np.sum(O * W)/np.sum(E * W)
    print(O * W)
    print(E * W)
    return(kappa)


# In[3]:

#Test cases

y1 = [2, 1, 0, 0, 1, 0]
y2 = [2, 1, 0, 1, 1, 1]
print("\n****qwk: {}".format(qwk(y1, y2)))
print("****SciKit: {}".format(cohen_kappa_score(y1, y2, weights="quadratic")))


# ## 5) 

# The median is a more robust estimator for data with high magnitude variables which could dominate results (otherwise known as a ‘long tail’).
# http://scikit-learn.org/stable/auto_examples/missing_values.html#sphx-glr-auto-examples-missing-values-py
# 
# http://scikit-learn.org/stable/modules/preprocessing.html#imputation-of-missing-values
# 
# https://www.kaggle.com/c/prudential-life-insurance-assessment/data
# 

# In[4]:

data = pd.read_csv("train.csv")


# In[5]:

frac = 0.005
#df = data.sample(frac = frac, replace = False)
df = data


# In[6]:

kappa_scorer = make_scorer(cohen_kappa_score, weights='quadratic')


# In[7]:

#impute missing continuous -> mean values
imp1 = Imputer(missing_values='NaN', strategy='mean', axis=0)
cols1 = ['Product_Info_4', 'Ins_Age', 'Ht', 'Wt', 'BMI', 'Employment_Info_1', 'Employment_Info_4', 'Employment_Info_6', 'Insurance_History_5', 'Family_Hist_2', 'Family_Hist_3', 'Family_Hist_4', 'Family_Hist_5'] 
df[cols1] = imp1.fit_transform(df[cols1])


# In[8]:

#impute missing discrete -> most frequent 
imp2 = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
cols2 = ['Medical_History_1', 'Medical_History_10', 'Medical_History_15', 'Medical_History_24', 'Medical_History_32']
df[cols2] = imp2.fit_transform(df[cols2])


# In[9]:

#transform categorical features
cols3 = ['Product_Info_1', 'Product_Info_2', 'Product_Info_3', 'Product_Info_5', 'Product_Info_6', 'Product_Info_7', 'Employment_Info_2', 'Employment_Info_3', 'Employment_Info_5', 'InsuredInfo_1', 'InsuredInfo_2', 'InsuredInfo_3', 'InsuredInfo_4', 'InsuredInfo_5', 'InsuredInfo_6', 'InsuredInfo_7', 'Insurance_History_1', 'Insurance_History_2', 'Insurance_History_3', 'Insurance_History_4', 'Insurance_History_7', 'Insurance_History_8', 'Insurance_History_9', 'Family_Hist_1', 'Medical_History_2', 'Medical_History_3', 'Medical_History_4', 'Medical_History_5', 'Medical_History_6', 'Medical_History_7', 'Medical_History_8', 'Medical_History_9', 'Medical_History_11', 'Medical_History_12', 'Medical_History_13', 'Medical_History_14', 'Medical_History_16', 'Medical_History_17', 'Medical_History_18', 'Medical_History_19', 'Medical_History_20', 'Medical_History_21', 'Medical_History_22', 'Medical_History_23', 'Medical_History_25', 'Medical_History_26', 'Medical_History_27', 'Medical_History_28', 'Medical_History_29', 'Medical_History_30', 'Medical_History_31', 'Medical_History_33', 'Medical_History_34', 'Medical_History_35', 'Medical_History_36', 'Medical_History_37', 'Medical_History_38', 'Medical_History_39', 'Medical_History_40', 'Medical_History_41']


# In[10]:

le = LabelEncoder()
for col in cols3:
    le.fit(df[col])
    df[col] = le.transform(df[col])


# In[11]:

enc = OneHotEncoder(sparse = False)

new_data = df
for col in cols3:
    data = df[[col]]
    enc.fit(data)
    temp = enc.transform(data)
    temp = pd.DataFrame(temp, columns = [col+'_'+str(i) for i in data[col].value_counts().index])
    new_data = pd.concat([new_data, temp], axis = 1)
new_data = new_data.drop(cols3, axis = 1)


# #get dummies for Product Info 2
# #https://stackoverflow.com/questions/11587782/creating-dummy-variables-in-pandas-for-python
# just_dummies = pd.get_dummies(df['Product_Info_2'], drop_first=True)
# 
# df = pd.concat([df, just_dummies], axis=1)      
# df.drop(['Product_Info_2'], inplace=True, axis=1)

# In[12]:

y = df['Response']
X = df.drop(['Response'], axis = 1)


# In[13]:

# only three fold cross validation possible because 
depth_range = range(2, 20)
train_scores, test_scores = validation_curve(tree.DecisionTreeClassifier(), X, y, 'max_depth', depth_range, cv=5, scoring = kappa_scorer)
train_scores, test_scores


# In[14]:

param_range = range(2, 20)

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.title("Validation Curve with Decision Tree")
plt.xlabel("Depth")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
lw = 2
plt.plot(depth_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)

plt.plot(depth_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
plt.legend(loc="best")
plt.show()


# In[15]:

#Max depth is 10th level
print(test_scores_mean)
print(test_scores_mean.argmax() + 2)


# In[16]:

train_sizes2, train_scores2, test_scores2 = learning_curve(tree.DecisionTreeClassifier(max_depth = 11), X, y, train_sizes= range(1000, 48000, 1000), cv=5, scoring = kappa_scorer)


# In[17]:

train_sizes2


# In[18]:

param_range = train_sizes2
train_scores_mean2 = np.mean(train_scores2, axis=1)
test_scores_mean2 = np.mean(test_scores2, axis=1)

plt.title("Learning Curve with Decision Tree")
plt.xlabel("Training Set Size")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
lw = 2
plt.plot(param_range, train_scores_mean2, label="Training score",
             color="darkorange", lw=lw)

plt.plot(param_range, test_scores_mean2, label="Cross-validation score",
             color="navy", lw=lw)
plt.legend(loc="best")
plt.show()


# In[ ]:




# In[32]:

Cinput = [1e-4, 4e-4, 8e-4, 1.2e-3, 1.6e-3, 2e-3]
train_scores3, test_scores3 = validation_curve(LogisticRegression(), X, y, 'C', Cinput, cv=5, scoring = kappa_scorer)
train_scores3, test_scores3


# In[33]:

param_range = Cinput

train_scores_mean3 = np.mean(train_scores3, axis=1)
test_scores_mean3 = np.mean(test_scores3, axis=1)

plt.title("Validation Curve with Logistic Regression")
plt.xlabel("C")
plt.ylabel("Score")
plt.ylim(0, 0.35)
lw = 2
plt.plot(param_range, train_scores_mean3, label="Training score",
             color="darkorange", lw=lw)

plt.plot(param_range, test_scores_mean3, label="Cross-validation score",
             color="navy", lw=lw)
plt.legend(loc="best")
plt.show()


# In[35]:

train_sizes4, train_scores4, test_scores4 = learning_curve(LogisticRegression(C = 1.2e-3), X, y, train_sizes= range(1000, 48000, 1000), cv=5, scoring = kappa_scorer)


# In[38]:

param_range = train_sizes4
train_scores_mean4 = np.mean(train_scores4, axis=1)
test_scores_mean4 = np.mean(test_scores4, axis=1)

plt.title("Learning Curve with Logistic Regression")
plt.xlabel("Training Set Size")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
lw = 2
plt.plot(param_range, train_scores_mean4, label="Training score",
             color="darkorange", lw=lw)

plt.plot(param_range, test_scores_mean4, label="Cross-validation score",
             color="navy", lw=lw)
plt.legend(loc="best")
plt.show()


# In[ ]:



