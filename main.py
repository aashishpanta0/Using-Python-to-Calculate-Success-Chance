import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
# %matplotlib inLine

with open("ks-projects-201801.csv",'r') as f:
    array=np.zeros((40000,6))
    s=pd.read_csv("ks-projects-201801.csv")
    array=s

    success=s.loc[s['state']=='successful']
    failed=s.loc[s['state']=='failed']

    enc = LabelEncoder()

    s_copy = s.copy()

    for column in ['main_category','goal', 'backers','country']:
        s_copy[column] = enc.fit_transform(s_copy[column])
        s_copy = s_copy[s_copy['state'].apply(lambda x: x in ['successful', 'failed'])]


    print("The total number of projects is",len(s))
    print("\nThe number of total successful projects is",len(success))

    print("The number of total failed projects is",len(failed))
    state_per=round(s["state"].value_counts()/len(s)*100,2)
    print("\nPercentage based on project's state: \n", state_per)

    # average_back=round(s["country"]['goal'].value_counts()/s['backers'].value_counts(),2)
    # print('\nPledges per backer is:', average_back)

    country_per=round(s["country"].value_counts()/len(s)*100,2)
    print("Percecntage based on the country: \n",country_per)

    def dataSplit(m, n):
        x = s_copy.iloc[:300000][m]
        y = s_copy.iloc[:300000][n]
        test_x = s_copy.iloc[300000:len(s_copy)][m]
        test_y = s_copy.iloc[300000:len(s_copy)][n]


        return x, y, test_x, test_y


    x, y, test_x, test_y = dataSplit([ 'main_category', 'goal', 'backers'], 'state')


    model = linear_model.LogisticRegression()
    model.fit(x, y)
    gnb = GaussianNB()



    def printCM(y,y_pred):
        truen, falsep, falsen, truep = confusion_matrix(y, y_pred).reshape(-1)

        print("\nOverall test accuracy: " ,(truep+truen)/float((truep+truen+falsep+falsen)))
        print("Precision (truep/truep+falsep): " + str(truep/float((truep+falsep))))


    print('\nUsing naive Bayes:')
    printCM(test_y, gnb.fit(test_x, test_y).predict(test_x))



    print('\nUsing Logistic Regression: ')
    printCM(test_y,model.predict(test_x))



    clf = tree.DecisionTreeClassifier(max_depth=3)
    print('\nUsing Decision Tree: ')
    printCM(test_y,clf.fit(test_x,test_y).predict(test_x))




    s['main_category'].value_counts().plot(kind='bar', color ='b')
    plt.title('Main Categories and No. of projects')
    plt.ylabel('No. of Projects')
    plt.xlabel('Category')
    plt.show(block=True)




    ratio = s.pivot_table(index='main_category', columns='state', values='ID', aggfunc='count')
    ratio['Success to fail ratio'] = ratio['successful'] / ratio['failed']
    ratio['Success to fail ratio'].sort_values(ascending=False).plot(kind='bar', color='r')
    plt.title('Success to Failure Ratio on Kickstarter')
    plt.xlabel('Category')
    plt.ylabel('Ratio')
    plt.show(block=True)

    # s.groupby('status').duration.mean().sort_index().plot(kind='box', color='y')
    # plt.title('Average Days')
    # plt.show(block=True)



    # backerratio=s.pivot_table(index='main_category', columns='backers', values='ID', aggfunc='count')
    # backerratio['Pledged to Backer ratio']= ratio['backers']/len(s)
    # backerratio['Pledged to Backer ratio'].sort_values(ascending = False).plot(kind='bar',color='r')
    # plt.title('Pledged to Backer ratio on Kickstarter')
    # plt.xlabel('Category')
    # plt.ylabel('Ratio')
    # plt.show(block=True)
    # s(pledged.tot, aes('main_category'', total/1000000, fill=total)) + geom_bar(stat="identity") +
    # ggtitle("Total Amount Pledged by Category") + xlab("Project Category") +
    # ylab("Amount Pledged (USD millions)") +
    # geom_text(aes(label=paste0("$", round(total/1000000,1))), vjust=-0.5) + theme_economist() +
    # theme(plot.title=element_text(hjust=0.5), axis.title=element_text(size=12, face="bold"))



