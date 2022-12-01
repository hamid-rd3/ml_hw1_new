import numpy as np
import pandas as pd
import time 

def get_entropy(df):

    # max entropy is log2 (c) where c is amount of features
    colnum = df.size/df.shape[0]
    size = df.shape[0]
    #could not to pick up last column of an one column dataframe  ...
    # -> df[df.columns[-1]] vs df
    if colnum != 1:
        return sum([-x/size*np.log2(x/size) for x in df[df.columns[-1]].value_counts()])
    elif colnum == 1:
        # labels column
        return sum([-x/size*np.log2(x/size) for x in df.value_counts()])


def get_gini(df):

    # max gini impurity is 1-1/c where c is amount of features

    colnum = df.size/df.shape[0]
    size = df.shape[0]
    if colnum != 1:
        return sum([x/size*(1-x/size) for x in df[df.columns[-1]].value_counts()])
    elif colnum == 1:
        # labels column
        return sum([x/size*(1-x/size) for x in df.value_counts()])


def getIG(df, feature, criterion='entropy'):

    size = df.shape[0]
    colnum = df.size/df.shape[0]
    if colnum != 1:
        value_count = df[feature].value_counts()
    else:
        value_count = df
    if criterion == 'gini':
        #select branch dataframe and calculate average information
        avg_info = sum([value_count[val] * get_gini(df[df[feature] == val])
                        for val in df[feature].unique()])
        return get_gini(df)-avg_info/size
    else:
        avg_info = sum([value_count[val] * get_entropy(df[df[feature] == val])
                        for val in df[feature].unique()])
        return get_entropy(df)-avg_info/size


def getbestfeat(df, criterion='entropy'):
    '''return best feature as the biggest information gain '''
    colnum = df.size/df.shape[0]
    if colnum != 1:
        return max(df.columns[:-1], key=lambda feature: getIG(df, feature, criterion))
    elif colnum == 1:
        return df.name


class Node:
    def __init__(self, attribute=None, branch_childs=None):
        self.attribute = attribute
        self.branch_childs = branch_childs

    def __repr__(self):
        if self.branch_childs:
            return f"attribute = {self.attribute} \n [branch -> childs] : \n {self.branch_childs}\n "
        else:
            return f"attribute = {self.attribute}"


class tree:
    # assume that last column is target feature
    def __init__(self, df=None, depth=1, criterion='entropy', root=None):
        self.df = df
        self.depth = depth
        self.critertion = criterion
        self.root = root
        self.y_predict = None

    def fit(self):
        ''' it makes decision tree based on ID3 algorithm '''
        if self.df.shape[1] == 2 or self.depth == 1:
            #leaf Node or reachs maximum depth 
            labels = self.df.columns[-1]
            #label data = max lable value 
            self.root = Node(max(self.df[
                labels].unique(), key=lambda x: self.df[labels].value_counts()[x]))
            return self.root
        else:

            self.root = Node(getbestfeat(self.df), {})
            residual_columns = [
                k for k in self.df.columns if k != self.root.attribute]

            #Iterate through brnaches of the root 
            for val in self.df[self.root.attribute].unique():
                df_tmp = self.df[self.df[self.root.attribute]
                                 == val][residual_columns]
                #creates temprory tree to be fitted and take the ...
                #attribute of child node based on information gain
                tree_tmp = tree(df_tmp, self.depth-1,
                                criterion=self.critertion)
                root_tmp = tree_tmp.fit()
                self.root.branch_childs[val] = root_tmp
        return self.root

    def predict(self, X_tr, node=None):
        '''it predicts X test based on the fitted tree and return predicted y '''
        if node is None:
            node = self.root
        if self.y_predict is None:

            self.y_predict = X_tr[X_tr.columns[-1]].copy()
            # no matter which column of X_tr is selected
            # it just needs the indexes!
            self.y_predict[:] = None
            self.y_predict.name = self.df.columns[-1]
        if node.branch_childs:

            for b, c in node.branch_childs.items():
                self.predict(X_tr[X_tr[node.attribute] == b], c)
        else:
            #create y_predict rows from leaf attibute 
            self.y_predict.loc[X_tr.axes[0]] = node.attribute
        return self.y_predict

    def accuracy(self, ytest):
        '''compate ytest and ypredict'''
        return round(sum(self.y_predict == ytest)/self.y_predict.shape[0], 2)

