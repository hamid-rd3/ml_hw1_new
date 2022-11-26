import numpy as np
import pandas as pd


def get_entropy(df):

    # max entropy is log2 (c) where c is amount of features
    colnum = df.size/df.shape[0]
    size = df.shape[0]
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
        avg_info = sum([value_count[val] * get_gini(df[df[feature] == val])
                        for val in df[feature].unique()])
        return get_gini(df)-avg_info/size
    else:
        avg_info = sum([value_count[val] * get_entropy(df[df[feature] == val])
                        for val in df[feature].unique()])
        return get_entropy(df)-avg_info/size


def getbestfeat(df, criterion='entropy'):
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
        if self.df.shape[1] == 2 or self.depth == 1:
            labels = self.df.columns[-1]
            self.root = Node(max(self.df[
                labels].unique(), key=lambda x: self.df[labels].value_counts()[x]))
            return self.root
        else:

            self.root = Node(getbestfeat(self.df), {})
            residual_columns = [
                k for k in self.df.columns if k != self.root.attribute]

            for val in self.df[self.root.attribute].unique():
                df_tmp = self.df[self.df[self.root.attribute]
                                 == val][residual_columns]
                tree_tmp = tree(df_tmp, self.depth-1,
                                criterion=self.critertion)
                root_tmp = tree_tmp.fit()
                self.root.branch_childs[val] = root_tmp
        return self.root

    def predict(self, X_tr, node=None):
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
            self.y_predict.loc[X_tr.axes[0]] = node.attribute
        return self.y_predict

    def accuracy(self, ytest):
        return round(sum(self.y_predict == ytest)/self.y_predict.shape[0], 2)

    # for feature in df


# part a
df1 = pd.read_csv('nursery.csv')
df1.dropna()
train = df1.sample(frac=0.8, random_state=200)
test = df1.drop(train.index)
# y=df1['final evaluation']
# y[:]=None
# y.loc[df1['health'][[1,300, 600, 900,12958]].axes[0]] = 'asd'
# print(y)

# print(df1.loc[[300,600,900]])
# print(df1.head())
# print (train.shape,test.shape)
# print(df1[0:2].size/df1[0:2].shape[0])
# print(df1['final evaluation'].value_counts())
# print([df1[df1['final evaluation']==x] for x in df1['final evaluation'].value_counts()])
# print(getIG(df1, 'final evaluation'), getIG(df1, 'health'))
# print(getIG(df1, 'final evaluation', 'gini'), getIG(df1, 'health', 'gini'))

# print(getbestfeat(df1), getbestfeat(df1,'gini'))

# n=Node('health',{'x':Node('mamad',{'x':0}),'y':0,'z':'1'})

# print(Node('health'),)
# print(n)
# print(df1[df1['health'] == 'priority' and df1['children'] == 1])
# print(max(df1['final evaluation'].unique(),key=lambda x: df1['final evaluation'].value_counts()[x]))
# print(df1['final evaluation'].name)
# x.remove('health')
# print(df1[x])
# print(df1[['final evaluation','health']])
t = tree(df1[['form', 'social', 'health', 'final evaluation']], 3)

# t = tree(df1, 3)
t.fit()
# print(t.root)
# print(df1[df1.columns[-1]])
print(t.predict(test[test.columns[:-1]]))
print(t.accuracy(test[test.columns[-1]]))
# print(t.root.branch_childs['recommended'])
# print(df1[['form','social']][:500]['form'][400:])
