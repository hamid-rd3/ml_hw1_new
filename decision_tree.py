import numpy as np
import pandas as pd 


def  get_entropy(df):
    
    #max entropy is log2 (c) where c is amount of features
    colnum = df.size/df.shape[0]
    size=df.shape[0]
    if colnum!=1:
        return sum([-x/size*np.log2(x/size) for x in df[df.columns[-1]].value_counts() ])
    elif colnum==1:
        # labels column
        return sum([-x/size*np.log2(x/size) for x in df.value_counts() ])
def get_gini(df):

    #max gini impurity is 1-1/c where c is amount of features

    colnum = df.size/df.shape[0]
    size = df.shape[0]
    if colnum != 1:
        return sum([x/size*(1-x/size) for x in df[df.columns[-1]].value_counts()])
    elif colnum == 1:
        # labels column
        return sum([x/size*(1-x/size) for x in df.value_counts()])
    

def getIG(df,feature,criterion='entropy'):
    
    size = df.shape[0]
    colnum = df.size/df.shape[0]
    if colnum != 1:
        value_count=df[feature].value_counts()
    else :
        value_count=df
    if criterion=='gini' :
       avg_info = sum([value_count[val] * get_gini(df[df[feature] == val])
                       for val in df[feature].unique()])
       return get_gini(df)-avg_info/size
    else : 
        avg_info=sum([value_count[val] * get_entropy(df[df[feature] == val])
            for val in df[feature].unique()])
        return get_entropy(df)-avg_info/size   

def getbestfeat(df,criterion='entropy'):
    colnum = df.size/df.shape[0]
    if colnum!=1 :
        return max(df.columns[:-1], key=lambda feature: getIG(df, feature,criterion))
    elif colnum==1 :
        return None 


class Node :
    def __init__(self ,attribute=None,branche_childs=None):
        self.attribute=attribute
        self.branche_childs=branche_childs
    def __repr__(self) :
        return f"attribute = {self.attribute} \n branches -> child : \n {self.branche_childs}\n "
    
class tree :
    #assume that last column is target feature
    def __init__(self,df=None,depth=1,criterion='entropy',root=None):
        self.df=df
        self.depth=depth
        self.critertion=criterion
        self.root=root
    def fit(self):
        if self.df.shape[1]==2 or self.depth==1:
            
            self.root = Node(max(self.df[self.df.columns[-1]].unique(
            ), key=lambda x: self.df[self.df.columns[-1]].value_counts()[x]))
        else :
            
            self.root=Node(getbestfeat(self.df),{})
            residual_columns = [
                k for k in self.df.columns if k != self.root.attribute]

            for val in self.df[self.root.attribute].unique():
                df_tmp=self.df[self.df[self.root.attribute]== val][residual_columns]
                tree_tmp=tree(df_tmp,self.depth-1,criterion=self.critertion)
                tree_tmp.fit()
                self.root.branche_childs[val]=tree_tmp.root.attribute
                 
    # for feature in df
    
#part a
df1 = pd.read_csv('nursery.csv')
df1.dropna()
train = df1.sample(frac=0.8, random_state=200)
test = df1.drop(train.index)

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
# x=list(df1.columns)
# x.remove('health')
# print(df1[x])
# print(df1[['final evaluation','health']])
# t = tree(df1[['form','social','health', 'final evaluation']], 3)
t = tree(df1, 7)
t.fit()
print(t.root)
