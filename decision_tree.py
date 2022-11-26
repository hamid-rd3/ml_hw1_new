import numpy as np
import pandas as pd 


def  get_entropy(df):
    
    colnum = df.size/df.shape[0]
    size=df.shape[0]
    if colnum!=1:
        return sum([-x/size*np.log2(x/size) for x in df[df.columns[-1]].value_counts() ])
    elif colnum==1:
        # labels column
        return sum([-x/size*np.log2(x/size) for x in df.value_counts() ])

def getIG(df,feature):
    
    size = df.shape[0]
    colnum = df.size/df.shape[0]
    if colnum != 1:
        value_count=df[feature].value_counts()
    else :
        value_count=df
    avg_info=sum([value_count[val] * get_entropy(df[df[feature] == val])
         for val in df[feature].unique()])
    return get_entropy(df)-avg_info/size   

def getbestfeat(df):
    IGfeatures=[]
    colnum = df.size/df.shape[0]
    if colnum!=1 :
        for feature in df.columns :
            IGfeatures.append(getIG(df,feature))
    elif colnum==1 :
        return None 
    return max(df.columns[:-1], key=lambda feature: getIG(df, feature))

    # for feature in df
#part a
df1 = pd.read_csv('nursery.csv')
df1.dropna()
train = df1.sample(frac=0.8, random_state=200)
test = df1.drop(train.index)

# print (train.shape,test.shape)
# print(df1[0:2].size/df1[0:2].shape[0])
# print(df1['final evaluation'].value_counts()['not_recom'])
# print([df1[df1['final evaluation']==x] for x in df1['final evaluation'].value_counts()])
# print(getIG(df1, 'final evaluation'), getIG(df1, 'health'))
print(getbestfeat(df1['final evaluation']))

