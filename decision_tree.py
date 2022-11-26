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
    value_count=df[feature].value_counts()
    avg_info=sum([value_count[val] * get_entropy(df[df[feature] == val])
         for val in df[feature].unique()])
    return get_entropy(df)-avg_info/size   

#part a
df = pd.read_csv('nursery.csv')
df.dropna()
train = df.sample(frac=0.8, random_state=200)
test = df.drop(train.index)

# print (train.shape,test.shape)
# print(df[0:2].size/df[0:2].shape[0])
# print(df['final evaluation'].value_counts()['not_recom'])
# print([df[df['final evaluation']==x] for x in df['final evaluation'].value_counts()])
# print(getIG(df, 'final evaluation'), getIG(df, 'health'))
