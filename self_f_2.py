#!/usr/bin/env python
# coding: utf-8

# In[4]:


#!/usr/bin/env python
# coding: utf-8


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import math 
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier,_tree
from IPython.display import display
import warnings
warnings.filterwarnings('ignore')


#安装决策树相关的包，
# conda install graphviz
# conda install pydotplus

def character_VarFindFunc(df):
    """
    查找非数值型特征
    param:
        df -- 需要查询的数据框
    return：
        character_name -- 非数值型特征名称列表
    """
    count=0
    character_name=[]
    for i in df.columns:#返回的是pandas的索引<class 'pandas.core.indexes.base.Index'>，如果是df.columns.values返回的是<class 'numpy.ndarray'>
        if df[i].dtype == np.dtype('object') or df[i].dtype == np.dtype('bool'):#这样写也可str(df[i].dtype) == 'object': 不添加这个条件是因为当部分变量的值存在连续型数值时，后续使用LabelEncoder会存在这个报错y contains previously unseen labels，or df[i].nunique()<=5
            count=count+1
            character_name.append(i)
    if(count==0):
        print('no charchater variables exists')
    return character_name,count



def find_abnormal(df):
    """
    查找数值型特征的异常值，利用三分位数和一分位数的1.5倍来判断
    param:
        df -- 需要查询的数据框
    return：
        df_new -- 添加异常值标记的新数据框
    """
    count=0
    #list_abnormal=[]
    for i in df.columns:
        if df[i].dtype == np.dtype('object') or df[i].dtype == np.dtype('bool'):#这样写也可str(df[i].dtype) == 'object'
            break
        else:
            q3=df[i].quantile(q=0.75)
            q1=df[i].quantile(q=0.25)
            Q=q3-q1
            if any(df[i]<q1-Q)==True or any(df[i]>q3+Q)==True:
                count=count+1
                df[i+'_abnormal']='zhengchang'
                df[i+'_abnormal'][df[i]<q1-1.5*Q]='abnormal'
                df[i+'_abnormal'][df[i]>q3+1.5*Q]='abnormal'                         
    if(count==0):
        print('no abnormal variables exists')
    #list_abnormal[0]=count
    #list_abnormal[1]=df
    df_new=df
    return df_new


def findNaFunc(df,is_show_all=0):
    """
  缺失值统计
  param:
      df:需要计算缺失值到数据框
      is_show_all -- 不缺失变量是否展示，默认为0，不展示，-1表示展示
  return:
      df_new:返回df中有缺失值到变量缺失个数和缺失率
    """   
    count=0
    df_new=pd.DataFrame()
    df_new["queshi_num"]=df.isnull().sum()
    df_new["na_rate"]=round(df_new["queshi_num"]/len(df),4)
    df_new["na_rate_new"]=df_new["na_rate"].apply(lambda x: format(x, '.2%'))
    df_new=df_new[df_new["queshi_num"]>is_show_all]
    df_new=df_new.sort_values(by='queshi_num', axis=0, ascending=False)
    df_new=df_new.reset_index().rename(columns={"index":"var_name"})
    count=df_new.shape[0]
# 若无缺失，则显示没有缺失变量
    if(count==0):
        print("no variable contain NA!")
    else:
        return df_new,count

    
def na_fill(df,continuous=-999,discrete='缺失'):
    """
    对离散型变量和连续型变量，缺失值默认填充不一样的缺失值，主要由于使用labelencoder编码时，
    如果离散型变量使用数值进行填充，填充的数值，再编码映射时会转换成字符型，导致在编码映射时，找不到填充数值对应的编码
    param:
        df -- 需要进行缺失填补的数据框
        continuous -- 连续型变量默认填充的数值，默认-999
        discrete -- 离散型变量默认填充的数值，默认“缺失”
    return:
        df_new -- 填充后的数据框
    """
    character_name,cate_count=character_VarFindFunc(df)
    df_new=df.copy()
    
    for i in character_name:
        df_new[i].fillna(value=discrete,inplace=True)
        
    df_new.fillna(value=continuous,inplace=True)
            
    return df_new



def feature_labelencoder(df):
    """
    对离散型变量进行labelencoder编码
    param:
        df -- 需要进行离散数据编码的数据框
    return:
        df_new -- 针对离散型变量labelencoder后的数据框
    """
    le=preprocessing.LabelEncoder()
    character_name,cate_count=character_VarFindFunc(df)
    df_new=df.copy()
#     character_name.remove(target)
    for i in character_name:
        code_list=list(df_new[i].unique())
        encoder= le.fit(code_list)
        df_new[i] = encoder.transform(df_new[i])
            
    return df_new



def Random_feature_filter(df,target='target',im_value=0.01,n_estimators=200,random_state=1,min_samples_leaf=50):
    """
    利用随机森林进行变量筛选，并筛选重要度大于一定阈值度变量
    param:
        df -- 需要进行变量筛选度数据框
        target -- 对应度y字段，默认为target
        im_value -- 筛选重要度度大于多少阈值的变量，默认为0.01
        n_estimators -- 随机森林参数，训练多少棵树
        random_state -- 随机森林参数，随机树取多少
        min_samples_leaf -- 随机森林参数，叶子节点最少的数量
    return:
        importance_list_final -- 返回筛选里重要度的变量以及变量重要度的数据框
    """
    y = df[target]
    x = df.drop(target, axis=1)
    forest = RandomForestClassifier(n_estimators=n_estimators,n_jobs=-1,random_state=random_state,min_samples_leaf = min_samples_leaf)
    forest.fit(x, df[target])
    importance = forest.feature_importances_
    indices = np.argsort(importance)[::-1]
    features = x.columns
    importance_list=pd.DataFrame()
    for f in range(x.shape[1]):
        d1=pd.DataFrame([[features[f],importance[indices[f]]]],columns=["var_name","importance"])
        importance_list=importance_list.append(d1,ignore_index=True)
        
    importance_list_final=importance_list[importance_list['importance']>=im_value]
    importance_list_delete=importance_list[importance_list['importance']<im_value]
    return importance_list_final,importance_list_delete



def tree_split(df,col,target,max_bin,min_binpct,nan_value):
    """
    决策树分箱
    param:
        df -- 数据集 Dataframe
        col -- 分箱的字段名 string
        target -- 标签的字段名 string
        max_bin -- 最大分箱数 int
        min_binpct -- 箱体的最小占比 float
        nan_value -- 缺失的映射值 int/float
    return:
        split_list -- 分割点 list
    """
    miss_value_rate = df[df[col]==nan_value].shape[0]/df.shape[0]
    # 如果缺失占比小于5%，则直接对特征进行分箱
    if miss_value_rate<0.05:
        x = np.array(df[col]).reshape(-1,1)
        y = np.array(df[target])
        tree = DecisionTreeClassifier(max_leaf_nodes=max_bin,
                                  min_samples_leaf = min_binpct)
        tree.fit(x,y)
        thresholds = tree.tree_.threshold
        thresholds = thresholds[thresholds!=_tree.TREE_UNDEFINED]
        split_list = sorted(thresholds.tolist())
    # 如果缺失占比大于5%，则把缺失单独分为一箱，剩余部分再进行决策树分箱
    else:
        max_bin2 = max_bin-1
        x = np.array(df[~(df[col]==nan_value)][col]).reshape(-1,1)
        y = np.array(df[~(df[col]==nan_value)][target])
        tree = DecisionTreeClassifier(max_leaf_nodes=max_bin2,
                                  min_samples_leaf = min_binpct)
        tree.fit(x,y)
        thresholds = tree.tree_.threshold
        thresholds = thresholds[thresholds!=_tree.TREE_UNDEFINED]
        split_list = sorted(thresholds.tolist())
        split_list.insert(0,nan_value)
    
    return split_list



def status(x) : 
    """
    针对数值型特征进行各类描述性统计，使用时结合df.apply(status来使用)
    """
    return pd.Series([x.count(),x.min(),x.idxmin(),x.quantile(.25),x.median(),
                      x.quantile(.75),x.mean(),x.max(),x.idxmax(),x.mad(),x.var(),
                      x.std(),x.skew(),x.kurt()],index=['总数','最小值','最小值位置','25%分位数',
                    '中位数','75%分位数','均值','最大值','最大值位数','平均绝对偏差','方差','标准差','偏度','峰度'])


def quantile_split(df,col,target,max_bin,nan_value):
    """
    等频分箱
    param:
        df -- 数据集 Dataframe
        col -- 分箱的字段名 string
        target -- 标签的字段名 string
        max_bin -- 最大分箱数 int
        nan_value -- 缺失的映射值 int/float
    return:
        split_list -- 分割点 list
    """
    miss_value_rate = df[df[col]==nan_value].shape[0]/df.shape[0]
    
    # 如果缺失占比小于5%，则直接对特征进行分箱
    if miss_value_rate<0.05:
        bin_series,bin_cut = pd.qcut(df[col],q=max_bin,duplicates='drop',retbins=True)
        split_list = bin_cut.tolist()
        split_list.remove(split_list[0])
    # 如果缺失占比大于5%，则把缺失单独分为一箱，剩余部分再进行等频分箱
    else:
        df2 = df[~(df[col]==nan_value)]
        max_bin2 = max_bin-1
        bin_series,bin_cut = pd.qcut(df2[col],q=max_bin2,duplicates='drop',retbins=True)
        split_list = bin_cut.tolist()
        split_list[0] = nan_value
        
    split_list.remove(split_list[-1])
    
    # 当出现某个箱体只有好用户或只有坏用户时，进行前向合并箱体
    var_arr = np.array(df[col])
    target_arr = np.array(df[target])
    bin_trans = np.digitize(var_arr,split_list,right=True)
    var_tuple = [(x,y) for x,y in zip(bin_trans,target_arr)]
    
    delete_cut_list = []
    for i in set(bin_trans):
        target_list = [y for x,y in var_tuple if x==i]
        if target_list.count(1)==0 or target_list.count(0)==0:
            if i ==min(bin_trans):
                index=i
            else:
                index = i-1
            delete_cut_list.append(split_list[index])
    split_list = [x for x in split_list if x not in delete_cut_list]
    
    return split_list




def cal_woe(df,col,target,nan_value,cut=None):
    """
    计算woe
    param：
        df -- 数据集 Dataframe
        col -- 分箱的字段名 string
        target -- 标签的字段名 string
        nan_value -- 缺失的映射值 int/float
        cut -- 箱体分割点 list
    return:
        woe_list -- 每个箱体的woe list
    """
    total = df[target].count()
    bad = df[target].sum()
    good = total-bad
    
    bucket = pd.cut(df[col],cut)
    group = df.groupby(bucket)
        
    bin_df = pd.DataFrame()
    bin_df['total'] = group[target].count()
    bin_df['bad'] = group[target].sum()
    bin_df['good'] = bin_df['total'] - bin_df['bad']
    bin_df['badattr'] = bin_df['bad']/bad
    bin_df['goodattr'] = bin_df['good']/good
    bin_df['woe'] = np.log(bin_df['badattr']/bin_df['goodattr'])
    # 当cut里有缺失映射值时，说明是把缺失单独分为一箱的，后续在进行调成单调分箱时
    # 不考虑缺失的箱，故将缺失映射值剔除
    if nan_value in cut:
        woe_list = bin_df['woe'].tolist()[1:]
    else:
        woe_list = bin_df['woe'].tolist()
    return woe_list




def monot_trim(df,col,target,nan_value,cut=None):
    """
    woe调成单调递减或单调递增
    param:
        df -- 数据集 Dataframe
        col -- 分箱的字段名 string
        target -- 标签的字段名 string
        nan_value -- 缺失的映射值 int/float
        cut -- 箱体分割点 list
    return:
        new_cut -- 调整后的分割点 list
    """
    woe_lst = cal_woe(df,col,target,nan_value,cut = cut)
    count1=0
    count2=0
    #new_cut=[]
    # 若第一个箱体大于0，说明特征整体上服从单调递减
    if woe_lst[0]>0:
        while not judge_decreasing(woe_lst) and count1<=10:
            # 找出哪几个箱不服从单调递减的趋势
            judge_list = [x>y for x, y in zip(woe_lst, woe_lst[1:])]
            # 用前向合并箱体的方式，找出需要剔除的分割点的索引，如果有缺失映射值，则索引+1
            if nan_value in cut:
                index_list = [i+2 for i,j in enumerate(judge_list) if j==False]
            else:
                index_list = [i+1 for i,j in enumerate(judge_list) if j==False]
            new_cut=[j for i,j in enumerate(cut) if i not in index_list]
            woe_lst = cal_woe(df,col,target,nan_value,cut = new_cut)
            count1=count1+1
            woe_lst_new=woe_lst.copy()
    # 若第一个箱体小于0，说明特征整体上服从单调递增
    elif woe_lst[0]<0:
        while not judge_increasing(woe_lst) and count2<=10:
            # 找出哪几个箱不服从单调递增的趋势
            judge_list = [x<y for x, y in zip(woe_lst, woe_lst[1:])]
            # 用前向合并箱体的方式，找出需要剔除的分割点的索引，如果有缺失映射值，则索引+1
            if nan_value in cut:
                index_list = [i+2 for i,j in enumerate(judge_list) if j==False]
            else:
                index_list = [i+1 for i,j in enumerate(judge_list) if j==False]
            new_cut=[j for i,j in enumerate(cut) if i not in index_list]
            woe_lst = cal_woe(df,col,target,nan_value,cut = new_cut)
            count2=count2+1
            woe_lst_new=woe_lst.copy()
    return new_cut,woe_lst_new




def judge_increasing(L):
    """
    判断一个list是否单调递增
    """
    return all(x<y for x, y in zip(L, L[1:]))

def judge_decreasing(L):
    """
    判断一个list是否单调递减
    """
    return all(x>y for x, y in zip(L, L[1:]))




def binning_var(df,col,target,bin_type='dt',max_bin=5,min_binpct=0.05,nan_value=-999):
    """
    特征分箱，计算iv
    param:
        df -- 数据集 Dataframe
        col -- 分箱的字段名 string
        target -- 标签的字段名 string
        bin_type -- 分箱方式 默认是'dt',还有'quantile'(等频分箱)
        max_bin -- 最大分箱数 int,默认为5
        min_binpct -- 箱体的最小占比 float,默认为0.05
        nan_value -- 缺失映射值 int/float 默认为-999
    return:
        bin_df -- 特征的分箱明细表 Dataframe
        cut -- 分割点 list
    """
    total = df[target].count()
    bad = df[target].sum()
    good = total-bad
    
    # 离散型特征分箱,直接根据类别进行groupby
    if df[col].dtype == np.dtype('object') or df[col].dtype == np.dtype('bool') or df[col].nunique()<=max_bin:
        group = df.groupby([col],as_index=True)
        bin_df = pd.DataFrame()

        bin_df['total'] = group[target].count()
        bin_df['totalrate'] = bin_df['total']/total
        bin_df['bad'] = group[target].sum()
        bin_df['badrate'] = bin_df['bad']/bin_df['total']
        bin_df['good'] = bin_df['total'] - bin_df['bad']
        bin_df['goodrate'] = bin_df['good']/bin_df['total']
        bin_df['badattr'] = bin_df['bad']/bad
        bin_df['goodattr'] = (bin_df['total']-bin_df['bad'])/good
        bin_df['woe'] = np.log(bin_df['badattr']/bin_df['goodattr'])
        bin_df['bin_iv'] = (bin_df['badattr']-bin_df['goodattr'])*bin_df['woe']
        bin_df['IV'] = bin_df['bin_iv'].sum()
        cut = df[col].unique().tolist()
    # 连续型特征的分箱
    else:
        if bin_type=='dt':
            cut = tree_split(df,col,target,max_bin=max_bin,min_binpct=min_binpct,nan_value=nan_value)
        elif bin_type=='quantile':
            cut = quantile_split(df,col,target,max_bin=max_bin,nan_value=nan_value)
        cut.insert(0,float('-inf'))
        cut.append(float('inf'))
        
        bucket = pd.cut(df[col],cut)
        group = df.groupby(bucket)
        bin_df = pd.DataFrame()

        bin_df['total'] = group[target].count()
        bin_df['totalrate'] = bin_df['total']/total
        bin_df['bad'] = group[target].sum()
        bin_df['badrate'] = bin_df['bad']/bin_df['total']
        bin_df['good'] = bin_df['total'] - bin_df['bad']
        bin_df['goodrate'] = bin_df['good']/bin_df['total']
        bin_df['badattr'] = bin_df['bad']/bad
        bin_df['goodattr'] = (bin_df['total']-bin_df['bad'])/good
        bin_df['woe'] = np.log(bin_df['badattr']/bin_df['goodattr'])
        bin_df['bin_iv'] = (bin_df['badattr']-bin_df['goodattr'])*bin_df['woe']
        bin_df['IV'] = bin_df['bin_iv'].sum()
        
    return bin_df,cut





def binning_trim(df,col,target,cut=None,right_border=True):
    """
    调整单调后的分箱，计算IV
    param:
        df -- 数据集 Dataframe
        col -- 分箱的字段名 string
        target -- 标签的字段名 string
        cut -- 分割点 list
        right_border -- 箱体的右边界是否闭合 bool
    return:
        bin_df -- 特征的分箱明细表 Dataframe
    """
    total = df[target].count()
    bad = df[target].sum()
    good = total - bad
    bucket = pd.cut(df[col],cut,right=right_border)
    
    group = df.groupby(bucket)
    bin_df = pd.DataFrame()
    bin_df['total'] = group[target].count()
    bin_df['totalrate'] = bin_df['total']/total
    bin_df['bad'] = group[target].sum()
    bin_df['badrate'] = bin_df['bad']/bin_df['total']
    bin_df['good'] = bin_df['total'] - bin_df['bad']
    bin_df['goodrate'] = bin_df['good']/bin_df['total']
    bin_df['badattr'] = bin_df['bad']/bad
    bin_df['goodattr'] = (bin_df['total']-bin_df['bad'])/good
    bin_df['woe'] = np.log(bin_df['badattr']/bin_df['goodattr'])
    bin_df['bin_iv'] = (bin_df['badattr']-bin_df['goodattr'])*bin_df['woe']
    bin_df['IV'] = bin_df['bin_iv'].sum()
    
    return bin_df



def forward_corr_delete(df,col_list,cut_value=0.7):
    """
    相关性筛选，亮点是当某个变量因为相关性高，需要进行删除时，此变量不再和后续变量进行相关性计算，否则会把后续不应删除的变量由于和已经删除变量相关性高，而进行删除
    param:
        df -- 数据集 Dataframe
        col_list -- 需要筛选的特征集合,需要提前按IV值从大到小排序好 list
        corr_value -- 相关性阈值，高于此阈值的比那里按照iv高低进行删除，默认0.7
    return:
        select_corr_col -- 筛选后的特征集合 list
    """
    corr_list=[]
    corr_list.append(col_list[0])
    delete_col = []
    # 根据IV值的大小进行遍历
    for col in col_list[1:]:
        corr_list.append(col)
            #当多个变量存在相关性时，如果前述某个变量已经删除，则不应再和别的变量计算相关性
        if len(delete_col)>0:#判断是否有需要删除的变量
            for i in delete_col:
                if i in corr_list:#由于delete_col是一直累加的，已经删除的变量也会出现在delete_col里，因此需要判断变量当前是否还在corr_list里，否则remove会报错
                    corr_list.remove(i)
#         print(delete_col)
#         print(corr_list)
#         print('---------')
        corr = df.loc[:,corr_list].corr()
        corr_tup = [(x,y) for x,y in zip(corr[col].index,corr[col].values)]
        corr_value = [y for x,y in corr_tup if x!=col]
        # 若出现相关系数大于0.65，则将该特征剔除
        if len([x for x in corr_value if abs(x)>=cut_value])>0:
            delete_col.append(col)
#             print(delete_col)
    select_corr_col = [x for x in col_list if x not in delete_col]
    return select_corr_col,delete_col




def vif_delete(df,list_corr):
    """
    多重共线性筛选
    param:
        df -- 数据集 Dataframe
        list_corr -- 相关性筛选后的特征集合，按IV值从大到小排序 list
    return:
        col_list -- 筛选后的特征集合 list
    """
    col_list = list_corr.copy()
    delete_col=[]
    # 计算各个特征的方差膨胀因子
    vif_matrix=np.matrix(df[col_list])
    vifs_list=[variance_inflation_factor(vif_matrix,i) for i in range(vif_matrix.shape[1])]
    # 筛选出系数>10的特征
    vif_high = [x for x,y in zip(col_list,vifs_list) if y>10]
    
    # 根据IV从小到大的顺序进行遍历
    if len(vif_high)>0:
        for col in reversed(vif_high):
            col_list.remove(col)
            delete_col.append(col)
            vif_matrix=np.matrix(df[col_list])
            vifs=[variance_inflation_factor(vif_matrix,i) for i in range(vif_matrix.shape[1])]
            # 当系数矩阵里没有>10的特征时，循环停止
            if len([x for x in vifs if x>10])==0:
                break
    return col_list,delete_col




def forward_pvalue_delete(x,y,threshold=0.05):
    """
    显著性筛选，前向逐步回归
    param:
        x -- 特征数据集,woe转化后，且字段顺序按IV值从大到小排列 Dataframe
        y -- 标签列 Series 
        threshold -- 显著性筛选对应的p值，默认为0.05
    return:
        pvalues_col -- 筛选后的特征集合 list
    """
    col_list = x.columns.tolist()
    pvalues_col=[]
    pvalues_col_delete=[]
    # 按IV值逐个引入模型
    for col in col_list:
        pvalues_col.append(col)
        # 每引入一个特征就做一次显著性检验
        x_const = sm.add_constant(x.loc[:,pvalues_col])
        sm_lr = sm.Logit(y,x_const)
        sm_lr = sm_lr.fit()
        print(sm_lr.summary())
        print("\033[1;31m ---------------------------------------------------\033[0m")
        pvalue = sm_lr.pvalues[col]
        # 当引入的特征P值>=0.05时，则剔除，原先满足显著性检验的则保留，不再剔除
        if pvalue>=threshold:
            pvalues_col.remove(col)
            pvalues_col_delete.append(col)
    return pvalues_col,pvalues_col_delete



def backward_pvalue_delete(x,y):
    """
    显著性筛选，后向逐步回归
    param:
        x -- 特征数据集,woe转化后，且字段顺序按IV值从大到小排列 Dataframe
        y -- 标签列 Series
    return:
        pvalues_col -- 筛选后的特征集合 list
    """
    x_c = x.copy()
    # 所有特征引入模型，做显著性检验
    x_const = sm.add_constant(x_c)
    sm_lr = sm.Logit(y,x_const).fit()
    pvalue_tup = [(i,j) for i,j in zip(sm_lr.pvalues.index,sm_lr.pvalues.values)][1:]
    delete_count = len([i for i,j in pvalue_tup if j>=0.05])
    # 当有P值>=0.05的特征时，执行循环
    while delete_count>0:
        # 按IV值从小到大的顺序依次逐个剔除
        remove_col = [i for i,j in pvalue_tup if j>=0.05][-1]
        del x_c[remove_col]
        # 每次剔除特征后都要重新做显著性检验，直到入模的特征P值都小于0.05
        x2_const = sm.add_constant(x_c)
        sm_lr2 = sm.Logit(y,x2_const).fit()
        pvalue_tup2 = [(i,j) for i,j in zip(sm_lr2.pvalues.index,sm_lr2.pvalues.values)][1:]
        delete_count = len([i for i,j in pvalue_tup2 if j>=0.05])
        
    pvalues_col = x_c.columns.tolist()
    
    return pvalues_col




def forward_delete_coef(x,y):
    """
    系数一致筛选
    param:
        x -- 特征数据集,woe转化后，且字段顺序按IV值从大到小排列 Dataframe
        y -- 标签列 Series
    return:
        coef_col -- 筛选后的特征集合 list
    """
    col_list = list(x.columns)
    coef_col = []
    coef_col_delete=[]
    # 按IV值逐个引入模型，输出系数
    for col in col_list:
        coef_col.append(col)
        x2 = x.loc[:,coef_col]
        sk_lr = LogisticRegression(random_state=0).fit(x2,y)
        coef_dict = {k:v for k,v in zip(coef_col,sk_lr.coef_[0])}
        # 当引入特征的系数为负，则将其剔除
        if coef_dict[col]<0:
            coef_col.remove(col)
            coef_col_delete.append(col)

    return coef_col,coef_col_delete


def iv_transform_df(bin_df_list):  
    """
    将变量iv列表，转换为变量iv数据框，便于查看1
    param:
        bin_df_list -- 变量iv列表
    return：
        iv_df -- 变量iv数据框        
    """
    iv_df=pd.DataFrame()
    for x in bin_df_list:
        if "bins" in x.columns:
            iv_df=iv_df.append(x,ignore_index=True)#不添加,ignore_index=True，会报categories must match existing categories when appending的错
        else:
            x["bins"]=x.index
            x["col"]=x.index.name
            #iv_df1 = x.reset_index().assign(var_name=x.index.name).rename(columns={x.index.name:'bins'})
            iv_df=iv_df.append(x,ignore_index=True)
                 
    iv_df=iv_df.reset_index(drop=True)

    var_name_order=['col','bins', 'IV', 'woe', 'bin_iv','total', 'totalrate', 'bad', 'badrate', 'good', 'goodrate', 'badattr',
           'goodattr' ]
    iv_df=iv_df[var_name_order]
    iv_df["fuzhu"]=iv_df.index
    iv_df.sort_values(by=['IV','fuzhu'], axis=0, ascending=(False,True), inplace=True)
    iv_df=iv_df.reset_index(drop=True)
    
    return iv_df



def get_map_df(bin_df_list):
    """
    得到特征woe映射集合表
    param:
        bin_df_list -- 每个特征的woe映射表 list
    return:
        map_merge_df -- 特征woe映射集合表 Dataframe
    """
    map_df_list=[]
    for dd in bin_df_list:
        # 添加特征名列
        map_df = dd.reset_index().assign(col=dd.index.name).rename(columns={dd.index.name:'bin'})
        # 将特征名列移到第一列，便于查看
        temp1 = map_df['col']
        temp2 = map_df.iloc[:,:-1]
        map_df2 = pd.concat([temp1,temp2],axis=1)
        map_df_list.append(map_df2)   
    map_merge_df = pd.concat(map_df_list,axis=0)
    var_name_order=['col', 'bin', 'IV','woe','total', 'totalrate', 'bad', 'badrate', 'good',
       'goodrate', 'badattr', 'goodattr', 'bin_iv', 'bins',
       'var_name' ]
    map_merge_df=map_merge_df[var_name_order]
    map_merge_df["fuzhu"]=map_merge_df.index
    # map_merge_df.sort_values(by='IV', axis=0, ascending=False, inplace=True)
    map_merge_df.sort_values(by=['IV','fuzhu'], axis=0, ascending=(False,True), inplace=True)#如果不添加fuzhu字段单个变量的分箱区间会不是按照顺序排列
    map_merge_df=map_merge_df.reset_index(drop=True)
    
    return map_merge_df




def var_mapping(df,map_df,var_map,target,key_name='id'):
    """
    特征映射
    param:
        df -- 原始数据集 Dataframe
        map_df -- 特征映射集合表 Dataframe
        var_map -- map_df里映射的字段名，如"woe","score" string
        target -- 标签字段名 string
        id -- 用户唯一识别id,默认为"id"
    return:
        df2 -- 映射后的数据集 Dataframe
    """
    df2 = df.copy()
    # 去掉标签字段，遍历特征
    if key_name in df2.columns:
        for col in df2.drop([target,key_name],axis=1).columns:
            x = df2[col]
            # 找到特征的映射表
            bin_map = map_df[map_df.col==col]
            # 新建一个映射array，填充0
            bin_res = np.array([0]*x.shape[0],dtype=float)
            for i in bin_map.index:
                # 每个箱的最小值和最大值
                lower = bin_map['min_bin'][i]
                upper = bin_map['max_bin'][i]
                # 对于类别型特征，每个箱的lower和upper时一样的
                if lower == upper:
                    x1 = x[np.where(x == lower)[0]]
                # 连续型特征，左开右闭
                else:
                    x1 = x[np.where((x>lower)&(x<=upper))[0]]
                mask = np.in1d(x,x1)
                # 映射array里填充对应的映射值
                bin_res[mask] = bin_map[var_map][i]
            bin_res = pd.Series(bin_res,index=x.index)
            bin_res.name = x.name
            # 将原始值替换为映射值
            df2[col] = bin_res
    else:
        for col in df2.drop([target],axis=1).columns:
            x = df2[col]
            # 找到特征的映射表
            bin_map = map_df[map_df.col==col]
            # 新建一个映射array，填充0
            bin_res = np.array([0]*x.shape[0],dtype=float)
            for i in bin_map.index:
                # 每个箱的最小值和最大值
                lower = bin_map['min_bin'][i]
                upper = bin_map['max_bin'][i]
                # 对于类别型特征，每个箱的lower和upper时一样的
                if lower == upper:
                    x1 = x[np.where(x == lower)[0]]
                # 连续型特征，左开右闭
                else:
                    x1 = x[np.where((x>lower)&(x<=upper))[0]]
                mask = np.in1d(x,x1)
                # 映射array里填充对应的映射值
                bin_res[mask] = bin_map[var_map][i]
            bin_res = pd.Series(bin_res,index=x.index)
            bin_res.name = x.name
            # 将原始值替换为映射值
            df2[col] = bin_res
        
    return df2




def plot_roc(y_label,y_pred):
    """
    绘制roc曲线
    param:
        y_label -- 真实的y值 list/array
        y_pred -- 预测的y值 list/array
    return:
        roc曲线
    """
    tpr,fpr,threshold = metrics.roc_curve(y_label,y_pred) 
    AUC = metrics.roc_auc_score(y_label,y_pred) 
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(1,1,1)
    ax.plot(tpr,fpr,color='blue',label='AUC=%.3f'%AUC) 
    ax.plot([0,1],[0,1],'r--')
    ax.set_ylim(0,1)
    ax.set_xlim(0,1)
    ax.set_title('ROC')
    ax.legend(loc='best')
    return plt.show(ax)




def plot_model_ks(y_label,y_pred):
    """
    绘制ks曲线
    param:
        y_label -- 真实的y值 list/array
        y_pred -- 预测的y值 list/array
    return:
        ks曲线
    """
    pred_list = list(y_pred) 
    label_list = list(y_label)
    total_bad = sum(label_list)
    total_good = len(label_list)-total_bad 
    items = sorted(zip(pred_list,label_list),key=lambda x:x[0]) 
    step = (max(pred_list)-min(pred_list))/200 
    
    pred_bin=[]
    good_rate=[] 
    bad_rate=[] 
    ks_list = [] 
    for i in range(1,201): 
        idx = min(pred_list)+i*step 
        pred_bin.append(idx) 
        label_bin = [x[1] for x in items if x[0]<idx] 
        bad_num = sum(label_bin)
        good_num = len(label_bin)-bad_num  
        goodrate = good_num/total_good 
        badrate = bad_num/total_bad
        ks = abs(goodrate-badrate) 
        good_rate.append(goodrate)
        bad_rate.append(badrate)
        ks_list.append(ks)
    
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(1,1,1)
    ax.plot(pred_bin,good_rate,color='green',label='good_rate')
    ax.plot(pred_bin,bad_rate,color='red',label='bad_rate')
    ax.plot(pred_bin,ks_list,color='blue',label='good-bad')
    ax.set_title('KS:{:.3f}'.format(max(ks_list)))
    ax.legend(loc='best')
    return plt.show(ax)




def cal_scale(P0,odds,PDO,model):
    """
    计算分数校准的A，B值，基础分
    param:
        odds：设定的坏好比 float
        P0: 在这个odds下的分数 int
        PDO: 好坏翻倍比 int
        model:模型
    return:
        A,B,base_score(基础分)
    """
    B = PDO/np.log(2)
    A = P0+B*np.log(odds)
    base_score = A-B*model.intercept_[0]
    return A,B,base_score




def get_score_map(woe_df,coe_dict,B):
    """
    得到特征score的映射集合表
    param:
        woe_df -- woe映射集合表 Dataframe
        coe_dict -- 系数对应的字典
    return:
        score_df -- score的映射集合表 Dataframe
    """
    scores=[]
    coe_show=[]
    for cc in woe_df.col.unique():
        woe_list = woe_df[woe_df.col==cc]['woe'].tolist()
        coe = coe_dict[cc]
        score = [round(coe*-B*w,0) for w in woe_list]#因为在我的刻度体系里B本身是正的，所以这里的B需要添加一个负号，以和公式A-B*log(odds)匹配
        scores.extend(score)
        coe_1=[round(coe,4) for w in woe_list]
        coe_show.extend(coe_1)
    woe_df['score'] = scores
    woe_df['coe'] = coe_show
    score_df = woe_df.copy()
    var_name_order=['col', 'bins','IV', 'score','coe','woe','total','totalrate', 'bad', 'badrate', 'good',
       'goodrate', 'badattr', 'goodattr', 'bin_iv',
        'min_bin', 'max_bin']
    score_df=score_df[var_name_order]
    score_df["fuzhu"]=score_df.index
    score_df.sort_values(by=['IV','fuzhu'], axis=0, ascending=(False,True), inplace=True)
    score_df=score_df.reset_index(drop=True)
    
    return score_df


def generate_odds_score(PDO,P0,odds,a=0.0001):
    """
    计算相应违约几率下得分，#B为评分卡刻度，即pdo/log(2)，A为补偿，A=P0+B*log(theta) P0为某个违约几率下的得分，theta即为那个违约几率，a为违约几率,即odds
    param:
        odds -- 设定的坏好比 float
        P0 -- 在这个odds下的分数 int
        PDO -- 好坏翻倍比 int
        a -- 初始违约几率,默认为0.0001
    return:
        odds_score -- 各违约概率，几率下的得分数据框 dataframe
    """
    B = PDO/np.log(2)
    A = P0+B*np.log(odds)
    score=[]
    p=[]
    Odds=[]
    while a<1000:
#         p1=format(a/(1+a), '.0%')
        p1="%.2f%%" %(a/(1+a)*100)
        p.append(p1)
        Odds.append(a)
        score1=round(A-B*np.log(a),0)
        score.append(score1)
        b=2*a
        a=b
    dict_score={"违约概率":p,"几率":Odds,"得分":score}
    odds_score=pd.DataFrame(dict_score)#将字典转换成为数据框
    return(odds_score)



def score_distribution(df,target="target",column_score="final_score",group_type="qcut",qcut_cut_list=[0.1,0.2,0.3,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.8,0.9,1.0],cut_cut_list=10,right_border=True):
    """
    计算各个分数区间对应好坏用户占比，累计用户占比
    param:
        df -- 需要进行分箱操作的数据框
        group_type -- 采用的分箱方式，默认采用pandas的qcut函数来分箱，可以定义分箱分位数间隔(list)或者指定箱的组数；如果指定分箱方式为cut，则采用距离分布，可指定需要的分箱数，或者指定分箱的具体分箱点（list)
        qcut_cut_list -- 指分箱方式为qcut时对应的分箱节点列表，为分位数，默认为[0.1,0.2,0.3,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.8,0.9,1.0]，也可指定箱的组数而不指定具体列表，例如10
        cut_cut_list -- 指分箱方式为cut时对应的分箱节点列表，为分箱具体节点，也可指定箱的组数而不指定具体列表，例如10
    return:
        bin_df -- 最终得分分组后各类数量统计的数据框
    
    """
    
    total = df[target].count()
    bad = df[target].sum()
    good = total - bad
    if group_type=="qcut":
        bucket= pd.qcut(df[column_score],qcut_cut_list)        
    elif group_type=="cut":
        bucket = pd.cut(df[column_score],cut_cut_list,right=right_border)
    
#     bin_column_name=column_score+'_bins'
    
    group = df.groupby(bucket)
    bin_df = pd.DataFrame()
    bin_df['total'] = group[target].count()
    bin_df['totalrate'] = bin_df['total']/total
    bin_df['bad'] = group[target].sum()
    bin_df['badrate'] =bin_df['bad']/bin_df['total']    
    bin_df['good'] = bin_df['total'] - bin_df['bad']
    bin_df['goodrate'] = bin_df['good']/bin_df['total']  
    bin_df['lj_bad'] =  bin_df['bad'].cumsum()
    bin_df['lj_bad_rate'] =bin_df['lj_bad']/bad
    bin_df['lj_good'] = bin_df['total'].cumsum()-bin_df['bad'].cumsum()
    bin_df['lj_good_rate'] = bin_df['lj_good']/good
    bin_df['ks'] = abs(bin_df['lj_good']/good-bin_df['lj_bad']/bad)      
    bin_df['badattr'] =bin_df['bad']/bad
    bin_df['goodattr'] = (bin_df['total']-bin_df['bad'])/good
    bin_df['pass_rate'] = 1-bin_df['total'].cumsum()/total
    bin_df[column_score]=bin_df.index
    
    bin_df=bin_df.reset_index(drop=True)
    var_name_order=[ 'final_score','ks','pass_rate','total', 'totalrate', 'bad', 'badrate', 'good', 'goodrate', 'lj_bad',
       'lj_bad_rate', 'lj_good', 'lj_good_rate',  'badattr', 'goodattr' ]
    bin_df=bin_df[var_name_order]
    bin_df.sort_values(by=['final_score'], axis=0, ascending=True, inplace=True)
    

#格式转换，小数转换为百分数显示
    for col in ["totalrate","badrate","goodrate","lj_bad_rate","lj_good_rate","badattr","goodattr","pass_rate","ks"]:
        bin_df[col]=bin_df[col].map(lambda x: "%.2f%%" %(x*100))

    return bin_df
    
    

def plot_score_hist(df,target,score_col,title,plt_size=None):
    """
    绘制好坏用户得分分布图
    param:
        df -- 数据集 Dataframe
        target -- 标签字段名 string
        score_col -- 模型分的字段名 string
        plt_size -- 绘图尺寸 tuple
        title -- 图表标题 string
    return:
        好坏用户得分分布图
    """    
    plt.figure(figsize=plt_size)
    plt.title(title)
    x1 = df[df[target]==1][score_col]
    x2 = df[df[target]==0][score_col]
    sns.kdeplot(x1,shade=True,label='bad',color='hotpink')
    sns.kdeplot(x2,shade=True,label='good',color ='seagreen')
    plt.legend()
    return plt.show()




def cal_psi(df1,df2,col,bin_num=5):
    """
    计算psi
    param:
        df1 -- 数据集A Dataframe
        df2 -- 数据集B Dataframe
        col -- 字段名 string
        bin_num -- 连续型特征的分箱数 默认为5
    return:
        psi float
        bin_df -- psi明细表 Dataframe
    """
    # 对于离散型特征直接根据类别进行分箱，分箱逻辑以数据集A为准
    if df1[col].dtype == np.dtype('object') or df1[col].dtype == np.dtype('bool') or df1[col].nunique()<=bin_num:
        bin_df1 = df1[col].value_counts().to_frame().reset_index().rename(columns={'index':col,col:'total_A'})
        bin_df1['totalrate_A'] = bin_df1['total_A']/df1.shape[0]
        bin_df2 = df2[col].value_counts().to_frame().reset_index().rename(columns={'index':col,col:'total_B'})
        bin_df2['totalrate_B'] = bin_df2['total_B']/df2.shape[0]
    else:
        # 这里采用的是等频分箱
        bin_series,bin_cut = pd.qcut(df1[col],q=bin_num,duplicates='drop',retbins=True)
        bin_cut[0] = float('-inf')
        bin_cut[-1] = float('inf')
        bucket1 = pd.cut(df1[col],bins=bin_cut)
        group1 = df1.groupby(bucket1)
        bin_df1=pd.DataFrame()
        bin_df1['total_A'] = group1[col].count()
        bin_df1['totalrate_A'] = bin_df1['total_A']/df1.shape[0]
        bin_df1 = bin_df1.reset_index()

        bucket2 = pd.cut(df2[col],bins=bin_cut)
        group2 = df2.groupby(bucket2)
        bin_df2=pd.DataFrame()
        bin_df2['total_B'] = group2[col].count()
        bin_df2['totalrate_B'] = bin_df2['total_B']/df2.shape[0]
        bin_df2 = bin_df2.reset_index()
    # 计算psi
    bin_df = pd.merge(bin_df1,bin_df2,on=col)
    bin_df['a'] = bin_df['totalrate_B'] - bin_df['totalrate_A']
    bin_df['b'] = np.log(bin_df['totalrate_B']/bin_df['totalrate_A'])
    bin_df['Index'] = bin_df['a']*bin_df['b']
    bin_df['PSI'] = bin_df['Index'].sum().round(4)
    bin_df = bin_df.drop(['a','b'],axis=1)
    
    psi =bin_df.PSI.iloc[0]
    
    return psi,bin_df



# try:   
#     get_ipython().system('jupyter nbconvert --to python file_name.ipynb')
#     # python即转化为.py，script即转化为.html
#     # file_name.ipynb即当前module的文件名
# except:
#     pass

