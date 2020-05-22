#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#数据来源" data-toc-modified-id="数据来源-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>数据来源</a></span></li><li><span><a href="#名词解释" data-toc-modified-id="名词解释-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>名词解释</a></span></li><li><span><a href="#总结" data-toc-modified-id="总结-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>总结</a></span></li><li><span><a href="#数据读取&amp;基本情况查看" data-toc-modified-id="数据读取&amp;基本情况查看-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>数据读取&amp;基本情况查看</a></span></li><li><span><a href="#数据预处理" data-toc-modified-id="数据预处理-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>数据预处理</a></span><ul class="toc-item"><li><span><a href="#异常值检测&amp;处理" data-toc-modified-id="异常值检测&amp;处理-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>异常值检测&amp;处理</a></span></li><li><span><a href="#特征衍生" data-toc-modified-id="特征衍生-5.2"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>特征衍生</a></span></li><li><span><a href="#缺失值检测&amp;处理" data-toc-modified-id="缺失值检测&amp;处理-5.3"><span class="toc-item-num">5.3&nbsp;&nbsp;</span>缺失值检测&amp;处理</a></span><ul class="toc-item"><li><span><a href="#缺失过多特征删除" data-toc-modified-id="缺失过多特征删除-5.3.1"><span class="toc-item-num">5.3.1&nbsp;&nbsp;</span>缺失过多特征删除</a></span></li><li><span><a href="#直接填充默认缺失值" data-toc-modified-id="直接填充默认缺失值-5.3.2"><span class="toc-item-num">5.3.2&nbsp;&nbsp;</span>直接填充默认缺失值</a></span></li></ul></li></ul></li><li><span><a href="#特征筛选" data-toc-modified-id="特征筛选-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>特征筛选</a></span><ul class="toc-item"><li><span><a href="#psi筛选" data-toc-modified-id="psi筛选-6.1"><span class="toc-item-num">6.1&nbsp;&nbsp;</span>psi筛选</a></span></li><li><span><a href="#随机森林筛选特征" data-toc-modified-id="随机森林筛选特征-6.2"><span class="toc-item-num">6.2&nbsp;&nbsp;</span>随机森林筛选特征</a></span></li><li><span><a href="#iv筛选" data-toc-modified-id="iv筛选-6.3"><span class="toc-item-num">6.3&nbsp;&nbsp;</span>iv筛选</a></span></li><li><span><a href="#人工去除偏事后特征&amp;iv箱数过多特征" data-toc-modified-id="人工去除偏事后特征&amp;iv箱数过多特征-6.4"><span class="toc-item-num">6.4&nbsp;&nbsp;</span>人工去除偏事后特征&amp;iv箱数过多特征</a></span></li><li><span><a href="#相关性筛选" data-toc-modified-id="相关性筛选-6.5"><span class="toc-item-num">6.5&nbsp;&nbsp;</span>相关性筛选</a></span></li><li><span><a href="#分箱调整" data-toc-modified-id="分箱调整-6.6"><span class="toc-item-num">6.6&nbsp;&nbsp;</span>分箱调整</a></span><ul class="toc-item"><li><span><a href="#自动分箱调整" data-toc-modified-id="自动分箱调整-6.6.1"><span class="toc-item-num">6.6.1&nbsp;&nbsp;</span>自动分箱调整</a></span></li><li><span><a href="#针对自动调整分箱未单调变量_手动调整分箱" data-toc-modified-id="针对自动调整分箱未单调变量_手动调整分箱-6.6.2"><span class="toc-item-num">6.6.2&nbsp;&nbsp;</span>针对自动调整分箱未单调变量_手动调整分箱</a></span></li><li><span><a href="#调整后的分箱再使用iv筛选一轮" data-toc-modified-id="调整后的分箱再使用iv筛选一轮-6.6.3"><span class="toc-item-num">6.6.3&nbsp;&nbsp;</span>调整后的分箱再使用iv筛选一轮</a></span></li></ul></li><li><span><a href="#woe编码后相关性&amp;多重共线性筛选" data-toc-modified-id="woe编码后相关性&amp;多重共线性筛选-6.7"><span class="toc-item-num">6.7&nbsp;&nbsp;</span>woe编码后相关性&amp;多重共线性筛选</a></span></li><li><span><a href="#显著性筛选" data-toc-modified-id="显著性筛选-6.8"><span class="toc-item-num">6.8&nbsp;&nbsp;</span>显著性筛选</a></span></li><li><span><a href="#剔除系数和其他系数符号不一致的特征" data-toc-modified-id="剔除系数和其他系数符号不一致的特征-6.9"><span class="toc-item-num">6.9&nbsp;&nbsp;</span>剔除系数和其他系数符号不一致的特征</a></span></li></ul></li><li><span><a href="#模型训练和评价" data-toc-modified-id="模型训练和评价-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>模型训练和评价</a></span></li><li><span><a href="#分数映射&amp;分数分布" data-toc-modified-id="分数映射&amp;分数分布-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>分数映射&amp;分数分布</a></span><ul class="toc-item"><li><span><a href="#分数刻度&amp;各入模变量相应分箱得分" data-toc-modified-id="分数刻度&amp;各入模变量相应分箱得分-8.1"><span class="toc-item-num">8.1&nbsp;&nbsp;</span>分数刻度&amp;各入模变量相应分箱得分</a></span></li><li><span><a href="#训练集&amp;验证集&amp;时间外样本分数转换" data-toc-modified-id="训练集&amp;验证集&amp;时间外样本分数转换-8.2"><span class="toc-item-num">8.2&nbsp;&nbsp;</span>训练集&amp;验证集&amp;时间外样本分数转换</a></span></li><li><span><a href="#训练集&amp;验证集&amp;时间外样本分数分箱分布" data-toc-modified-id="训练集&amp;验证集&amp;时间外样本分数分箱分布-8.3"><span class="toc-item-num">8.3&nbsp;&nbsp;</span>训练集&amp;验证集&amp;时间外样本分数分箱分布</a></span></li></ul></li></ul></div>

# # 评分卡建模模板

# author:33
# 
# date:2020/5/14

# In[1]:


#隐藏导出html和pdf里面的代码
#这个在你自己编辑notebook时不起作用，对导出结果中所有代码起作用：导出html或者pdf中是没有代码的
from IPython.display import display
from IPython.display import HTML
import IPython.core.display as di
di.display_html('<script>jQuery(function() {if (jQuery("body.notebook_app").length == 0) { jQuery(".input_area").toggle(); jQuery(".prompt").toggle();}});</script>', raw=True)
CSS = """#notebook div.output_subarea {max-width:100%;}""" #changes output_subarea width to 100% (from 100% - 14ex)
HTML('<style>{}</style>'.format(CSS))


# In[2]:


#包加载
import sys
import os ##显示工作路径，修改工作路径的包
import numpy as np##是python数值计算基石，听过多种数据结构，算法以及设计计算所需的接口
import pandas as pd##提供了高级数据结构和函数，dataframe,series
import math 
import statsmodels.api as sm##统计分析包，方差分析，时间序列分析，非参数估计 import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier,_tree
from IPython.display import display
import warnings
warnings.filterwarnings('ignore')
#可视化
import matplotlib.pyplot as plt ##python里适用最广泛的 可视化库，与其他库良好整合
import seaborn as sns ##此包是基于matplotlib的可视化包
import itables.interactive#每一个表都是交互型的表
from itables import show
import itables.options as opt
opt.classes = ["display", "nowrap"]
opt.lengthMenu = [10, 10, 100, 200, 500, 1000, 2000, 5000]
opt.maxBytes = 90000000
opt.maxColumns=350
opt.columnDefs=[{"width": "150px", "targets": "_all"}]
from datetime import datetime


# In[3]:


# ?LogisticRegression


# In[4]:


##导入自定义函数
##当自定义函数和当前默认工作路径不一致时
sys.path.append('') #自定义函数对应的路径
import self_f_2  #加载自定义的函数py文件
# sys.path
warnings.filterwarnings('ignore')

##当自定义函数和当前默认工作路径一致时
# import FileWriteAbout  # 导入另一个文件
# FileWriteAbout.writeList()  #调用其中的函数名

os.getcwd()
# print(os.getcwd())#显示当前路径
os.chdir('')#更改路径，''里面为更改的路径


# ## 数据来源
# 


# ## 名词解释
# 


# ## 总结

# ## 数据读取&基本情况查看

# In[5]:


# pd.set_option('display.max_columns',400)
# pd.set_option('display.max_rows',500)

data=pd.read_csv("",dtype={"order_id":"str","account_id":"str"})
data.rename(columns={"apply_dt":"apply_time","deadline":"repay_date"},inplace=True)


# data['apply_time']=pd.to_datetime(data['apply_time'],format='%Y-%m-%d')
data["apply_time"]=data["apply_time"].map(lambda x:datetime.strptime(x, "%Y-%m-%d").date())
data["repay_date"]=data["repay_date"].map(lambda x:datetime.strptime(x, "%Y-%m-%d").date())
# data["register_time"]=data["register_time"].map(lambda x:datetime.strptime(x, "%Y-%m-%d").date())
data["apply_time"]=data["apply_time"].astype("datetime64")
data["repay_date"]=data["repay_date"].astype("datetime64")
# data["register_time"]=data["register_time"].astype("datetime64")
# data["register_time_hour"]=data["register_time"].hour

data["os_new"]=data["os"].map(lambda x:x[0:10])


# data["app_type_num"] = data.apply(lambda x : sum([1 if x[i] > 0 else 0 for i in var_list]) , axis=1)
# data["device_manufacturer_new"]=data["device_manufacturer"]
# data["device_manufacturer_new"][data["device_manufacturer_new"].isin(device_manufacturer_hebing)]='others'

print("\033[1;31m样本量\n \033[0m",data.shape)
print(data.columns.values)
# print(type(data))
# print(data.dtypes)
# print(data.describe())
show(data.describe())

# ,dtype={"order_id":"str","account_id":"str"}


# In[6]:


#数据读取



# print(data.children_number.dtypes)


# print(data.info())
# show(data.head())

def sample_tag(x):
    if x>0:
        return 1
    elif x<=0:
        return 0
    else:
        return 2
    
        

data['target']=data.overdue_days.apply(lambda x:sample_tag(x))
data=data[data['apply_time']>'2020-04-03']



print("\033[1;31m样本好坏用户数量\n \033[0m",data['target'].value_counts())

# data=data[(data['repay_date']<='2020-03-30')&(data['repay_date']>='2020-03-04')]
# print(data.target.value_counts())
print("\033[1;31m申请时间分布\n \033[0m")
show(data['apply_time'].value_counts())
# print(data['repay_date'].value_counts())

data.to_excel("data.xlsx")

# print(data.apply_time.value_counts())
# overdue_days=data["overdue_days"].to_frame()
# show(overdue_days.describe())

# device_memory=data["device_memory"].to_frame()
# show(device_memory.describe())


# show(data.head(),columnDefs=[{"width": "300px", "targets": "_all"}])
# print("\033[1;31m连续型变量分布\n \033[0m")
# show(data.describe(),columnDefs=[{"width": "200px","high": "80px", "targets": "_all"}])
# print("\033[1;31m离散型变量分布\n \033[0m")
# show(data.describe(include=['object']))


# In[7]:


# 1229/(1229+4415)


# In[8]:



# a=pd.DataFrame()
# a1=data[data['network'].isin([0])]
# a=a.append(a1)
# a2=data[data['dev_sms_year_num'].isin([3])]
# a=a.append(a2)
# a.to_excel("a.xlsx")


# In[9]:


# print(data['repay_time'].head(20))
# data['repay_time']=pd.to_datetime(data['repay_time'],format='%Y-%m-%d')
# print(data['repay_time'].head(20))

# print(data['apply_time'].head(20))
# data['apply_time']=pd.to_datetime(data['apply_time'],format='%Y-%m-%d')
# print(data['apply_time'].head(20))
# data.repay_time.max()


# ## 数据预处理
# 
# ### 异常值检测&处理
# 
# 针对异常变量可根据对数据对理解进行极端值分数映射转换或者删除极端值对应对样本

# In[10]:


#1.数据预处理

#1.1异常值
    #1.1.1异常值检测

character_feature,count=self_f_2.character_VarFindFunc(data)
data_num=data.drop(character_feature,axis=1)
print("\033[1;31m离散型变量个数\n \033[0m",count)
print("\033[1;31m连续型变量个数\n \033[0m",len(data.columns)-count)

data_abnormal=data.copy()

data_abnormal=self_f_2.find_abnormal(data_abnormal)

# data_abnormal.head()
# data_abnormal.describe()
data_abnormal.describe(include=['object'])#查看离散型特征情况

#针对异常变量可根据对数据对理解进行极端值分数映射转换或者删除极端值对应对样本


# ### 特征衍生
# 这里主要是做特征各种交叉衍生的工作

# ### 缺失值检测&处理

# In[11]:


#1.2.1缺失值统计

feature_na,na_feature_num=self_f_2.findNaFunc(data,is_show_all=-1)

feature_na.to_excel("feature_na.xlsx")
# feature_na=feature_na.rename(columns={"queshi_num":"缺失值数量","na_rate":"缺失率"})
print("\033[1;31m有缺失值的变量个数:\n \033[0m",na_feature_num)
print("\033[1;31m各变量缺失率展示\n \033[0m")
feature_na_show=feature_na.sort_values(by='queshi_num', axis=0, ascending=False)
feature_na_show=feature_na_show.reset_index(drop=False)
show(feature_na_show)


# In[12]:


#缺失值按日统计
# apply_time_list=data['apply_time'].unique()
# print(apply_time_list)
# print(data['apply_time'].value_counts())
# apply_time_list=data['apply_time'].unique()
# feature_na_day_ttl=pd.DataFrame()
# for i in apply_time_list:
#     data_temp=data[data['apply_time']==i]
#     feature_na_day,na_feature_num_day=self_f_2.findNaFunc(data_temp,is_show_all=-1)
#     feature_na_day['date']=i
#     var_name_order=["date","var_name","queshi_num","na_rate","na_rate_new"]
#     feature_na_day=feature_na_day[var_name_order]
#     feature_na_day_ttl=pd.concat([feature_na_day_ttl,feature_na_day])

# feature_na_day_ttl_new=feature_na_day_ttl[(feature_na_day_ttl['date']>'2020-02-22') & (feature_na_day_ttl['date']<'2020-03-23')] 
# # print(feature_na_day_ttl_new.date.value_counts())
# var_name_list= feature_na_day_ttl_new["var_name"].unique()
# # print(var_name_list)
# var_na_rate_variance=pd.DataFrame()
# for i in var_name_list:
#     data_temp1=feature_na_day_ttl_new[feature_na_day_ttl_new['var_name']==i]
#     data_temp1["variance"]=data_temp1["na_rate"].var()
#     var_na_rate_variance=pd.concat([var_na_rate_variance,data_temp1])
    
# var_na_rate_variance.to_excel("变量按日缺失率方差20200331.xlsx")   
    
# feature_na_day_ttl.to_excel("feature_na_day_ttl.xlsx")


# #### 缺失过多特征删除

# In[13]:


#缺失过多特征删除
delete_feature=feature_na[feature_na["na_rate"]>0.99].var_name #当不想删除缺失特征时，可提高na_rate阈值
data_delete=data.drop(delete_feature,axis=1)

# data_delete.info()

# print(data_delete.apply_time)
print("\033[1;31m删除的缺失值过高的特征:\n \033[0m",delete_feature)


# #### 直接填充默认缺失值
# 
# - 连续型特征填充-999，离散型特征填充“缺失”字样
# 
# - 另外的缺失值填充方法
#     - 根据相关性填充
#     - 根据后续结果看是否缺失值要根据特征中位数或者均值来填补

# In[14]:


#直接填充默认缺失值，连续型特征填充-999，离散型特征填充“缺失”字样
data_fillna=self_f_2.na_fill(data_delete)
print("\033[1;31m缺失值填充后变量分布\n \033[0m")
show(data_fillna.describe())
show(data_fillna.describe(include=['object']))
data_fillna.to_excel("data_fillna.xlsx")

#另外的缺失值填充方法
#根据相关性填充
#根据后续结果看是否缺失值要根据特征中位数或者均值来填补


# ## 特征筛选

# In[15]:


# a=data.columns.to_frame()
# a.to_excel("var.xlsx")


# In[16]:


# group_temp = data.groupby('apply_time')
# bin_df_temp = pd.DataFrame()
# bin_df_temp['apply_time_num'] = group_temp['is_overdue'].count()
# print(bin_df_temp)

#训练集，时间外验证集拆分
data_fillna=data_fillna[data_fillna['target'] !=2]
train_data=data_fillna[data_fillna['apply_time']<'2020-04-23']
oot_data=data_fillna[data_fillna['apply_time']>='2020-04-23']
# print(data_fillna['apply_time'].value_counts())

# print(train_data.overdue_days.value_counts())
# print(oot_data.overdue_days.value_counts())

# train_data=data_fillna[data_fillna['repay_date']<='2020-03-22']
# oot_data=data_fillna[(data_fillna['repay_date']>'2020-03-22')&(data_fillna['repay_date']<'2020-03-31')]

# print(oot_data.repay_date.value_counts())

# train_data.to_excel("train_data_original.xlsx")
# oot_data.to_excel("oot_data_original.xlsx")

print("\033[1;31m训练集好坏用户，1表示坏用户：\n \033[0m",train_data.target.value_counts())
print("\033[1;31m时间外验证集好坏用户，1表示坏用户：\n\033[0m",oot_data.target.value_counts())


#不作为自特征的特征
target="target" #目标特征对应的字段名称
key_name="order_id" #唯一识别用户的字段，在最后面需要知道具体是哪个用户的得分的时候，需要用到
target_series=pd.Series("target")
not_model_feature=["media_source",'order_id', 'account_id', 'loan', 'period', 'apply_time', 'loan_dt',
       'finish_dt', 'repay_date', 'overdue_days',"date_of_birth","imei","imei_md5",'dt', 'deadline_days','os']

#建模前记录下train_data的用户ID和index
train_id=train_data[key_name]
# print(train_id.head())

#删除掉
train_data=train_data.drop(not_model_feature,axis=1)

print("\033[1;31m去除事后以及事后以及根据经验较为波动不适宜入模的变量：\n \033[0m",not_model_feature)


# ###  psi筛选

# In[17]:


### psi筛选
# 3.1psi筛选
all_col = [x for x in train_data.columns if x!=target]
psi_tup = []
psi_bin_df=pd.DataFrame()
for col in all_col:
# for col in ['dev_apps_casino_cnt']:
    psi,psi_bin_df1 = self_f_2.cal_psi(train_data,oot_data,col)
    psi_tup.append((col,psi))
    psi_bin_df=psi_bin_df.append(psi_bin_df1)
psi_delete = [x for x,y in psi_tup if y>=0.2]#这里psi阈值定位0.25

#变量和psi转换成数据框
psi_df=pd.DataFrame(pd.Series(dict(psi_tup)),columns=["psi"])
psi_df=psi_df.reset_index().rename(columns={'index':'var_name'})
psi_df=psi_df.sort_values(by='psi', axis=0, ascending=False)

train_data = train_data.drop(psi_delete,axis=1)

print('\033[1;31mpsi筛选删除的特征: \033[0m',psi_delete)
print('\033[1;31mpsi筛选的特征: \033[0m',train_data.columns)

print('\033[1;31m各变量psi: \033[0m')
show(psi_df)
print('\033[1;31m各变量psi明细: \033[0m')
show(psi_bin_df)
print('\033[1;31mpsi筛选特征完成--------------------------------\033[0m')


# ### 随机森林筛选特征
# 
# 此版本，没有使用随机森林筛选变量，即所有变量都进入下一轮iv的筛选

# In[18]:


# df=train
# le=preprocessing.LabelEncoder()
# character_name,cate_count=self_f_2.character_VarFindFunc(df)
# df_new=df.copy()
# # print(df_new.info())
# character_name.remove(target)
# for i in character_name:
#     code_list=list(df_new[i].unique())
#     encoder= le.fit(code_list)
#     print(i)
#     print(encoder)
#     print("------------")
#     df_new[i] = encoder.transform(df_new[i])
#     print(df_new[i])
            
# show(df_new)


# In[19]:


#2.2随机森林筛选特征
#由于树模型不能处理非数值型特征，对离散型特征进行labelencoder编码，即将离散型特征使用连续数字来编码，
#没有啥业务含义，只是一个将离散型特征映射为数字而已

#data_label_encode=self_f_2.feature_labelencoder(train.drop(not_model_feature,axis=1))

#！重要，当没有psi这步时，需要进行这步赋值
# train=train_data

data_label_encode=self_f_2.feature_labelencoder(train_data) 

#将重要性大于一定阈值的特征筛选出来          
df_random_var,importance_list_delete=self_f_2.Random_feature_filter(data_label_encode,im_value=0.0)
random_var_name=df_random_var['var_name']#重要度大于阈值到特征，不含目标特征

print("\033[1;31m随机森林删除的特征: \033[0m")
show(importance_list_delete)
print("\033[1;31m随机森林筛选的特征: \033[0m")
show(df_random_var)

df_random_var.to_excel("df_random_var.xlsx")
random_var_name=random_var_name.append(target_series)#添加目标特征，即y特征
train_data=train_data[random_var_name]


# ### iv筛选

# In[20]:


#2.3iv筛选

# 特征分箱,默认用的是决策树分箱
train_col = [x for x in train_data.columns if x!=target]
bin_df_list=[]
cut_list=[]
for col in train_col:
    try:
        bin_df,cut = self_f_2.binning_var(train_data,col,target)
        bin_df_list.append(bin_df)
        cut_list.append(cut)
    except:
        pass
print('\033[1;31m特征分箱完成---------------------------\033[0m')

# print(bin_df_list[0])

bin_df_list_copy=bin_df_list.copy()
iv_df=self_f_2.iv_transform_df(bin_df_list_copy)
# print(bin_df_list[0])#为什么会改变呢

print("\033[1;31m变量iv以及分箱表 \033[0m")
iv_df_new=iv_df[~iv_df['col'].isin(["brand"])]
show(iv_df_new,columnDefs=[{"width": "90px", "targets": "_all"}])
iv_df.to_excel("iv_df.xlsx")


# In[21]:


# 保存每个特征的分割点list，此步骤需要在剔除
cut_dict={}
for dd,cc in zip(bin_df_list,cut_list):
    col = dd.index.name
#     print(col)
#     col = dd['var_name'].unique()
    cut_dict[col] = cc
    
# print(len(cut_dict))

# 剔除iv无限大的特征
woe_inf_var=[x.index.name for x in bin_df_list if x.IV.iloc[0]==float('inf')]
print("\033[1;31mwoe存在inf的特征:\033[0m",woe_inf_var)
bin_df_list = [x for x in bin_df_list if x.IV.iloc[0]!=float('inf')]

# print(bin_df_list)

# 将IV从大到小进行排序
iv_col = [x.index.name for x in bin_df_list]
iv_value = [x.IV.iloc[0] for x in bin_df_list]
iv_sort = sorted(zip(iv_col,iv_value),key=lambda x:x[1],reverse=True)


# iv筛选，筛选iv大于0.02的特征
iv_select_col = [x for x,y in iv_sort if y>=0.02]
iv_delete_col= [x for x,y in iv_sort if y<0.02]

print("\033[1;31miv筛选的特征:\033[0m",iv_select_col)
print("\033[1;31miv删除的特征:\033[0m",iv_delete_col)
print('\033[1;31miv筛选特征完成-----------------------------------\033[0m')


# ### 人工去除偏事后特征&iv箱数过多特征
# 
# 之所以在这个步骤去除，是因为一定程度上做过特征筛选，特征没有那么多了，可以更好地对每个特征进行解读；为什么放在相关性筛选之前做这个动作，是因为相关性筛选里面涉及到不同变量间关系而对变量进行删除，可能存在事后变量的存在而导致删除其余非事后变量这种现象，因此人工剔除偏事后特征在这一步做相对比较合适；

# ### 相关性筛选

# In[22]:


#2.5相关性筛选，多重共线性筛选
# 特征分类
cate_col = []
num_col = []
for col in iv_select_col:
    if train_data[col].dtype==np.dtype('object') or train_data[col].dtype==np.dtype('bool') or train_data[col].nunique()<=5:
        cate_col.append(col)
    else:
        num_col.append(col)

#相关性筛选，默认相关系数阈值0.7
corr=pd.DataFrame(train_data[num_col].corr())
print("\033[1;31m相关矩阵\033[0m")
show(corr,columnDefs=[{"width": "250px", "targets": "_all"}])

corr.to_excel("corr.xlsx")

#按照iv高低删除相关性高的变量
corr_select_col,corr_delete_col= self_f_2.forward_corr_delete(train_data,num_col,cut_value=0.72)

print("\033[1;31m相关性筛选的特征:\033[0m",corr_select_col)
print("\033[1;31m相关性筛选删除的特征:\033[0m",corr_delete_col)
print('\033[1;31m连续变量相关性筛选完成----------------------------- \033[0m')

#  # 多重共线性筛选，系数阈值10
# vif_select_col,vif_delete_col = self_f_2.vif_delete(train,corr_select_col)


# print("\033[1;31m多重共线性筛选的特征:\033[0m",vif_select_col)
# print("\033[1;31m多重共线性删除的特征:\033[0m",vif_delete_col)
# print('\033[1;31m连续变量多重共线性筛选完成------------------------- \033[0m')


# In[23]:


iv_df_chubu_filter=iv_df[iv_df["col"].isin(corr_select_col)]
# iv_df_chubu_filter=iv_df[iv_df["col"].isin(vif_delete_col)]
print("\033[1;31m初筛后剩余变量的iv列表\033[0m")
show(iv_df_chubu_filter)


# ### 分箱调整

# In[24]:


#分箱不单调的变量
nan_value= -999
# trim_var_dict = {k:v for k,v in cut_dict.items() if k in vif_select_col}
trim_var_dict = {k:v for k,v in cut_dict.items() if k in corr_select_col}

col_not_dandiao_befor_adjust_list=[]
for col in trim_var_dict.keys():
    bin_cut = trim_var_dict[col]
    df_bin_before_adjust = [x for x in bin_df_list if x.index.name==col][0]
    if nan_value in bin_cut: # 当cut里有缺失映射值时，说明是把缺失单独分为一箱的，后续在进行调成单调分箱时
    # 不考虑缺失的箱，故将缺失映射值剔除
        woe_list = df_bin_before_adjust['woe'].tolist()[1:]
    else:
        woe_list = df_bin_before_adjust['woe'].tolist() 

    if not self_f_2.judge_decreasing(woe_list) and not self_f_2.judge_increasing(woe_list):
        col_not_dandiao_befor_adjust_list.append(col)
        
print("\033[1;31m调整前_woe不单调的变量\033[0m",col_not_dandiao_befor_adjust_list)


# In[25]:


iv_df_not_dandiao=iv_df_chubu_filter[iv_df_chubu_filter['col'].isin(col_not_dandiao_befor_adjust_list)]

print("\033[1;31m调整前_woe不单调的变量对应woe分箱情况\033[0m")
show(iv_df_not_dandiao)


# #### 自动分箱调整

# In[26]:


# 自动调整单调分箱

#不需要调整分箱都变量筛选,筛选出需要调整分箱都变量
not_adjust_var=["longitude"]
all_car=list(trim_var_dict.keys())
# print(all_car)
for i in not_adjust_var:
    all_car.remove(i)

need_adjust_var=all_car
# print(need_adjust_var)
# print(trim_var_dict)
#不需要调整分箱都变量筛选

trim_bin_list=[]
for col in need_adjust_var:
    bin_cut = trim_var_dict[col]
    df_bin = [x for x in bin_df_list if x.index.name==col][0]
    if nan_value in bin_cut: # 当cut里有缺失映射值时，说明是把缺失单独分为一箱的，后续在进行调成单调分箱时
    # 不考虑缺失的箱，故将缺失映射值剔除
        woe_list = df_bin['woe'].tolist()[1:]
    else:
        woe_list = df_bin['woe'].tolist() 
#     print("\033[1;31m------------------\033[0m")
#     print(col,bin_cut)
#     print(col,woe_list)
    if not self_f_2.judge_decreasing(woe_list) and not self_f_2.judge_increasing(woe_list): 
#         print(col)
        monot_cut,woe_lst_new = self_f_2.monot_trim(train_data, col, target, nan_value=nan_value, cut=bin_cut)
#         print(monot_cut)
#         print(woe_lst_new)
        monot_bin_df = self_f_2.binning_trim(train_data, col, target, cut=monot_cut, right_border=True)
        trim_bin_list.append(monot_bin_df)
    else:
        trim_bin_list.append(df_bin)

#添加不需要调整分箱都变量的分箱信息
for col in not_adjust_var:
    bin_cut = trim_var_dict[col]
    df_bin = [x for x in bin_df_list if x.index.name==col][0]
    trim_bin_list.append(df_bin) 

col_not_dandiao_list=[]    
for col in need_adjust_var:
    bin_cut = trim_var_dict[col]
    df_bin_after_adjust = [x for x in trim_bin_list if x.index.name==col][0]
    if nan_value in bin_cut: # 当cut里有缺失映射值时，说明是把缺失单独分为一箱的，后续在进行调成单调分箱时
    # 不考虑缺失的箱，故将缺失映射值剔除
        woe_list = df_bin_after_adjust['woe'].tolist()[1:]
    else:
        woe_list = df_bin_after_adjust['woe'].tolist() 

    if not self_f_2.judge_decreasing(woe_list) and not self_f_2.judge_increasing(woe_list):
        col_not_dandiao_list.append(col)

trim_bin_list_copy=trim_bin_list.copy()
auto_adjust_iv_df=self_f_2.iv_transform_df(trim_bin_list_copy)

col_not_dandiao_iv_df=auto_adjust_iv_df[auto_adjust_iv_df['col'].isin(col_not_dandiao_list)]
col_not_dandiao_iv_df.col.unique()
        
print("\033[1;31m不单调也无需调整分箱的变量\033[0m",not_adjust_var)
print("\033[1;31m调整后_仍然不单调的变量\033[0m",col_not_dandiao_iv_df.col.unique())
print("\033[1;31m自动调整分箱后woe情况\033[0m")
show(auto_adjust_iv_df)


# #### 针对自动调整分箱未单调变量_手动调整分箱

# In[27]:


#删除自动调整后不单调变量
col_rengong_tiaozheng_list=["dev_sms_pdl_amnt_min"] 

auto_adjust_iv_df=auto_adjust_iv_df[~auto_adjust_iv_df['col'].isin(col_rengong_tiaozheng_list)]
# print("删除自动调整后不单调变量后剩余的变量",auto_adjust_iv_df.col.unique())


#针对自动调整没有单调的变量，手动调整
rengong_adjust_bin_list=[]
dev_sms_pdl_amnt_min_bin = self_f_2.binning_trim(train_data, "dev_sms_pdl_amnt_min", target, 
                                                cut=[float('-inf'),-999,1427,float('inf')], right_border=True)
rengong_adjust_bin_list.append(dev_sms_pdl_amnt_min_bin)


rengong_adjust_iv_df=self_f_2.iv_transform_df(rengong_adjust_bin_list)  


#添加人工调整分箱的变量

adjust_iv_df_all=pd.concat([rengong_adjust_iv_df,auto_adjust_iv_df])

adjust_iv_df_all_rengong=adjust_iv_df_all[adjust_iv_df_all["col"].isin(col_rengong_tiaozheng_list)]
print("\033[1;31m人工手动调整后_调整变量woe情况\033[0m")
show(adjust_iv_df_all_rengong)


# #### 调整后的分箱再使用iv筛选一轮
# 
# - 调整后筛选iv大于0.02的变量，同时针对不单调的iv较低变量也进行删除；

# In[28]:


##### 调整后的分箱再根据iv筛选一遍

adjust_iv_df_all=adjust_iv_df_all[adjust_iv_df_all["IV"]>0.03]

low_iv_not_dandiao=['dev_apps_EDUorBOOKS_gapdays_max',
       'dev_contact_tel_same_avg','dev_apps_COMMorSOC_gapdays_min',
       'dev_apps_GAME_NOTCASINO_installdays_max',
       'dev_apps_COMMorSOC_gapdays_mean']

adjust_iv_df_all=adjust_iv_df_all[-adjust_iv_df_all['col'].isin(low_iv_not_dandiao)]

# show(adjust_iv_df_all)

# select_num_df = []
# select_num_var=[]
# for dd in trim_bin_list:
# # for dd in bin_df_list:
#     if dd.IV.iloc[0]>=0.02 and dd.index.name in vif_select_col:
#         select_num_df.append(dd)
#         select_num_var.append(dd.index.name)

after_adjust_var=adjust_iv_df_all.col.unique()
print('\033[1;31m调整分箱后筛选的变量 \033[0m',after_adjust_var)


# In[29]:


# 连续型特征的woe映射集合表
# woe_map_num = self_f_2.get_map_df(select_num_df)
woe_map_num=adjust_iv_df_all.copy()
woe_map_num['bins'] = woe_map_num['bins'].map(lambda x:str(x))
woe_map_num['min_bin'] = woe_map_num['bins'].map(lambda x:x.split(',')[0][1:])
woe_map_num.to_excel("woe_map_num.xlsx")
woe_map_num['max_bin'] = woe_map_num['bins'].map(lambda x:x.split(',')[1][:-1])
# show(woe_map_num)
woe_map_num['min_bin'] = woe_map_num['min_bin'].map(lambda x:float(x))
woe_map_num['max_bin'] = woe_map_num['max_bin'].map(lambda x:float(x))


print("\033[1;31m连续型变量分箱调整后分箱和woe情况 \033[0m")
show(woe_map_num)
# print(woe_map_num.columns)


# In[30]:


if len(cate_col)>0:
#     print(cate_col)
    bin_cate_list = [x for x in bin_df_list if x.index.name in cate_col]
    select_cate_df=bin_cate_list.copy()
    # 剔除woe不单调的离散形特征,正常离散型特征无需管woe是否单调，除非是有等级顺序的离散型特征
#     select_cate_df=[]
#     for i,dd in enumerate(bin_cate_list):
#         woe_lst = dd['woe'].tolist()
#         if self_f_2.judge_decreasing(woe_lst) or self_f_2.judge_increasing(woe_lst):
#             select_cate_df.append(dd)
    # 离散型特征的woe映射集合表
    if len(select_cate_df)>0:
#         woe_map_cate = self_f_2.get_map_df(select_cate_df)
        woe_map_cate=self_f_2.iv_transform_df(select_cate_df)  
#         print(woe_map_cate)
        woe_map_cate['min_bin'] = list(woe_map_cate['bins'])
        woe_map_cate['max_bin'] = list(woe_map_cate['bins'])
        woe_map_df = pd.concat([woe_map_cate,woe_map_num],axis=0).reset_index(drop=True)
else:
    woe_map_df = woe_map_num.reset_index(drop=True)
    
discrete_var_woe_df=woe_map_df[woe_map_df["col"].isin(cate_col)]
discrete_var_woe_df_new=discrete_var_woe_df[discrete_var_woe_df['IV']>0.03]



cate_col_total=discrete_var_woe_df.col.unique()
cate_col_filter=discrete_var_woe_df_new.col.unique()
cate_col_delete=list(set(cate_col_total).difference(set(cate_col_filter)))

print('\033[1;31miv筛选后新去除的离散型变量 \033[0m',cate_col_delete)
print('\033[1;31miv筛选后的离散型变量 \033[0m',cate_col_filter)

print("\033[1;31m连续和离散型变量分箱调整后分箱和woe情况 \033[0m")

woe_map_df=woe_map_df[-woe_map_df["col"].isin(cate_col_delete)]
show(woe_map_df)

woe_map_df.to_csv("woe_map_df.csv")


# ### woe编码后相关性&多重共线性筛选

# In[31]:


# woe编码后相关性筛选

#对train使用woe编码替换
select_all_col = woe_map_df['col'].unique().tolist()
select_sort_col = [x for x,y in iv_sort if x in select_all_col]
# print(select_sort_col)

train2 = train_data.loc[:,select_sort_col+[target]].reset_index(drop=True)
# print(train2.shape[0])


# woe映射
train_woe = self_f_2.var_mapping(train2,woe_map_df,'woe',target)
X = train_woe.loc[:,select_sort_col]
y = train_woe[target]
    
#相关性展示
corr=pd.DataFrame(train_data[num_col].corr())
corr_woe=pd.DataFrame(train_woe[select_sort_col].corr())
show(corr_woe, columnDefs=[{"width": "200px", "targets": "_all"}])

#相关性筛选，默认阈值0.7
corr_select_col_woe,corr_delete_col_woe = self_f_2.forward_corr_delete(train_woe,select_sort_col,cut_value=0.7)
print("\033[1;31m woe编码后相关性筛选的特征:\033[0m",corr_select_col_woe)
print("\033[1;31m woe编码后相关性删除的特征:\033[0m",corr_delete_col_woe)
print("\033[1;31m 相关性筛选完成-----------------------------\033[0m")

#多重共线性筛选，系数阈值10
vif_select_col_woe,vif_delete_col_woe  = self_f_2.vif_delete(train_woe,corr_select_col_woe)
print("\033[1;31m woe编码后多重共线性筛选的特征:\033[0m",vif_select_col_woe)
print("\033[1;31m woe编码后多重共线性删除的特征:\033[0m",vif_delete_col_woe)
print('\033[1;31m 多重共线性筛选完成--------------------------\033[0m')

#经过woe变换后的特征相关性和共线性筛选后的特征数据框
X1 = X.loc[:,vif_select_col_woe]


# ### 显著性筛选

# In[32]:


# 3.8 显著性筛选，前向逐步回归
pvalue_select_col,pvalue_delete_col = self_f_2.forward_pvalue_delete(X1,y)

print("\033[1;31m 显著性筛选的变量:\033[0m",pvalue_select_col)
print("\033[1;31m 显著性筛选删除的变量:\033[0m",pvalue_delete_col)
print('\033[1;31m 显著性筛选完成--------------------------\033[0m')


# ### 剔除系数和其他系数符号不一致的特征
# **这里之所以没有说系数为正或者为负，是因为如果计算woe时为坏比好，那么逻辑回归系数就为正，如果计算woe时为好比坏，那么逻辑回归系数就为负**

# In[33]:


# 剔除系数为负数的特征
X2 = X1.loc[:,pvalue_select_col]
# X2 = X1 #不做显著性筛选时
coef_select_col,coef_delete_col = self_f_2.forward_delete_coef(X2,y)

print("\033[1;31m 删除系数为负的特征: \033[0m",coef_delete_col)
print("\033[1;31m 最终入模特征: \033[0m",coef_select_col)


# ## 模型训练和评价

# In[34]:


# LR建模
X3 = X2.loc[:,coef_select_col]
# x_train,x_valid,y_train,y_valid = train_test_split(X3,y,test_size=0.2,random_state=0)

#当没有验证集时
x_train=X3
y_train=y
# 保存训练集和验证集的index
train_index = x_train.index.tolist()
# valid_index = x_valid.index.tolist()

# show(X3,maxBytes=0, columnDefs=[{"width": "200px", "targets": "_all"}])

lr_model = LogisticRegression(C=1.0).fit(x_train,y_train)
coe_dict = {x:y for x,y in zip(x_train.columns,lr_model.coef_[0])}
print("\033[1;31m 截距：\033[0m",lr_model.intercept_[0])
print("\033[1;31m 入模特征个数： \033[0m",len(list(coe_dict.keys())))
print("\033[1;31m 特征系数： \033[0m",coe_dict)
print('\033[1;31m 建模完成----------------------------------------\n \033[0m')

print("\033[1;31m 训练集好坏样本数：\n \033[0m",y_train.value_counts())
# print("\033[1;31m 验证集好坏样本数：\n \033[0m",y_valid.value_counts())

#绘制训练集的auc，ks
train_pre = lr_model.predict_proba(x_train)[:,1]
print('\033[1;31m 训练集的AUC，KS: \033[0m')
self_f_2.plot_roc(y_train,train_pre)
self_f_2.plot_model_ks(y_train,train_pre)

# 绘制验证集的auc，ks
# valid_pre = lr_model.predict_proba(x_valid)[:,1]
# print('\033[1;31m 验证集的AUC，KS: \033[0m')
# self_f_2.plot_roc(y_valid,valid_pre)
# self_f_2.plot_model_ks(y_valid,valid_pre)  


# In[35]:


woe_map_df2 = woe_map_df[woe_map_df.col.isin(coef_select_col)].reset_index(drop=True)
# 绘制测试集的auc，ks
oot = oot_data.loc[:,coef_select_col+[target]].reset_index(drop=True)
oot_woe = self_f_2.var_mapping(oot,woe_map_df2,'woe',target)
x_oot = oot_woe.drop([target],axis=1)
y_oot = oot_woe[target]
oot_pre = lr_model.predict_proba(x_oot)[:,1]

print("\033[1;31m 时间外样本集好坏样本数 \033[0m")
print(y_oot.value_counts())

print('\033[1;31m 时间外样本集的AUC，KS \033[0m')
self_f_2.plot_roc(y_oot,oot_pre)
self_f_2.plot_model_ks(y_oot,oot_pre)


# ## 分数映射&分数分布

# ### 分数刻度&各入模变量相应分箱得分
# 
# - 分数和违约几率对应的分数映射关系为<font color=#D2691E >$ score=A-B*log(odds) $ (odds=p/1-p,p为违约概率，$log(odds)=B0+B1X+...BnX)$</font>
# 
# - 基于以下2个假设可以计算出A和B
# 
#         
# 
# - 分数越高用户越好；

# In[37]:


# 评分转换
# A,B,base_score = self_f_2.cal_scale(650,1,30,lr_model)

# A=710
# B=round(20/np.log(2),4)
# base_score=592
dict_score_scale={"A":A,"B":B,"base_score":base_score}
df_score_scale=pd.DataFrame(pd.Series(dict_score_scale),columns=["value"])
df_score_scale=df_score_scale.reset_index().rename(columns={'index':'type'})

print("\033[1;31m 评分卡刻度 \033[0m")
show(df_score_scale)
# df_score_scale.to_excel("df_score_scale.xlsx")

score_map_df  = self_f_2.get_score_map(woe_map_df2,coe_dict,B)
print("\033[1;31m 变量各分箱对应的分数 \033[0m\n")

show(score_map_df)
print(score_map_df.columns)
score_map_df.to_excel("score_map_df.xlsx",index=False)

need_columns=["col","min_bin","max_bin"]
cut_list_df=score_map_df[need_columns]


# In[38]:


#最终入模变量cut_list
need_columns=["col","min_bin","max_bin"]
cut_list_df=score_map_df[need_columns]
cut_list_df["min_bin"]=cut_list_df["min_bin"].astype("str")
cut_list_df["cut1"]=cut_list_df["min_bin"].apply(lambda x: x +",")
cut_list_group1=cut_list_df.groupby("col").agg({'cut1':'sum'})
cut_list_group1["cut_right"]='inf'
cut_list_group1["cut_new"]=cut_list_group1["cut1"]+cut_list_group1["cut_right"]


cut_list_group1.reset_index(inplace=True)
show(cut_list_group1)
cut_list=cut_list_group1[["col","cut_new"]]
cut_list.to_excel("cut_list.xlsx",index=False)


#最终入模变量系数var_coe
var_coe=score_map_df[["col","coe"]]
var_coe_new=var_coe.drop_duplicates(subset=None, keep='first', inplace=False)
var_coe_new.to_excel("var_coe.xlsx",index=False)

var_woe=score_map_df[["col","bins","woe"]]
var_woe.to_excel("var_woe.xlsx",index=False)


# ### 训练集&验证集&时间外样本分数转换

# In[39]:


# 分数映射
#训练集
train2_new=pd.merge(train_data,train_id,left_index=True,right_index=True)#之所以在这步就进行合并，将用户唯一识别ID识别出来，
#因为在下一步由于是针对原始训练集进行来训练集和验证集抽样，进行索引重制时，会更新原有索引，无法通过索引关联，嫁给你用户唯一识别码合并进来
# print(train2.shape[0])
# print(train2.columns)
# print(train2_new.shape[0])
# print(train2_new.columns)
# show(train2_new)

train_data_2 = train2_new.iloc[train_index,:].loc[:,[key_name]+coef_select_col+[target]].reset_index(drop=True)

train_score = self_f_2.var_mapping(train_data_2,score_map_df,'score',target,key_name=key_name)

train_score['final_score'] = base_score
for col in coef_select_col:
    train_score['final_score']+=train_score[col]
train_score['final_score'] = train_score['final_score'].map(lambda x:int(x))
# train_score_new=pd.merge(train_id,train_score,left_index=True,right_index=True)
train_score.to_excel("train_score.xlsx")

#验证集
# valid_data = train2_new.iloc[valid_index,:].loc[:,[key_name]+coef_select_col+[target]].reset_index(drop=True)
# valid_score = self_f_2.var_mapping(valid_data,score_map_df,'score',target)
# valid_score['final_score'] = base_score
# for col in coef_select_col:
#     valid_score['final_score']+=valid_score[col]
# valid_score['final_score'] = valid_score['final_score'].map(lambda x:int(x))
# valid_score.to_excel("valid_score.xlsx")

oot_new=oot_data.loc[:,[key_name]+coef_select_col+[target]].reset_index(drop=True)
oot_score = self_f_2.var_mapping(oot_new,score_map_df,'score',target,key_name=key_name)
oot_score['final_score'] = base_score
for col in coef_select_col:
    oot_score['final_score']+=oot_score[col]
oot_score['final_score'] = oot_score['final_score'].map(lambda x:int(x))
oot_score.to_excel("oot_score.xlsx")

print('\033[1;31m 评分转换完成--------------------------------------- \033[0m')


# ### 训练集&验证集&时间外样本分数分箱分布

# In[40]:


#分数分布，默认按照以下分位点分箱[0.1,0.2,0.3,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.8,0.9,1.0]
#训练集
print("\033[1;31m 训练集评分分箱分布\033[0m")
train_score_distribution=self_f_2.score_distribution(train_score,group_type='qcut',qcut_cut_list=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
# train_score_distribution=self_f_2.score_distribution(train_score,group_type='cut',cut_cut_list=10)
show(train_score_distribution,columnDefs=[{"width": "90px", "targets": "_all"}])

#验证集
# print("\033[1;31m 验证集评分分箱分布\033[0m")
# valid_score_distribution=self_f_2.score_distribution(valid_score,group_type="cut",cut_cut_list=10)
# show(valid_score_distribution,columnDefs=[{"width": "90px", "targets": "_all"}])

#时间外验证集
print("\033[1;31m 时间外验证集评分分箱分布\033[0m")
oot_score_distribution=self_f_2.score_distribution(oot_score,group_type='qcut',qcut_cut_list=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
# oot_score_distribution=self_f_2.score_distribution(oot_score,group_type="cut",cut_cut_list=10)
show(oot_score_distribution,columnDefs=[{"width": "90px", "targets": "_all"}])

# 训练集的评分分布
self_f_2.plot_score_hist(train_score, target, 'final_score','train_score',plt_size=(15,5))


# 验证集的评分分布
# self_f_2.plot_score_hist(valid_score, target, 'final_score','valid_score',plt_size=(6,4))
# 测试集的评分分布
self_f_2.plot_score_hist(oot_score, target, 'final_score','oot_score',plt_size=(15,5))


# In[41]:


from matplotlib.font_manager import FontManager
fm = FontManager()
mat_fonts = set(f.name for f in fm.ttflist)
# print(mat_fonts)看系统支持的字体，但是选取里面的字体仍然会初心啊不支持中文的情况，选用以下字体能支持中文
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
# 或者 plt.rc('font', family='SimHei', size=13)


# In[42]:


plt.style.use('ggplot')
fig = plt.figure(figsize=(15,5))
ax = fig.add_subplot(1,1,1)
x1=train_score_distribution['final_score'].map(lambda x:str(x))
y1=train_score_distribution['badrate'].map(lambda x:float(x.rstrip('%'))/100)
ax.plot(x1,y1,color='blue',label='badrate')
plt.xticks(rotation=30)   # 设置横坐标显示的角度，角度是逆时针，自己看
tick_spacing = 3
ax.set_title('训练集模型分和坏用户占比',size=15)
ax.legend(loc='best')


plt.style.use('ggplot')
fig = plt.figure(figsize=(15,5))
ax = fig.add_subplot(1,1,1)
x=oot_score_distribution['final_score'].map(lambda x:str(x))
y=oot_score_distribution['badrate'].map(lambda x:float(x.rstrip('%'))/100)
ax.plot(x,y,color='blue',label='badrate')
plt.xticks(rotation=30)   # 设置横坐标显示的角度，角度是逆时针，自己看
tick_spacing = 3
ax.set_title('时间外测试集模型分和坏用户占比',size=15)
ax.legend(loc='best')

