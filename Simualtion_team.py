#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import datetime
from datetime import datetime
from scipy.optimize import curve_fit
import seaborn as sns
from scipy.stats import chi
import statsmodels.formula.api as smf
get_ipython().run_line_magic('matplotlib', 'inline')

#plotly
import chart_studio.plotly as py
import plotly.graph_objects as go
from plotly.offline import iplot,init_notebook_mode

#!pip install cufflinks
import cufflinks
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)
import plotly.figure_factory as ff
import plotly.express as px

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
print(plt.style.available)
plt.style.use('ggplot')
sns.set_palette("husl")

plt.set_cmap('Accent')



# In[50]:


df=pd.read_excel('Covid-19.xlsx')
df.columns=['days', 'Total Confirm of new cases']


# In[51]:


#create an attribute with values datetime
dates=pd.date_range('27/2/2020',periods=39)
df['days']=pd.DataFrame(dates)
df.head()
#df.tail()


# In[52]:


#Create a new attributhe which is the cumulative of new cases
df['Cumulative Sum of new cases']=np.cumsum(df['Total Confirm of new cases'])
plt.plot(df['Cumulative Sum of new cases'])
df['Cumulative Sum of new cases'].iplot()


# In[5]:


#plot of timeseries
df.plot(x='days',y='Total Confirm of new cases')


colors = ['#333F44', '#37AA9C', '#94F3E4']

# Create distplot with curve_type set to 'normal'
#ig = ff.create_distplot(hist_data, group_labels, )
df[['Total Confirm of new cases','Cumulative Sum of new cases']].iplot(show_hist=False, colors=colors,title='Lineplots')



# In[6]:


#scatter plot 
plt.style.use('seaborn-dark-palette')
plt.figure(figsize=(11,7))
plt.scatter(df['days'],df['Total Confirm of new cases'])
plt.show()

#scatter plot with pyplot

fig = px.scatter(df,x=df['days'],y=df['Total Confirm of new cases'],title='Scatter plot of Total Confirm new cases')

fig.show()


# In[7]:


#seperate the dataset in two periods 
dates2=pd.date_range(df['days'][0],df['days'][38],periods=3)

#regarding a new counry, because the purpose is to fit the code on data of all the countries 
#date2=pd.date_range(df['days'],df['days'][df.shape[0]],periodw=3)


# In[8]:


s1=pd.Series(np.linspace(1,3,3),dates2)
s1
middle_day=s1.index[1] #we create a variable to seperate any dataset regarding the time period


# In[9]:


#The 1st period will have 19 days and the second the other 20
period1=df['days']<=df['days'][18]
period2=df['days']>df['days'][18]

#we create a mask for each country
#period1=df['days']<=middle_day
#period2>df['days']<=middle_day


# In[10]:


df1=df[period1]
df2=df[period2]
df1.shape,df2.shape


# In[11]:


#model 1 exponential
def func(x, a, b, c):
    return a * np.exp(b * x) - c


# In[12]:


#ploting the initial data timeseries
plt.style.use('seaborn-dark-palette')
plt.figure(figsize=(11,7))
xdata = np.linspace(1,3, 39)  
plt.plot(xdata,df['Total Confirm of new cases'].values, 'r-')

ydata=df['Total Confirm of new cases'].values

df.plot.line(x='days',y='Total Confirm of new cases')



# In[13]:


#density plot
plt.figure(figsize=(11,7))
df['Total Confirm of new cases'].plot.density()




# In[14]:


#smooth version of hist plot for the first period
plt.figure(figsize=(11,7))
df1['Total Confirm of new cases'].plot.density()


# In[15]:


#and for the second period
plt.figure(figsize=(11,7))
df2['Total Confirm of new cases'].plot.density()


# In[16]:


#both two periods
plt.figure(figsize=(11,7))
df2['Total Confirm of new cases'].plot.density(label='Period 2')
df1['Total Confirm of new cases'].plot.density(label ='Period 1')

plt.legend(loc='upper left')
plt.show()


# In[18]:


plt.hist(df['Total Confirm of new cases'],color = 'blue', edgecolor = 'black',
         bins =7,label = 'Histogram of Total num of new cases in Greece')


plt.xlabel('Number of new cases')
plt.ylabel('Days')
plt.legend()


# In[19]:


# seaborn histogram
plt.figure(figsize=(11,7))
sns.distplot(df2['Total Confirm of new cases'], hist=True, kde=True, 
             bins=7, color = 'blue',
             hist_kws={'edgecolor':'black'},label='period 2')
sns.distplot(df1['Total Confirm of new cases'], hist=True, kde=True, 
             bins=7, color = 'green',
             hist_kws={'edgecolor':'black'},label='period 1')



plt.legend(loc='upper left')
plt.show()



x1=df1['Total Confirm of new cases']
x2=df2['Total Confirm of new cases']
hist_data=[x2,x1]
rug_text_one = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
                'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                'u', 'v', 'w', 'x', 'y', 'z']

rug_text_two = ['aa', 'bb', 'cc', 'dd', 'ee', 'ff', 'gg', 'hh', 'ii', 'jj',
                'kk', 'll', 'mm', 'nn', 'oo', 'pp', 'qq', 'rr', 'ss', 'tt',
                'uu', 'vv', 'ww', 'xx', 'yy', 'zz']

rug_text = [rug_text_one, rug_text_two] # for hover in rug plot
colors = ['rgb(0, 0, 100)', 'rgb(0, 200, 200)']


fig = ff.create_distplot(hist_data, group_labels=['Period 2','Period 1'], bin_size=16.9, rug_text=rug_text, colors=colors)
fig.update_layout(title_text='Greece is Flattening the curve')
fig.show()


# In[20]:


#t-test,by this statistical function 
#we can provide information to each country , answering the question:'Is your country flattening the curve? '

#!pip install researchpy
import researchpy as rp
from scipy import stats

#1st Homogenity test
def Homogenity(x,y):
    p_value=stats.levene(x,y)[1]
    if p_value> 0.05:
        return print('The test is not significant meaning there is homogeneity of variances and we can proceed')
    else:
        return print('There is no Homogenity assumption')

Homogenity(df1['Total Confirm of new cases'], df2['Total Confirm of new cases'])
#p_value > 0.05 σ1=σ2

#Normality test
def Normality(x):
    if stats.shapiro(x)[1]>0.05:
        return print('The distribution is Normal')
    else:
        return print('The distribution is not Normal')

Normality(df2['Total Confirm of new cases'])
#The fisrt distribution is far from normal but the second one is normal


#t_test
def t_test(x,y):
    if stats.ttest_ind(x,y)[1]<0.05:
        return print('Greece is flattening the curve!')
t_test(df2['Total Confirm of new cases'],df1['Total Confirm of new cases'])

#By the statistical test we made observe that there is difference between the two curves and the second one which denotes 
#the after the measures period is more flatten

descriptives, results = rp.ttest(df2['Total Confirm of new cases'],df1['Total Confirm of new cases'])

descriptives


# In[21]:


#fitting the initial dataset 
plt.figure(figsize=(11,7))

plt.plot(xdata, ydata,'ko', label='data') #the original datapoints

#model1
popt, pcov = curve_fit(f=func, xdata=xdata, ydata=ydata, p0 = None, sigma = None) 

print (popt) # parameters
print (pcov) # covariance


plt.plot(xdata, func(xdata,*popt), 'g--',
         label='model1: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))


#model2
popt2, pcov2 = curve_fit(func, xdata, ydata, bounds=(0, [3., 1., 0.5]))

plt.plot(xdata, func(xdata, *popt2), 'r--',
         label='model2: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt2))


#model3
popt3, pcov3 = curve_fit(func, xdata, ydata, bounds=(0.01, [5., 4.,0.1]))

plt.plot(xdata, func(xdata, *popt3), 'b-',
         label='model3: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt3))

#model4
popt4=[3,1,10]

plt.plot(xdata, func(xdata, *popt4), 'y-',
         label='model4: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt4))



plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='upper left')


plt.show()

#By reviewing the figure we understand that the model 4 : 3 *e^x -10 make a good fit to the observed data--->popt4
#However in the tail seems that model 1 fits better : 160.862 * e^0.215*x -207.2781--->popt


# In[22]:


#fitting the 1st period¶
plt.figure(figsize=(11,7))

plt.plot(xdata[0:19], df1['Total Confirm of new cases'],'ko', label='data') #the original datapoints

#model1
popt0, pcov0 = curve_fit(f=func, xdata=xdata[0:19], ydata=df1['Total Confirm of new cases'])

#print (popt0) # parameters print (pcov0) # covariance

plt.plot(xdata[0:19], func(xdata[0:19],*popt0), 'g--', label='model1.1: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt0))

#model2 
popt02, pcov02 = curve_fit(func, xdata=xdata[0:19], ydata=df1['Total Confirm of new cases'], bounds=(0, [3., 1.,0.1]))

plt.plot(xdata[0:19], func(xdata[0:19],*popt02), 'r--', label='model1.2: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt02))

#print (popt02)

popt03=[2.43426633e-08,10.651,0]

plt.plot(xdata[0:19], func(xdata[0:19],*popt03), 'b--', label='model1.3: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt03))

plt.xlabel('x')
plt.ylabel('y') 
plt.title('Exponential fit for period 1')
plt.legend(loc='upper left')

plt.show()

#Regarding the 1st period the best fitting came from the model1.3:


# In[ ]:


plt.figure(figsize=(11,7))


plt.plot(xdata,ydata,'ko') #we add the line and the data ponts

plt.plot(xdata, ydata,'b', label='data')


plt.plot(xdata, func(xdata, *popt2), 'r--',
         label='model2: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))


plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='upper left')


plt.show()


# In[23]:


#In this part we start the evaluation of our models using Goodness of fit statistical tests

#1st_Method:Goodness of Fit Chisquare test
from scipy.stats import chisquare
import scipy

#data model 1

exp=list(func(xdata,*popt))
obs=list(ydata)

observed_values=scipy.array(obs)
expected_values=scipy.array(exp)

if scipy.stats.chisquare(observed_values, f_exp=expected_values)[1]<0.05:
    print('Model 1 does not fit to the observed values')


# In[24]:



#model 2
exp=list(func(xdata,*popt2))
obs=list(ydata)
observed_values=scipy.array(obs)
expected_values=scipy.array(exp)

if scipy.stats.chisquare(observed_values, f_exp=expected_values)[1]<0.05:
    scipy.stats.chisquare(observed_values, f_exp=expected_values)
    print('Model 2 does not fit to the observed values')


# In[25]:


#model 1 period 1
exp=list(func(xdata[0:19],*popt0))
obs=list(df1['Total Confirm of new cases'])
observed_values=scipy.array(obs)
expected_values=scipy.array(exp)

if scipy.stats.chisquare(observed_values, f_exp=expected_values)[1]<0.05:
    
    print('Model 1 in period 2 does not fit to the observed values')
    print(scipy.stats.chisquare(observed_values, f_exp=expected_values))


# In[26]:


#2nd_Method:Goodness of fit AIC, R^2


model1=smf.ols(formula='ydata~func(xdata,*popt)',data=df).fit()
model1.summary()


#The model 1 that we saw that made a good fit to the initial data explains the 54.9 % of the variance of the total cases and Aic = 376.7.


# In[27]:



model4=smf.ols(formula='ydata~func(xdata,*popt4)',data=df).fit()
model4.summary()

#The second best model regarding the previous assumptions was the model 4, as we can see it explains a bit less than the model 1
#but it has greater AIC value


# In[ ]:


#3d method:metric to evaluate the models MAPE(Mean absolute percentage error) it should be < 10%


def mean_absolute_percentage_error(y_true, y_pred): 
    for i in range(len(y_true)):
        if y_true[i]==0:
            y_true[i]= np.mean(y_true)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) /y_true)) * 100

mean_absolute_percentage_error(ydata,func(xdata,*popt4))  #The model 4 has MAPE=76.66%
mean_absolute_percentage_error(ydata,func(xdata,*popt)) #The model 1 : has MAPE=81.17%


# In[28]:


#model 2 log()
def func2(x, a, b, c):
    return a * np.log(b * x) -c


# In[29]:


#model 3
def func3(x,a,b,c,):
    return a*np.exp(-b * x) - c


# In[30]:


popt22, pcov22 = curve_fit(f=func2, xdata=xdata, ydata=ydata, p0 = None, sigma = None) 
func2(xdata,*popt22)

plt.plot(xdata, ydata,'ko', label='data')



#xx = np.linspace(1, 4, 39)
#xx=np.linspace(0,0.5,39)
plt.plot(xdata, func2(xdata,*popt22), 'g--',
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt2))


# In[31]:


#goodness of fit using R^2 for log model 
model3=smf.ols(formula='ydata~func2(xdata,*popt3)',data=df).fit()
model3.summary()


# In[33]:


#goodness of fit using R^2 for period 2
ydata2=df2['Total Confirm of new cases']
xdata2=xdata[19:39]
popt13, pcov13 = curve_fit(func, xdata2, ydata2, bounds=(0.01, [5., 4.,0.1]))
popt12, pcov12 = curve_fit(func, xdata2, ydata2, bounds=(0, [3., 1., 0.5]))


model3=smf.ols(formula='ydata2~func(xdata2,*popt12)',data=df2).fit()
#model3.summary()


# In[ ]:


#goodness of fit using R^2 for period 2 for func2
ydata2=df2['Total Confirm of new caces']
xdata2=xdata[19:39]
#popt3, pcov3 = curve_fit(func2, xdata2, ydata2, bounds=(0.01, [5., 4.,0.1]))
#popt2, pcov2 = curve_fit(func2, xdata2, ydata2, bounds=(0, [3., 1., 0.5]))
popt1, pcov1 = curve_fit(func2, xdata2, ydata2, p0 = None, sigma = None)
#popt4, pcov4 = curve_fit(func2, xdata2, ydata2, bounds=(0, [1., 6, 9]))

popt1

#model3=smf.ols(formula='ydata2~func2(xdata2,*popt1)',data=df2).fit()
#model3.summary()


# In[34]:


popt6=[194.321,1.06,109]
model6=smf.ols(formula='ydata2~func2(xdata2,*popt6)',data=df2).fit()
model6.summary()


# In[35]:


#fiiting the 2nd period in the observed data

plt.figure(figsize=(11,7))

popt12, pcov12 = curve_fit(f=func2, xdata=xdata2, ydata=ydata2, p0 = None, sigma = None) 
func2(xdata2,*popt12)

plt.plot(xdata2,ydata2,'ko', label='data')


plt.plot(xdata2, func2(xdata2,*popt12), 'g--',
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt12))

popt5=[196.792,1.06,103.983]
plt.plot(xdata2, func2(xdata2,*popt5), 'r--',
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt5))

popt6=[194.321,1.06,109]
plt.plot(xdata2, func2(xdata2,*popt6), 'b--',
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt6))

popt7=[198.19212272,0.95072867, 120.98279073]
            
plt.plot(xdata2, func2(xdata2,*popt7),
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt7))




plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='upper left')
plt.title('Logistic fit to the second period')
plt.show()

popt12


# In[39]:


#goodness of fit using R^2 for period 1
ydata1=df1['Total Confirm of new cases']
xdata1=xdata[0:19]
popt113, pcov113 = curve_fit(func, xdata1, ydata1, bounds=(0.01, [5., 4.,0.1]))
popt112, pcov112 = curve_fit(func, xdata1, ydata1, bounds=(0, [3., 1., 0.5]))
popt111, pcov111 = curve_fit(func, xdata1, ydata1)


model13=smf.ols(formula='ydata1~func(xdata1,*popt111)',data=df1).fit()
model13.summary()

#The model 13: 2.43426633e-08*e^1.14505058e+01*x - -3.74676026e+00 explains the  67.9% of the total variance


# In[ ]:


#Prediction of the num of cases on the day: 6/4/2020 !!!!
func(3.05263158,*popt2)

#prediction =63.52 when observed = 62 #popt2: a= 3, b=1, c=1.3 function:3*e^x-1.3


# In[ ]:


func2(3.05263158,*popt5) # more attempts to predict 6/4/2020 using log() popt5=[196.792, 1.06, 103.983]


# In[ ]:


func(3.60526316 ,*popt2) #  popt2 = [3.00000000e+00, 1.00000000e+00, 1.23643378e-16]  exponential model


# In[40]:


#In this part we deside to make a combination of 2 models exponential and logistic as well because in
#the 1st period exponential fits beter and in the second one log function

plt.plot(xdata2,func2(xdata2,*popt5),label='1')
plt.plot(xdata1,func(xdata1,*popt2),label='2')

plt.plot(xdata,ydata,'ko')


plt.legend()

popt2


# In[41]:



#we create the 1st sigma model sigma0 we used the best estimates regarding the previous analysis
def sigma0(x):
    z=np.zeros(39)
    for i in range(len(x)):
        if i in range(len(x)-20):
             z[i]= 3.00000000e+00 * np.exp((1.00000000e+00)*x[i])- 1.23643378e-16
        if i in range(19,39):
            z[i]=196.792* np.log( 1.06*x[i])- 103.983
    return z
plt.plot(sigma0(xdata))
plt.plot(ydata,'ko')

modelsigma0=smf.ols(formula='ydata~sigma0(xdata)',data=df).fit()
modelsigma0.summary()

#mean_absolute_percentage_error(ydata,sigma0(xdata))=86%
#AIC almost 400
#R^2 = 49.2 %


# In[44]:


#this model extracts the best information criteria
def sigma(x,a,b,c):
    z=np.zeros(39)
    for i in range(len(x)):
        if i in range(len(x)-20):
             z[i]= 2.43426633e-08 * np.exp((1.14505058e+01)*x[i]) --3.74676026e+00
        if i in range(19,39):
            z[i]=100/(1+88*np.exp(-2.1*x[i]))
            #z[i]=198.19212272 * np.log(0.95072867*x[i]-0.22) -120.98279073
    return z
plt.plot(sigma(xdata,*popt3))
plt.plot(ydata,'ko')


modelsigma=smf.ols(formula='ydata~sigma(xdata,*popt4)',data=df).fit()
modelsigma.summary()

#the next sigma models gave 67.4% R^2, which means that it can explain pretty well the variance of observed data
#AIC = 364
#BIC=367
#MAPE=48.93%


# In[53]:


#plot the fit using plotly

x1=df1['Total Confirm of new cases']
x2=sigma(xdata,*popt3)
hist_data=[x2,x1]

group_labels = ['exponential', 'logistic']

colors = ['slategray', 'magenta']
rug_text = [rug_text_one, rug_text_two] # for hover in rug plot
colors = ['rgb(0, 0, 100)', 'rgb(0, 200, 200)']



fig = px.line(df, x=xdata, y=x2, title='Sigma model')
fig.show()


# In[65]:



# we plot both obsered and expected to see the good fit!

fig = go.Figure()
fig.add_trace(go.Scatter(x=xdata, y=x1,
                    mode='lines',
                    name='actual'))
fig.add_trace(go.Scatter(x=xdata, y=x2,
                    mode='lines+markers',
                    name='expected'))


fig.update_layout(
    title="Actual vs Predicted Sigma from scratch model",
    xaxis_title="x",
    yaxis_title="Total new cases",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    )
)
fig.show()


# In[55]:


# Regarding that the max number of sick people in Greece are 100 ( c=100) a=88 and each people infects 2 people approximately
def logistic(x):
    return (100/(1+88*np.exp(-2.5*x)))
plt.plot(logistic(xdata))
plt.plot(ydata,'ko')


# In[66]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=xdata, y=x1,
                    mode='markers',
                    name='actual'))
fig.add_trace(go.Scatter(x=xdata, y=logistic(xdata),
                    mode='lines+markers',
                    name='expected'))


fig.update_layout(
    title="Actual Vs Predicted logistic Model",
    xaxis_title="x ",
    yaxis_title="Total new cases",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    )
)

fig.show()

#R^2=53.9
#AIC = 377
#MAPE >80%


# In[ ]:


#MAPE > 10% for logistic()=1941
#mean_absolute_percentage_error(xdata,logistic(xdata))


# In[67]:


#goodness of fit using R^2 for logistic()
modellog=smf.ols(formula='ydata~logistic(xdata)',data=df).fit()
modellog.summary()


# In[68]:


#Create a second logistic model this time we estimate the parameters using scipy library and curve fit
def logistic2(x,a,b,c):
    return (c/(1+a*np.exp(-b*x)))


# In[69]:



import scipy.optimize as optim

xdatanew=np.array([i for i in range(1,40,1)])
(a,b,c),cov=optim.curve_fit(logistic2,xdatanew,ydata,p0=np.random.exponential(size=3))


# In[70]:


a,b,c
#by usig this library we estimate a = 40.62337646974813, b = 0.16159761879835696 c = 98.5975157245406
#model: 98.59/1+46.62 * e^ -0.16*x


# In[71]:


def logistic2new(x):
    return (c/(1+a*np.exp(-b*x)))


# In[72]:


logistic2new(xdatanew)


# In[78]:


fig = go.Figure()
colors = ['slategray', 'magenta']
fig.add_trace(go.Scatter(x=xdata, y=x1,
                    mode='lines',
                    name='actual'))
fig.add_trace(go.Scatter(x=xdata, y=logistic2new(xdatanew),
                    mode='lines+markers',
                    name='expected'))

line_shape='hvh'
fig.update_layout(
    title="Actual Vs Predicted logistic Model evaluated from scipy",
    xaxis_title="x ",
    yaxis_title="Total new cases",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    )
)

fig.show()


# In[80]:


def sigma2(x,a,b,c):
    z=np.zeros(39)
    for i in range(len(x)):
        if i in range(len(x)-20):
             z[i]= 2.43426633e-08 * np.exp((1.14505058e+01)*x[i]) --3.74676026e+00
        if i in range(19,39):
            #z[i]=100/(1+88*np.exp(-2.1*x[i]))
            z[i]=198.19212272 * np.log(0.95072867*x[i]-0.22) -120.98279073
    return z
plt.plot(sigma2(xdata,*popt3))
plt.plot(ydata,'ko')


# In[ ]:


xdata7=np.array([i for i in range(1,40,1)])
def logistic3(x):
    return (c/(1+a*np.exp(-b*x)))
logistic3(xdata7)
plt.plot(xdata7,logistic3(xdata7))
plt.plot(ydata,'ko')


# In[ ]:


modellogistic3=smf.ols(formula='ydata~logistic3(xdata7)',data=df).fit()
modellogistic3.summary()


# In[ ]:


def logistic4(x,a,b):
    return 1/(1+np.exp(a + b*x))


# In[ ]:


#In this step we tried to estimate parameters with MCMC bayesian method
import pymc3 as pm
import theano.tensor as tt
import scipy
from scipy import optimize

y_simple = df['Cumulative Sum of new cases']
x_0=xdata
x_c = x_0 - x_0.mean()

with pm.Model() as model_simple:
    α = pm.Normal('α', mu=0, sd=10)
    β = pm.Normal('β', mu=0, sd=10)
    
        
    θ = pm.Deterministic('θ', 1. / (1. + tt.exp(β * x_c + α)))
    bd = pm.Deterministic('bd', -α/β)
    
    y_1 = pm.Bernoulli('y_1', p=θ, observed=y_simple)
    step = pm.Metropolis()
    trace_simple = pm.sample(5000, step)

#!pip install arviz
import arviz as az

az.summary(trace_simple, var_names=['α', 'β'])


# In[ ]:


#sigma(xdata,*popt3).iplot(kind='hist')
pd.DataFrame(sigma(xdata,*popt3)).iplot()


# In[ ]:


#Finaly we will try to make a 3d model and evaluate the parameters again from scipy
#From the plot below it is clear that we can extract a patern from the data points after good rotation
#The attributes are:days, Total new cases , Cumulative sum of new cases


# In[84]:


def func(x,y,a,b,c):
    return np.log(a)+ b*np.log(x)+ c*np.log(y+1) 

#print(curve_fit(func,(xdata,ydata),df['Cumulative Sum of new cases'].values))


# In[43]:


px.scatter_3d(x=df['days'],y=df['Total Confirm of new cases'],z=df['Cumulative Sum of new cases'])

