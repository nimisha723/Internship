#!/usr/bin/env python
# coding: utf-8

# # DATA SCIENCE AND BUSINESS ANALYTICS TASK

#  ## TASK 1: SUPERVISED MACHINE LEARNING 
# 

# ## SIMPLE LINEAR REGRESSION

# ##  To predict the percentage of scores in accordance the amount of hours a student studies

# #### Step 1 : Installing the required Python Libraries

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# #### Step 2 : Reading the data

# In[117]:


data1="https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
print("Data Imported Successfully")

data=pd.read_csv(data1)


# In[86]:


np.shape(data)


# In[114]:


data.describe()


# In[118]:


#dropping if any null values present along the rows

data.dropna(axis=0)
data.head()


# #### Step 3 : Plotting the data to find how scattered the data is 

# In[7]:


plt.scatter(x=data.Hours,y=data.Scores)
plt.title("Hours vs Percentage Graph")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()


# ### The graph clearly shows a positive relationship in between the two variables.

# ## A) Preparing the data

# #### Step 4 : Now we select our feature variable and our target value 
# #### < In this case 'Hours' is our feature variable and 'Scores' is our target value >

# In[60]:


y=data.Scores
features=['Hours']
X=data[features]


# #### Step 5 : In this step we divide our data for training and testing 

# In[125]:


from sklearn.model_selection import train_test_split

train_X,test_X,train_y,test_y=train_test_split(X,y,test_size=0.2,random_state=0)


# ## B) Training the Algorithm : LinearRegression

# In[126]:


from sklearn.linear_model import LinearRegression
 
#making model    
regressor= LinearRegression()

#fitting the model
regressor.fit(train_X,train_y)

print("Training Complete")


# #### Step 6 : We plot the regression line on the basis of our trained data

# In[127]:


regressorline= regressor.coef_*X+regressor.intercept_

plt.scatter(X,y)
plt.plot(X,regressorline,color='orange')
plt.show()

#Coefficient and Intercept in the simple linear regression, are the parameters of the fit line.

print(regressor.coef_)
print(regressor.intercept_)


# ## C) Prediction

# In[128]:


#predicting the values

y_predictions= regressor.predict(test_X)
print(y_predictions)


# #### Step 7: Comparing the predicted ones with the original Values

# In[129]:


val=pd.DataFrame({'Actual':test_y, 'Predicted':y_predictions})
val


# #### Step 8 : Calculating value for 9.25 hours

# In[130]:


hours= 9.25
hours=np.array([[hours]],)
pre= regressor.predict(hours)
print(" Number of hours a student devotes:{}".format(hours))
print("Amount of scores a student will score:{}".format(pre))


# ## D) Evaluating The Model

# #### Step 9 : Checking how accurate the model has been 

# In[132]:


from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score


print("Mean Absolute Error for the model is=",mean_absolute_error(test_y,y_predictions))
print("Mean Squared Error for the model is=",np.sqrt(mean_squared_error(test_y,y_predictions)))
print("The r-squared value for the data is=",r2_score(test_y,y_predictions))


# ### Since the r2_score value for the data comes out to be approx 0.9 the model is accurate enough to make further predictions 

# # Thank you!
