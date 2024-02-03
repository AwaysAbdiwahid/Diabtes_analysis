#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("D:/Data science/capstone project development/diabetes_012.csv")
features = df.columns
print('features count', len(features))
print("the dataset features are", features)


# In[3]:


import matplotlib.pyplot as plt
#creating a directory to save image
IMAGES_PATH = Path() / "images"/"diabetes_visualization_tools"
IMAGES_PATH.mkdir(parents=True, exist_ok = True)
def save_fig(fig_id, tight_layout = True, fig_extension = "png", resolution = 300):
  path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
  if tight_layout:
    plt.tight_layout()
  plt.savefig(path, format = fig_extension, dpi = resolution)

plt.rc("font", size = 14)
plt.rc("axes", labelsize = 14, titlesize = 14)
plt.rc("legend", fontsize = 14)
plt.rc("xtick", labelsize = 10)
plt.rc("ytick", labelsize = 10)


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


df.rename(columns = {'Diabetes_012':"Diabetes"}, inplace = True)


# In[7]:


df['HeartDiseaseorAttack'].value_counts()


# In[8]:


df.describe().T


# In[9]:


df.nunique()


# In[10]:


df.shape


# In[11]:


duplicates = df[df.duplicated()]
print(len(duplicates))


# In[12]:


df1 = df.drop_duplicates()
df1.shape


# In[13]:


df1['Diabetes'].value_counts()


# In[14]:


df1.hist(figsize = (20,15))
save_fig('diabetes histogram')


# analysing the the relationship of smoker variable with Diabetes variable in the dataset

# In[15]:


df1['Smoker'].value_counts()


# In[16]:


pd.crosstab(df1["Diabetes"], df1["Smoker"])


# In[17]:


grouped_SmokerAndDiabetes = df1.groupby(['Diabetes', 'Smoker']).size().reset_index(name='count')
color1 = "green"
color2 = "blue"
color3 = "orange"
color4 = "red"
color5 =  "pink"
color6 = "gray"

# plot results as a bar chart
plt.figure(figsize = (6, 6))
plt.bar(grouped_SmokerAndDiabetes['Diabetes'].astype('str') + "-" + grouped_SmokerAndDiabetes['Smoker'].astype("str"),
       grouped_SmokerAndDiabetes['count'], color = [color1,color2, color3, color4, color5, color6])
plt.grid(True)
plt.xlabel("Diabetes - Smoker")
plt.ylabel("count")
plt.title("the relation ship of diabetes and smoking history", fontsize = 14)
save_fig("the relationship of diabetes and smoking history")
plt.show()


# # feature selection using univariate(selectKBest, f_regression) technique and biavariate technique

# In[18]:


from sklearn.feature_selection import SelectKBest, f_regression

X = pd.DataFrame(df1, columns=df1.columns)
y = df1['Diabetes']

# Select the top 10 features using Univariate Feature Selection
k = 10
selector = SelectKBest(score_func=f_regression, k=k)
X_selected = selector.fit_transform(X, y)

# Get the indices of the selected features
selected_indices = np.argsort(selector.scores_)[::-1][:k]
selected_features = X.columns[selected_indices]

# Print the selected features
print("Selected Features:")
print(selected_features)


# In[19]:


df1.drop('Diabetes', axis=1).corrwith(df1.Diabetes).plot(kind='bar', grid=True, figsize=(12, 8)
, title="Correlation of Diabetes with other features",color="green");
save_fig("correlation of diabetes with other features")


# i applied univariate and bivariate to determine the most variables that effects the diabetes variable and i got i dentical answer i will filture out those variables that i got from the techniques i mentioned above for further analysis

# In[20]:


df12 = df1[['Diabetes','GenHlth', 'HighBP', "PhysActivity",'BMI', 'DiffWalk', 'HighChol', 'Age', 'HeartDiseaseorAttack', 'PhysHlth', 'Income', 'Education', 'Sex']]


# In[21]:


correlation = df12.corr()


# In[22]:


fig, ax = plt.subplots(figsize = (20, 10))
sns.heatmap(correlation, cmap = "coolwarm", annot = True,  ax = ax,
           annot_kws = {"fontsize": 12, "fontweight": "bold"})
save_fig("correlation of diabetes dataste")


# In[23]:


df12.shape


# In[24]:


df12['Diabetes'].value_counts()


# In[25]:


counts = df12['Diabetes'].value_counts()
max_index = counts.argmax()
explode = [0.1 if i == max_index else 0 for i in range(len(counts))]

labels = ['Non-Diabetes', 'Diabetes','prediabetes']
values = [counts[0], counts[1]]
plt.pie(counts,
        labels = labels,
        explode = explode,
        autopct = "%1.1f%%")
plt.title("the percentage people of none diabetes, prediabetes and diabetes")
save_fig("pie chart for diabetes and non diabetes")
plt.show()


# 0 stands for female and 1 is for male

# In[26]:


df12['Sex'].value_counts()


# In[27]:


pd.crosstab(df12["Diabetes"], df12["Sex"])


# In[28]:


diabetes = df12[(df12['Diabetes']==1) | (df12['Diabetes'] == 2)]


# In[29]:


pd.crosstab(diabetes["Diabetes"], diabetes["Sex"])


# In[30]:


pd.crosstab(df1["Diabetes"], df1["HighChol"])


# In[31]:


pd.crosstab(df1["Diabetes"], df1["HighBP"])


# In[32]:


pd.crosstab(df1["Diabetes"], df1["Education"])


# In[33]:


pd.crosstab(df1["Diabetes"], df1["Income"])


# In[34]:


counts = diabetes['Sex'].value_counts()
max_index = counts.argmax()
explode = [0.1 if i == max_index else 0 for i in range(len(counts))]
colors = ["green", "yellow"]
labels = ['female with diabetes and prediabetes','male with diabetes and prediabetes']
values = [counts[0], counts[1]]
plt.pie(counts,
        labels = labels,
        colors = colors,
        autopct = "%1.1f%%")
plt.title("the Average of Diabetes across female and male")
save_fig("diabetes femela VS male")
plt.show()


# In[35]:


diabetes['Sex'].value_counts()


# In[36]:


diabet = diabetes.copy()

diabet.loc[:,'Sex'] = diabetes['Sex'].replace({0:"Female", 1:"Male"})

diabet.loc[:,'Diabetes'] = diabetes['Diabetes'].replace({1:"Prediabetes", 2:"Diabetes"})

female_and_male=pd.crosstab(index=diabet['Diabetes'],columns=diabet['Sex'])

#female_and_male = female_and_male.replace({0: "Female", 1:"Male"})

ax= female_and_male.plot.bar(figsize = (5,5), rot = 0)
ax.grid(True)
ax.set_ylabel("count")
save_fig("diabetes and prediabetes of male and female")


# In[34]:


sex_by_diabete = diabet.groupby(['Sex', "Diabetes"]).size()
sex_by_diabete.plot(kind = "pie", autopct = "%1.1f%%")
save_fig("diabetes and prediabetes per Sex")
plt.show()


# describing which gender is most at risk with diabetes compared to other gender

# In[35]:


Diabetes_Sex = df12[['Diabetes', 'Sex']]

Males = Diabetes_Sex[Diabetes_Sex['Sex'] == 1]
Females = Diabetes_Sex[Diabetes_Sex['Sex'] == 0]

males_diabetes_prediabetes = Diabetes_Sex[(Diabetes_Sex['Sex'] == 1) & ((Diabetes_Sex['Diabetes'] == 1) |
                                                                        (Diabetes_Sex['Diabetes'] == 2))]
females_diabetes_prediabetes = Diabetes_Sex[(Diabetes_Sex['Sex'] == 0) & ((Diabetes_Sex['Diabetes'] == 1) | 
                                                                          (Diabetes_Sex['Diabetes'] == 2))]


males_risk_at_diabetes = len(males_diabetes_prediabetes)/len(Males)
females_risk_at_diabetes = len(females_diabetes_prediabetes)/len(Females)


if males_risk_at_diabetes > females_risk_at_diabetes:
    print("males are at higher risk for diabetes compared to the women")
elif females_risk_at_diabetes > males_risk_at_diabetes:
    print("females are at higher risk for diabetes as compared to the men")
else:
    print("both has equal chance of having diabetes")  


# setting the likelyhood for diabetes in a chart for good understanading

# In[36]:


#define the x-axis labeles
labels = ["males", "females"]
#define the y-axis values
risk = [males_risk_at_diabetes, females_risk_at_diabetes]

colors = ["green", "orange"]
#create the bar chart

plt.bar(labels, risk, color = colors)

#the title and labels to the chart
plt.grid(True)
plt.title("Risk of diabetes by gender")
plt.xlabel("Gender")
plt.ylabel("Risk of the diabetes")
save_fig("risk of diabetes by gender")
plt.show()


# In[37]:


diabetes['BMI'].describe()


# In[38]:


diabetes_2 = diabetes.copy()
def categories_bmi(bmi):
    if bmi < 18.5:
        return 'UnderWeight'
    elif 18.5 <= bmi < 24.9:
        return 'NormalWeight'
    elif 25 < bmi < 29.9:
        return "OverWeight"
    elif bmi > 30:
        return "Obesity"
diabetes_2['BMI category'] = diabetes_2['BMI'].apply(categories_bmi)


# In[39]:


diabetes_2['Diabetes'].replace({2: "Diabetes",1: "Prediabetes"}, inplace = True)
(diabetes_2['BMI category'].value_counts())


# In[40]:


17500 + 2600


# In[41]:


counts = diabetes_2.groupby(['Diabetes', 'BMI category']).size().reset_index(name = "counts")

bmi_order = ['UnderWeight', 'NormalWeight', 'OverWeight', 'Obesity']

counts['BMI category'] = pd.Categorical(counts['BMI category'], categories=bmi_order, ordered=True)

counts = counts.sort_values('BMI category', ascending = False)
fig, ax = plt.subplots(figsize = (7, 3))
color = "green", "orange"
ax = counts.pivot(index = "BMI category", columns = 'Diabetes', values = 'counts').plot(
    kind = "barh", stacked = True, ax = ax, color = color, grid = True)
ax.set_title("the Diabetes and Prediabetes with BMI category")
save_fig('BMI by Diabetes')
plt.show()


# In[42]:


counts = diabetes_2.groupby(['Diabetes', 'Age']).size().reset_index(name = "counts")


counts['Age'] = pd.Categorical(counts['Age'],ordered=True)

#counts = counts.sort_values('Age', ascending = False)
fig, ax = plt.subplots(figsize = (8, 6))
color = "green", "orange"
ax = counts.pivot(index = "Age", columns = 'Diabetes', values = 'counts').plot(
    kind = "bar", stacked = True, ax = ax, color = color, grid = True)
ax.set_ylabel("counts")
ax.set_title("the Diabetes and Prediabetes with Age group")
save_fig('Age Group by Diabetes')
plt.show()


# In[43]:


diabetes['Age'].value_counts()


# In[44]:


df1['Age'].value_counts()


# In[45]:


diabetes_2['Diabetes'].replace({2: "Diabetes",1: "Prediabetes"}, inplace = True)


# In[46]:


counts = diabetes_2.groupby(['Diabetes', 'GenHlth']).size().reset_index(name = "counts")


#counts['Age'] = pd.Categorical(counts['Age'],ordered=True)

#counts = counts.sort_values('Age', ascending = False)
fig, ax = plt.subplots(figsize = (8, 6))
color = "green", "orange", "blue"
ax = counts.pivot(index = "GenHlth", columns = 'Diabetes', values = 'counts').plot(
    kind = "bar", stacked = True, ax = ax, color = color )
ax.set_title("the Diabetes and Prediabetes with Age group")
save_fig('diabetes and general health')
plt.show()


# In[47]:


diabetes_2['GenHlth'].value_counts()


# In[48]:


diabet_heart_diseas = diabetes.copy()

diabet_heart_diseas.loc[:,'HeartDiseaseorAttack'] = diabet_heart_diseas['HeartDiseaseorAttack'].replace({0:"No heartattack", 1:"heartattack"})

diabet_heart_diseas.loc[:,'Diabetes'] = diabet_heart_diseas['Diabetes'].replace({1:"Prediabetes", 2:"Diabetes"})

Blood_pressure=pd.crosstab(index=diabet_heart_diseas['HeartDiseaseorAttack'],columns=diabet_heart_diseas['Diabetes'])

#female_and_male = female_and_male.replace({0: "Female", 1:"Male"})

ax= Blood_pressure.plot.bar(figsize = (5,5), rot = 0)
ax.grid(True)
ax.set_ylabel("count")
save_fig("diabetes and Heart disease")


# In[37]:


blood_presure = diabetes.copy()

blood_presure.loc[:,'HighBP'] = blood_presure['HighBP'].replace({0:"No highBp", 1:"yes highBp"})

blood_presure.loc[:,'Diabetes'] = blood_presure['Diabetes'].replace({1:"Prediabetes", 2:"Diabetes"})

Blood_pressure=pd.crosstab(index=blood_presure['HighBP'],columns=blood_presure['Diabetes'])

#female_and_male = female_and_male.replace({0: "Female", 1:"Male"})

ax= Blood_pressure.plot.bar(figsize = (5,5), rot = 0)
ax.grid(True)
ax.set_ylabel("count")
save_fig("diabetes and HighBP")


# In[38]:


blood_presure = diabetes.copy()

blood_presure.loc[:,'HighChol'] = blood_presure['HighChol'].replace({0:"No high cholestrol", 1:"highBp cholestrol"})

blood_presure.loc[:,'Diabetes'] = blood_presure['Diabetes'].replace({1:"Prediabetes", 2:"Diabetes"})

Blood_pressure=pd.crosstab(index=blood_presure['HighChol'],columns=blood_presure['Diabetes'])

#female_and_male = female_and_male.replace({0: "Female", 1:"Male"})

ax= Blood_pressure.plot.bar(figsize = (5,5), rot = 0)
ax.grid(True)
ax.set_ylabel("count")
save_fig("diabetes and High cholestrol")


# In[50]:


#comparing the people with diabetes and heart disease and the people no heart disease and diabetes
Diabetes_heartdisease = df1[["Diabetes", "HeartDiseaseorAttack"]]

#selecting the entries of heart disease
heart_disease_attack = Diabetes_heartdisease[Diabetes_heartdisease['HeartDiseaseorAttack'] == 1]

#the entries of no heart disease
no_heart_disease = Diabetes_heartdisease[Diabetes_heartdisease['HeartDiseaseorAttack'] == 0]

#the people with heart disease and diabetes
diabetes_heartdisease = Diabetes_heartdisease[(Diabetes_heartdisease['HeartDiseaseorAttack'] == 1) 
                                              & ((Diabetes_heartdisease['Diabetes'] == 1)|                                        
                                                 (Diabetes_heartdisease['Diabetes'] == 2))]

#the people with no heart disease but with diabetes

diabetes_no_heartdisease = Diabetes_heartdisease[(Diabetes_heartdisease['HeartDiseaseorAttack'] == 0) 
                                                 & ((Diabetes_heartdisease['Diabetes'] == 1)|                                                
                                                    (Diabetes_heartdisease['Diabetes'] == 2))]

diabetes_and_heartdisease = len(diabetes_heartdisease)/len(heart_disease_attack)

diabetes_and_no_heartdisease = len(diabetes_no_heartdisease)/len(no_heart_disease)

#ploting the result for sipmle understanding

#define the x-axis labeles
labels = ["heart disease or attack", "no heart disease or attack"]
#define the y-axis values
risk = [diabetes_and_heartdisease, diabetes_and_no_heartdisease]

colors = ["green", "orange"]
#create the bar chart

plt.bar(labels, risk, color = colors)

#the title and labels to the chart
plt.grid(True)
plt.title("the heart disease and diabetes")
plt.xlabel("Heart disease")
plt.ylabel("heart disease")
save_fig("heart disease plot")
plt.show()


# In[51]:


print(diabetes_and_heartdisease)
print(diabetes_and_no_heartdisease)
#print(len(diabetes_heartdisease))
#print(len(diabetes_no_heartdisease))


# In[52]:


len(heart_disease_attack)
len(no_heart_disease)


# In[53]:


len(diabetes_heartdisease)


# the comparison of diabetes and physical activity

# In[ ]:





# In[54]:


physicalactivity_and_diabetes = diabetes.copy()

physicalactivity_and_diabetes.loc[:,'PhysActivity'] = physicalactivity_and_diabetes['PhysActivity'].replace({0:"No physical activity", 1:"Physical activity"})

physicalactivity_and_diabetes.loc[:,'Diabetes'] = physicalactivity_and_diabetes['Diabetes'].replace({1:"Prediabetes", 2:"Diabetes"})

physical_activity=pd.crosstab(index=physicalactivity_and_diabetes['PhysActivity'],columns=physicalactivity_and_diabetes['Diabetes'])

#female_and_male = female_and_male.replace({0: "Female", 1:"Male"})

ax= physical_activity.plot.bar(figsize = (5,5), rot = 0)

ax.set_ylabel("count")
save_fig("diabetes over physical activity")


# In[67]:


education_and_diabetes = df12.copy()


education_and_diabetes.loc[:,'Diabetes'] = education_and_diabetes['Diabetes'].replace({0: "no diabetes", 1:"Prediabetes", 2:"Diabetes"})

education_diabete=pd.crosstab(index=education_and_diabetes['Education'],columns=education_and_diabetes['Diabetes'])



ax= education_diabete.plot.bar(figsize = (5,5), rot = 0)
ax.grid(True)
ax.set_ylabel("count")
save_fig("diabetes over education")


# In[68]:


Income_and_diabetes = df12.copy()

#physicalactivity_and_diabetes.loc[:,'Education'] = physicalactivity_and_diabetes['PhysActivity'].replace({0:"No physical activity", 1:"Physical activity"})

Income_and_diabetes.loc[:,'Diabetes'] = Income_and_diabetes['Diabetes'].replace({0: "no diabetes", 1:"Prediabetes", 2:"Diabetes"})

Income_diabete=pd.crosstab(index=Income_and_diabetes['Income'],columns=Income_and_diabetes['Diabetes'])

#female_and_male = female_and_male.replace({0: "Female", 1:"Male"})

ax= Income_diabete.plot.bar(figsize = (5,5), rot = 0)
ax.grid(True)
ax.set_ylabel("count")
save_fig("diabetes over Income")


# In[ ]:




