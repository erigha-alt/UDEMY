#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 15:36:57 2023

@author: user
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

udemy_dataset = pd.read_csv("/Users/user/Downloads/udemy_courses 2.csv")

#printing the length of the data
print("number of datapoint:",len(udemy_dataset))

#Data cleaning
#print the head of the csv
print(udemy_dataset.head)

#print the tail of the csv
print(udemy_dataset.tail)

#print the info of the csv 
print(udemy_dataset.info)

#print the discription of the csv
print(udemy_dataset.describe)

# droping unnecessary columns
udemy = udemy_dataset.drop(['course_id', 'url'], axis = 1)
# cheching the null values in the dataset
print(udemy.isnull())

#printing the null values in each row
print(udemy.isnull().sum())

#printing the total null values in the csv
print(udemy.isnull().sum().sum())

#printing out the unique values in each column
print(udemy.nunique())

#printing out duplicated row if there is any
print(udemy.duplicated())



# count of paid and free
is_paid = udemy_dataset['is_paid'].value_counts()
label=['true','false']
plt.pie(is_paid.values,labels=label,autopct='%0.2f%%',shadow=True,explode=[0,0.2])
plt.title('count of paid and free')
plt.show()


# subscribers for paid and free course
num_sub=udemy.groupby('is_paid')['num_subscribers'].sum().sort_values(ascending=False)
label=['true','false']
plt.pie(num_sub.values,labels=label,autopct='%0.2f%%',shadow=True,explode=[0,0.1])
plt.title('Subscribers for Free or Paid Course')
plt.show()

# number of course per subject
sub = udemy['subject'].value_counts()
plt.xlabel('Subject')
plt.ylabel('Number of Courses')
plt.title('Number of Courses Per Subjects')
plt.bar(sub.index,sub.values,width=0.4)
plt.xticks(fontsize=8)
plt.show()

# number of course per level
level = udemy['level'].value_counts()
plt.xlabel('Levels')
plt.ylabel('Number of Courses')
plt.title('Number of Courses Per Levels')
plt.bar(level.index,level.values,width=0.4)
plt.xticks(fontsize=8)
plt.show()


# which course has more lectures
num_lec=udemy.groupby('is_paid')['num_lectures'].sum().sort_values(ascending=False)
label=['true','false']
fig,ax=plt.subplots(figsize=(2,3))
ax.bar(label,num_lec.values,width=0.4)
plt.ylabel('Number of Lectures')
plt.xlabel('Type of Course')
plt.show()

#distribution of the is_paid variable for each level 
levels = udemy['level'].unique()

for level in levels:
    is_paid = udemy[udemy['level'] == level].groupby('is_paid').size()
    num_slices = len(is_paid)
    explode = [0.1] + [0] * (num_slices - 1)  # Separate the first slice from the center
    plt.pie(is_paid, labels=is_paid.index, autopct='%0.1f%%', explode=explode, startangle=180)
    plt.legend(['true', 'false'], title='is_paid')
    plt.title(f'Distribution of is_paid for {level} Level')
    plt.show()


#Top 20 Courses with the highest Number of Subscribers: Paid vs Free
top_20_subscribers = udemy.nlargest(20,'num_subscribers')
plt.figure(figsize=(8, 6))
# Plotting paid courses
plt.barh(top_20_subscribers[top_20_subscribers['is_paid'] == True]['course_title'],
         top_20_subscribers[top_20_subscribers['is_paid'] == True]['num_subscribers'], color='orange', label='Paid')
# Plotting free courses
plt.barh(top_20_subscribers[top_20_subscribers['is_paid'] == False]['course_title'],
         top_20_subscribers[top_20_subscribers['is_paid'] == False]['num_subscribers'], color='blue', label='Free')
plt.xlabel('Number of Subscribers')
plt.ylabel('Course Title')
plt.title('Top 20 Most Popular Courses by Number of Subscribers: Paid vs Free')
plt.legend()
plt.gca().invert_yaxis()  # Invert the y-axis to have the course with the highest subscribers at the top
plt.tight_layout()

plt.show()





#Top 20 Courses with Highest Number of Reviews: Paid vs Free
top_20_reviews = udemy.nlargest(20, 'num_reviews')

plt.figure(figsize=(10, 6))

# Plotting paid courses
plt.barh(top_20_reviews[top_20_reviews['is_paid'] == True]['course_title'],
         top_20_reviews[top_20_reviews['is_paid'] == True]['num_reviews'], color='orange', label='Paid')

# Plotting free courses
plt.barh(top_20_reviews[top_20_reviews['is_paid'] == False]['course_title'],
         top_20_reviews[top_20_reviews['is_paid'] == False]['num_reviews'], color='blue', label='Free')

plt.xlabel('Number of Reviews')
plt.ylabel('Course Title')
plt.title('Top 20 Courses with Highest Number of Reviews: Paid vs Free')
plt.legend()
plt.gca().invert_yaxis()  # Invert the y-axis to have the course with the highest reviews at the top
plt.tight_layout()

plt.show()




# Convert 'published_timestamp' column to datetime
#number of courses posted per year
udemy['published_timestamp'] = pd.to_datetime(udemy['published_timestamp'])
udemy['published_year'] = udemy['published_timestamp'].dt.year
year_counts = udemy['published_year'].value_counts()
year_counts = year_counts.sort_index()
# Plotting the data
plt.figure(figsize=(10, 6))
plt.bar(year_counts.index, year_counts.values)
plt.xlabel('Year')
plt.ylabel('Number of Courses')
plt.title('Number of Courses Posted per Year')
plt.grid(True)
plt.show()

# count of subject
subject = udemy['subject'].value_counts()
#label=[subject]
plt.pie(subject.values,labels=subject.index,autopct='%0.2f%%',shadow=True)
plt.title('count of subject')
plt.show()

# Group data by level and calculate average price for each level and payment status
level_price = udemy.groupby(['level', 'is_paid'])['price'].mean().reset_index()

# Pivot the data for plotting
pivot_data = level_price.pivot(index='level', columns='is_paid', values='price')

# Plotting the data
pivot_data.plot(kind='bar', stacked=True, color= ['blue', 'orange'])
plt.xlabel('Level')
plt.ylabel('Average Price')
plt.title('Average Price of Paid and Free Courses by Level')
plt.xticks(rotation=45)
plt.legend(title='is_paid', labels=['Free', 'Paid'])
plt.tight_layout()
plt.show()


#outlier elimination
udemy=udemy[(udemy['num_subscribers']<2600)]
#Creating Five stars based bins
bins = ['-1','0','120','920','2200','2600']
labels = ['0','120','920','2200','2500']
udemy["Subscribers"] = pd.cut(udemy["num_subscribers"], bins=bins, labels=labels)
#outlier elimination
udemy=udemy[(udemy['num_lectures']<400)]
udemy=udemy[(udemy['content_duration']<40)]


# Initialize the label encoder
encode = LabelEncoder()
udemy['level'] = encode.fit_transform(udemy['level'])
udemy['subject'] = encode.fit_transform(udemy['subject'])
udemy['course_title'] = encode.fit_transform(udemy['course_title'])


udemy.content_duration=udemy['content_duration'].apply(lambda x: int(np.floor(x)))

#showing the correlations of each of the columns with price
def correlation_heatmap(df):
    _,ax=plt.subplots(figsize=(8,10))
    colormap=sns.diverging_palette(220,10,as_cmap=True)
    sns.heatmap(udemy.corr(),annot=True,cmap=colormap)

correlation_heatmap(udemy)
# Drop irrelavant columns
udemy.drop(["published_timestamp", "Subscribers"],axis=1,inplace=True)


x = udemy[["num_lectures", "subject", "level", "num_subscribers", "num_reviews"]]
y = udemy["is_paid"]
# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Initialize and train logistic regression model

model = LogisticRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print("ACCURACY:", accuracy)
print("=" * 50)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualize the confusion matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# calculating precision, recall, and f1 score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)









