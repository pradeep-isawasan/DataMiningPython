---
marp: true
theme: gaia
author: pradeep 
size: 16:9
title: "Slides"
---

<!-- _class: lead -->

# Basic Data Mining with Python
![w:70 h:60](Mypic.png)pradeep

---
<!-- paginate: true -->
<!-- footer: ![w:180 h:40](uitm-hitel.png) HITeL Innovative T&L Webinar Series 2022 -->

# What is Data Mining?

> Data Mining is a process of extracting insights from data.

---
# Technology/Tools
Programming Language
- Python, R

Software
- Weka, RapidMiner, Excel

Cloud
- R Studio Cloud, Power BI, Tableau, Google Collab

---
# What problem to solve?
Classification - Supervised Learning

Clustering - Unsupervised

---
# Methodology
Apa tu


---
# What is Data?

---
# Python
What
- High-Level Programming Language.
- Emphasizes on code readibilty.
- Rank = 1* for 2021 *[(IEEE Spectrum)](https://spectrum.ieee.org/top-programming-languages/)
- Consist of fantastic libraries!

---

# Python Libraries
A Python library is a collection of related modules. It makes Python Programming simpler and convenient for the programmer. 
```py
# Pandas
import pandas as pd
```

---
# Iris Dataset


---
# Reading Data

Usually we can use pandas library. Pandas store the imported data as DataFrame.

```py
# Default sep = ','
df = pd.read_csv("file_name.csv")
```
or if you want to use another separator, simply add sep='\t'

```py
df = pd.read_csv("file_name.csv", sep = '\t')
```

---

# View Data

You can have a look at the first five rows with .head():

```py
# by default is 5 rows
df.head()
# you can also customize the #-rows 
df.head(10)
```
or the last five rows with .tail():

```py
df.tail()
```

---

# Data Info

The shape property returns a tuple representing the dimensionality of the DataFrame. 

```py
df.shape
```

The info() method prints information about the DataFrame. 

```py
df.info
```

---

# Statistical Description

All standard statistical operations are present in Pandas:

```py
# Show the statistical summary on the numerical columns
df.describe()
# or individually
df.mean()
```


```py
# Show the statistical summary on the categorical columns
df.describe(include = 'object')
```
---

# Data Cleaning
###### Finding Missing Values

It is common to have not-a-number (NaN) values in your data set. 

```py
# Will give the total number of NaN in each column
df.isna().sum()
```

---
# Data Cleaning
###### Handling Missing Values

```py
# Remove the rows with NaN, not recommended
df.dropna() 
```
```py
# fill NaN with 0, also not recommended
df.fillna(0) 

# fill NaN with mean, better
df.fillna(df.mean(numeric_only=True)) 
```

---

# Data Cleaning
###### Problematic Values

Typo can be considered problematic.

```py
# View problematic values
df.iloc[[7]]

# Count unique categorical values
df.Species.unique()
df['Species'].value_counts()
```

---

# Data Cleaning
###### Handling Problematic Values

We can replace with the correct value using replace()

```py
df.replace(['SETSA'],'setosa')
```

Cleaning done! 

I have provided more detailed data cleaning in this [Kaggle post](https://www.kaggle.com/code/pradeepisawasan/handling-unusual-values).

---
# Data Visualization

We can replace with the correct value using replace()

```py
df.replace(['SETSA'],'setosa')
```

You can read more about data cleaning in this [Kaggle post](https://www.kaggle.com/code/pradeepisawasan/handling-unusual-values)