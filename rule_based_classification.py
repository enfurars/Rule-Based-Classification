# Setting up the working environment

import pandas as pd 
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)


# Step 1: Reading the "persona.csv" file and showing general information about the dataset.

file_path = r'C:\Users\enfur\Desktop\miuul\KuralTabanlSnflandrma-230620-161541\persona.csv'
df = pd.read_csv(file_path)

def general_info(dataframe, head=10):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

general_info(df)


# Step 2: Function to check the number of unique types and sales counts and frequency of each type for a column. 

def field_info(col):
    print("########################################")
    print("Total unique " + str(col) + " types: " + str(df[col].nunique()))
    print("########################################")
    print("Sales Counts:\n" + str(df[col].value_counts()))
    print("########################################")
    print("Frequencies:\n" + str(df[col].value_counts() * 100 / len(df)))

field_info("SOURCE")

field_info("PRICE")

field_info("COUNTRY")


# Step 3: Analyzing the price data broken down by different features of the dataset using the 'groupby' operation.

df.groupby("COUNTRY").agg({"PRICE": "sum"})

df.groupby("COUNTRY").agg({"PRICE": "mean"})

df.groupby("SOURCE").agg({"PRICE": "mean"})

df.groupby(["SOURCE", "COUNTRY"]).agg({"PRICE": "mean"})


# Step 4: Calculating the average revenues for COUNTRY, SOURCE, SEX, and AGE groups.

df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"})


# Step 5: Sorting the output by PRICE in descending order to better visualize the output.

agg_df = df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"}).sort_values(by="PRICE", ascending=False)


# Step 6: Converting the index names to variable names.
# In the output of the previous step, all variables except PRICE are index names. Let's fix it.

agg_df.reset_index(inplace=True)


# Step 7: Converting the AGE variable to a categorical variable and adding it to agg_df.
# Creating intervals that will be used for categorizing the AGE variable.

age_intervals = [0, 18, 23, 30, 40, agg_df["AGE"].max()]
age_labels = ['0_18', '19_23', '24_30', '31_40', '41_' + str(agg_df['AGE'].max())]

agg_df['AGE_CAT'] = pd.cut(agg_df['AGE'], bins=age_intervals, labels=age_labels)


# Step 8: Define and add new level-based customers to the dataset.
# Defining a variable called "customers_level_based" and adding this variable to the dataset.

"""Warning!
 After creating customers_level_based values using list comprehension, it is necessary to singularize these values.
 For example, there might be multiple instances of the following: USA_ANDROID_MALE_0_18
 These values need to be grouped and then the average prices need to be taken. """ 

agg_df['customers_level_based'] = [f"{str(row['COUNTRY']).upper()}_{str(row['SOURCE']).upper()}_{str(row['SEX']).upper()}_{str(row['AGE_CAT']).upper()}" for _, row in agg_df.iterrows()]

final_df = agg_df.groupby('customers_level_based').agg({"PRICE": "mean"})
final_df.reset_index(inplace=True)


# Step 9: Segmenting the new customers based on PRICE
# Adding the segments to agg_df with the name "SEGMENT"
# Describing the segments 

final_df["SEGMENT"] = pd.qcut(final_df["PRICE"], 4, ['A', 'B', 'C', 'D'])

final_df.groupby("SEGMENT").agg({"PRICE": ["max", "sum", "mean"]}).reset_index()


# Step 10: Classifying new incoming customers and predicting how much revenue they could generate.

def new_customer_predicter(age, sex, country, source):
    if age <= 18:
        age = "0_18"
    elif age <= 23:
        age = "19_23"
    elif age <= 30:
        age = "24_30"
    elif age <= 40:
        age = "31_40"
    elif age <= 70:
        age = "41_70"

    col_value = f"{country.upper()}_{source.upper()}_{sex.upper()}_{age.upper()}"
    print(final_df[final_df["customers_level_based"] == col_value])

# Example1: Determine which segment a 33-year-old male Turkish customer using Android belongs to, and estimate the average revenue they could generate.
new_customer_predicter(33, 'male', 'TUR', 'android')
# Example2: Determine which segment a 35-year-old female French customer using IOS belongs to, and estimate the average revenue they could generate.
new_customer_predicter(35, 'female', 'fra', 'IOS')
