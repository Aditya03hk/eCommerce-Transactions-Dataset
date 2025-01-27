import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
customers = pd.read_csv('Customers.csv')
products = pd.read_csv('Products.csv')
transactions = pd.read_csv('Transactions.csv')

# Data Overview
print(customers.head())
print(products.head())
print(transactions.head())

# Customer distribution by region
plt.figure(figsize=(8, 6))
sns.countplot(x='Region', data=customers)
plt.title('Customer Distribution by Region')
plt.xticks(rotation=45)
plt.show()

# Revenue per product category
merged_df = pd.merge(transactions, products, on='ProductID')
category_revenue = merged_df.groupby('Category')['TotalValue'].sum().sort_values(ascending=False)
category_revenue.plot(kind='bar', title="Revenue by Product Category", figsize=(10, 5))
plt.ylabel("Total Revenue (USD)")
plt.show()

# Save EDA PDF report
from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Exploratory Data Analysis (EDA)', 0, 1, 'C')

pdf = PDF()
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.multi_cell(0, 10, "EDA Report: Key Insights\n\n1. Most customers come from Europe.\n2. Electronics contribute the highest revenue.\n3. Some customers contribute a large portion of total revenue.\n")
pdf.output("Aditya_Harikantra_EDA.pdf")

