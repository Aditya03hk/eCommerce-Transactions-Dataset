import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
customers = pd.read_csv('Customers.csv')
transactions = pd.read_csv('Transactions.csv')

# Aggregate transaction data per customer
customer_transactions = transactions.groupby('CustomerID').agg({'TotalValue': 'sum', 'Quantity': 'sum'}).reset_index()
customer_data = pd.merge(customers, customer_transactions, on='CustomerID', how='left').fillna(0)

# Feature selection and scaling
features = ['TotalValue', 'Quantity']
scaler = StandardScaler()
X = scaler.fit_transform(customer_data[features])

# Apply KMeans clustering
optimal_clusters = 3
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
customer_data['Cluster'] = kmeans.fit_predict(X)

# Evaluate using Davies-Bouldin Index
db_index = davies_bouldin_score(X, customer_data['Cluster'])
print(f"Davies-Bouldin Index: {db_index:.2f}")

# Save clustering report to PDF
from fpdf import FPDF

pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.multi_cell(0, 10, f"Customer Clustering Report\n\nNumber of clusters: {optimal_clusters}\nDavies-Bouldin Index: {db_index:.2f}\n")
pdf.output("Aditya_Harikantra_Clustering.pdf")

# Visualization of clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x=customer_data['TotalValue'], y=customer_data['Quantity'], hue=customer_data['Cluster'], palette='viridis')
plt.title('Customer Segmentation')
plt.xlabel('Total Value')
plt.ylabel('Quantity')
plt.legend(title='Cluster')
plt.grid()
plt.show()
