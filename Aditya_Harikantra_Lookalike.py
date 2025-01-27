import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Load data
customers = pd.read_csv('Customers.csv')
transactions = pd.read_csv('Transactions.csv')

# Aggregate customer transaction data
customer_spending = transactions.groupby('CustomerID').agg({'TotalValue': 'sum', 'Quantity': 'sum'}).reset_index()

# Merge customer profiles with aggregated transaction data
customer_profiles = pd.merge(customers, customer_spending, on='CustomerID', how='left').fillna(0)

# Feature scaling and similarity calculation
scaler = StandardScaler()
X = scaler.fit_transform(customer_profiles[['TotalValue', 'Quantity']])

similarity_matrix = cosine_similarity(X)
similarity_df = pd.DataFrame(similarity_matrix, index=customer_profiles['CustomerID'], columns=customer_profiles['CustomerID'])

# Generate lookalikes for the first 20 customers
lookalike_dict = {}
for customer in customer_profiles['CustomerID'][:20]:
    similar_customers = similarity_df[customer].sort_values(ascending=False)[1:4].items()
    lookalike_dict[customer] = list(similar_customers)

# Save results to CSV
lookalike_df = pd.DataFrame(lookalike_dict.items(), columns=['CustomerID', 'Lookalikes'])
lookalike_df.to_csv("Aditya_Harikantra_Lookalike.csv", index=False)

print("Lookalike recommendations saved.")
