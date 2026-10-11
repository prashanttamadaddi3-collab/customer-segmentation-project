from pathlib import Path
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
ROOT=Path(__file__).resolve().parent
df=pd.read_csv(ROOT/'data'/'customer_data.csv')
features=['age','annual_income_inr','purchase_frequency_12m','average_order_value_inr','total_spend_12m_inr','days_since_last_purchase','return_rate']
X=StandardScaler().fit_transform(df[features]); model=KMeans(n_clusters=4,random_state=42,n_init=20); df['cluster_id']=model.fit_predict(X)
score=silhouette_score(X,df.cluster_id)
profiles=df.groupby('cluster_id').agg(customers=('customer_id','count'),avg_spend=('total_spend_12m_inr','mean'),avg_frequency=('purchase_frequency_12m','mean'),avg_recency=('days_since_last_purchase','mean'),avg_order_value=('average_order_value_inr','mean'),avg_income=('annual_income_inr','mean')).sort_values(['avg_spend','avg_frequency'],ascending=False)
labels=['High-Value Loyal','Regular Customers','Deal Seekers','At-Risk Customers']; mapping={cluster:labels[i] for i,cluster in enumerate(profiles.index)}; df['segment']=df.cluster_id.map(mapping)
df.to_csv(ROOT/'data'/'segmented_customers_output.csv',index=False); profiles.to_csv(ROOT/'data'/'cluster_profiles_output.csv')
print(f'Customers: {len(df)} | Silhouette score (K=4): {score:.3f}'); print(profiles.round(2).to_string())
