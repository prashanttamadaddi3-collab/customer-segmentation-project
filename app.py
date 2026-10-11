from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

st.set_page_config(page_title='Customer Segmentation | InsightLab', page_icon='📊', layout='wide')
ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'/'customer_data.csv'
FEATURES=['age','annual_income_inr','purchase_frequency_12m','average_order_value_inr','total_spend_12m_inr','days_since_last_purchase','return_rate']
COLORS={'High-Value Loyal':'#2563eb','Regular Customers':'#0d9488','Deal Seekers':'#d97706','At-Risk Customers':'#dc2626'}
@st.cache_data
def load_data(): return pd.read_csv(DATA)
@st.cache_data
def run_model(frame,k):
    scaled=StandardScaler().fit_transform(frame[FEATURES])
    model=KMeans(n_clusters=k,random_state=42,n_init=20)
    out=frame.copy(); out['cluster_id']=model.fit_predict(scaled)
    prof=out.groupby('cluster_id').agg(customers=('customer_id','count'),spend=('total_spend_12m_inr','mean'),frequency=('purchase_frequency_12m','mean'),recency=('days_since_last_purchase','mean'),aov=('average_order_value_inr','mean'),income=('annual_income_inr','mean'),returns=('return_rate','mean'))
    ranked=prof.sort_values(['spend','frequency'],ascending=[False,False]).index.tolist()
    labels=['High-Value Loyal','Regular Customers','Deal Seekers','At-Risk Customers']
    mapping={c:labels[i] if i<len(labels) else f'Segment {i+1}' for i,c in enumerate(ranked)}
    out['segment']=out.cluster_id.map(mapping)
    sil=silhouette_score(scaled,out.cluster_id) if len(set(out.cluster_id))>1 else float('nan')
    return out,prof,sil

data=load_data()
st.markdown('<style>.block-container{padding-top:1.4rem}.stMetric{background:#f8fafc;padding:14px;border-radius:12px}</style>',unsafe_allow_html=True)
st.title('📊 Customer Segmentation Dashboard')
st.caption('Behavior-led customer groups • K-Means clustering • Purchase patterns and actionable insights')
with st.sidebar:
    st.header('Model controls'); k=st.slider('Number of segments (K)',3,5,4)
    st.markdown('---'); st.write(f'**Dataset:** {len(data):,} sample customers')
    st.caption('Synthetic demonstration data—not actual company/customer records.')
    st.write('**Method:** K-Means + StandardScaler')
seg,profiles,sil=run_model(data,k)
c1,c2,c3,c4=st.columns(4)
c1.metric('Customers analyzed',f'{len(seg):,}')
c2.metric('Total 12-month spend',f"₹{seg.total_spend_12m_inr.sum():,.0f}")
c3.metric('Average spend/customer',f"₹{seg.total_spend_12m_inr.mean():,.0f}")
c4.metric('Silhouette score',f'{sil:.3f}' if np.isfinite(sil) else 'N/A')
t1,t2,t3,t4=st.tabs(['Overview','Segment profiles','Customer explorer','Methodology'])
with t1:
    left,right=st.columns(2)
    spend=seg.groupby('segment',as_index=False).agg(total_spend=('total_spend_12m_inr','sum')).sort_values('total_spend',ascending=False)
    with left:
        fig=px.bar(spend,x='segment',y='total_spend',color='segment',color_discrete_map=COLORS,title='Total customer spend by segment',labels={'segment':'Segment','total_spend':'Total spend (₹)'})
        fig.update_layout(showlegend=False,xaxis_tickangle=-12); st.plotly_chart(fig,use_container_width=True)
    with right:
        counts=seg.segment.value_counts().rename_axis('segment').reset_index(name='customers')
        fig=px.pie(counts,names='segment',values='customers',hole=.58,color='segment',color_discrete_map=COLORS,title='Customer distribution')
        st.plotly_chart(fig,use_container_width=True)
    left,right=st.columns(2)
    with left:
        fig=px.scatter(seg,x='annual_income_inr',y='total_spend_12m_inr',color='segment',color_discrete_map=COLORS,hover_data=['customer_id','age','purchase_frequency_12m','preferred_category'],title='Annual income vs. 12-month spend',labels={'annual_income_inr':'Annual income (₹)','total_spend_12m_inr':'12-month spend (₹)'})
        st.plotly_chart(fig,use_container_width=True)
    with right:
        cat=seg.groupby(['segment','preferred_category']).size().reset_index(name='customers')
        fig=px.bar(cat,x='preferred_category',y='customers',color='segment',barmode='group',color_discrete_map=COLORS,title='Preferred category by segment',labels={'preferred_category':'Category','customers':'Customers'})
        fig.update_layout(xaxis_tickangle=-15); st.plotly_chart(fig,use_container_width=True)
    st.subheader('Key business takeaways')
    prof=seg.groupby('segment').agg(customers=('customer_id','count'),avg_spend=('total_spend_12m_inr','mean'),avg_freq=('purchase_frequency_12m','mean'),recency=('days_since_last_purchase','mean'))
    if 'High-Value Loyal' in prof.index:
        r=prof.loc['High-Value Loyal']; st.success(f"High-Value Loyal: {int(r.customers)} customers; average 12-month spend ₹{r.avg_spend:,.0f}. Prioritize retention and loyalty rewards.")
    if 'At-Risk Customers' in prof.index:
        r=prof.loc['At-Risk Customers']; st.warning(f"At-Risk Customers: average {r.recency:.0f} days since last purchase. Test win-back messaging and targeted reactivation offers.")
    st.info('Validate segment-specific strategies against real, approved customer data before making operational decisions.')
with t2:
    table=seg.groupby('segment').agg(Customers=('customer_id','count'),**{'Avg age':('age','mean'),'Avg income (₹)':('annual_income_inr','mean'),'Avg purchases/year':('purchase_frequency_12m','mean'),'Avg order value (₹)':('average_order_value_inr','mean'),'Avg 12-month spend (₹)':('total_spend_12m_inr','mean'),'Avg days since purchase':('days_since_last_purchase','mean'),'Avg return rate':('return_rate','mean')}).round(2).reset_index()
    st.dataframe(table,use_container_width=True,hide_index=True)
    st.download_button('Download segment profile CSV',table.to_csv(index=False).encode(), 'segment_profiles.csv','text/csv')
with t3:
    chosen=st.multiselect('Filter segments',sorted(seg.segment.unique()),default=sorted(seg.segment.unique()))
    category=st.selectbox('Preferred category',['All']+sorted(seg.preferred_category.unique()))
    filtered=seg[seg.segment.isin(chosen)]
    if category!='All': filtered=filtered[filtered.preferred_category==category]
    cols=['customer_id','age','gender','region','annual_income_inr','purchase_frequency_12m','average_order_value_inr','total_spend_12m_inr','days_since_last_purchase','return_rate','preferred_category','segment']
    st.caption(f'Showing {len(filtered):,} customers'); st.dataframe(filtered[cols].sort_values('total_spend_12m_inr',ascending=False),use_container_width=True,hide_index=True)
    st.download_button('Download filtered customer CSV',filtered[cols].to_csv(index=False).encode(),'segmented_customers.csv','text/csv')
with t4:
    st.markdown('''### Methodology\n1. **Preprocess:** choose demographic and behavior variables.\n2. **Scale:** `StandardScaler` prevents rupee-valued fields from dominating smaller-scale features.\n3. **Cluster:** K-Means groups customers by distance to centroids.\n4. **Profile:** compare spend, frequency, recency, order value, income and return rate.\n5. **Evaluate:** silhouette score is one signal of cluster separation, not proof of business usefulness.\n\n**Limitations:** labels are heuristic; K-Means is sensitive to outliers and feature selection; data is synthetic and should not be represented as actual business data.''')
    xy=PCA(n_components=2,random_state=42).fit_transform(StandardScaler().fit_transform(seg[FEATURES]))
    pca=pd.DataFrame({'PC1':xy[:,0],'PC2':xy[:,1],'segment':seg.segment,'customer_id':seg.customer_id})
    st.plotly_chart(px.scatter(pca,x='PC1',y='PC2',color='segment',color_discrete_map=COLORS,hover_data=['customer_id'],title='Cluster separation (PCA projection)'),use_container_width=True)
st.markdown('---'); st.caption('Customer Segmentation Project • Python • pandas • scikit-learn • Streamlit • Plotly')
