from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parent
rng=np.random.default_rng(20261011); n=600
profiles=rng.choice(['loyal','regular','bargain','risk'],n,p=[.22,.38,.24,.16]); rows=[]
for i,p in enumerate(profiles,1001):
    age=int(np.clip(rng.normal({'loyal':38,'regular':34,'bargain':29,'risk':46}[p],9),18,75))
    gender=rng.choice(['Female','Male','Other'],p=[.49,.49,.02])
    income=float(np.clip(rng.normal({'loyal':82000,'regular':57000,'bargain':39000,'risk':61000}[p],14000),15000,150000))
    freq=int(np.clip(round(rng.normal({'loyal':18,'regular':10,'bargain':8,'risk':3}[p],{'loyal':4,'regular':3,'bargain':3,'risk':2}[p])),1,35))
    aov=float(np.clip(rng.normal({'loyal':4200,'regular':2300,'bargain':1200,'risk':1900}[p],{'loyal':800,'regular':500,'bargain':300,'risk':500}[p]),300,8000))
    spend=max(0,freq*aov*rng.uniform(.75,1.2))
    recency=int(np.clip(rng.normal({'loyal':12,'regular':28,'bargain':36,'risk':105}[p],{'loyal':8,'regular':14,'bargain':18,'risk':30}[p]),1,180))
    returns=float(np.clip(rng.normal({'loyal':.035,'regular':.06,'bargain':.09,'risk':.12}[p],.025),0,.35))
    category=rng.choice(['Electronics','Fashion','Home','Beauty','Sports','Grocery'],p={'loyal':[.25,.18,.20,.12,.15,.10],'regular':[.16,.23,.18,.14,.10,.19],'bargain':[.10,.28,.12,.16,.08,.26],'risk':[.18,.15,.21,.12,.10,.24]}[p])
    rows.append({'customer_id':f'C{i}','age':age,'gender':gender,'region':rng.choice(['North','South','East','West','Central']),'annual_income_inr':round(income,2),'purchase_frequency_12m':freq,'average_order_value_inr':round(aov,2),'total_spend_12m_inr':round(spend,2),'days_since_last_purchase':recency,'return_rate':round(returns,3),'preferred_category':category})
df=pd.DataFrame(rows); df.to_csv(ROOT/'data/customer_data.csv',index=False); df.describe(include='all').transpose().to_csv(ROOT/'data/dataset_summary.csv')
print('Generated',len(df),'synthetic customer records')
