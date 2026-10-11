# Customer Segmentation Project

An interactive customer analytics dashboard using Python, pandas, scikit-learn, Streamlit and Plotly.

## Features
- K-Means clustering using demographic and behavioral features
- StandardScaler preprocessing and silhouette score
- Segment profiles for spend, frequency, average order value, recency, income and returns
- Interactive charts: segment spend, customer distribution, income vs spend, preferred categories and PCA projection
- Filterable customer explorer and CSV downloads

## Dataset disclosure
`data/customer_data.csv` contains 600 **synthetic demonstration records**, generated reproducibly for this project. They are not actual company/customer records. Replace with approved real data before making real business decisions.

## Run locally (Python 3.10–3.12 recommended)
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
Open the local URL Streamlit prints (usually http://localhost:8501).

## Run batch analysis
```bash
python analysis.py
```
This creates `data/segmented_customers_output.csv` and `data/cluster_profiles_output.csv`.

## Method
1. Select age, annual income, purchase frequency, average order value, total spend, recency and return rate.
2. Standardize features with `StandardScaler`.
3. Fit K-Means with a reproducible seed and multiple initializations.
4. Profile clusters and assign heuristic descriptive labels based on spend/frequency ranking.
5. Review silhouette score and PCA projection alongside business interpretability.

## Deployment
This is a Streamlit application. For Streamlit Community Cloud, push the project to GitHub and select `app.py` as the entry point with `requirements.txt` as dependencies. The app must be opened and tested after publishing; the ZIP itself is not a live deployment.
