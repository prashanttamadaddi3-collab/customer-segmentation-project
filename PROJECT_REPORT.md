# Customer Segmentation Project — Report

## Objective
Group customers into interpretable segments using demographic and purchasing behavior, then visualize segment size, spend contribution and preferences to support targeted marketing decisions.

## Dataset
The included CSV contains 600 synthetic customer records with age, gender, region, annual income, purchase frequency over 12 months, average order value, total 12-month spend, days since last purchase, return rate and preferred category. The dataset is explicitly synthetic, not real company data.

## Technology
- Python for implementation
- pandas and NumPy for data handling
- scikit-learn for StandardScaler, K-Means, silhouette score and PCA
- Streamlit for interactive dashboard
- Plotly for charts

## Methodology
1. Load the CSV and inspect columns.
2. Select numerical demographic and behavior variables.
3. Standardize the features because they use different units/ranges.
4. Fit K-Means with reproducible random state and multiple initializations.
5. Profile clusters by spend, purchase frequency, order value, recency, income and return rate.
6. Assign descriptive heuristic labels based on cluster spending/frequency rankings.
7. Evaluate silhouette score and inspect a PCA projection.
8. Explore category preferences and export profiles/customer tables.

## Dashboard views
- KPI summary: customer count, total spend, average spend and silhouette score
- Total spend by segment and customer distribution
- Income vs total spend scatter plot
- Preferred category by segment
- Segment profile table and downloadable CSV
- Filterable customer explorer and CSV download
- PCA cluster-separation projection and methodology notes

## Business interpretation
- **High-Value Loyal:** retain with loyalty benefits, early access and personalized service.
- **Regular Customers:** encourage repeat purchase with relevant recommendations and bundles.
- **Deal Seekers:** test value bundles and promotions while tracking margins.
- **At-Risk Customers:** test win-back messaging and measure reactivation.

Labels are heuristic, not ground-truth classes. Actual segment composition can change when K changes.

## Limitations
K-Means is sensitive to outliers, feature selection and chosen K. Standardization reduces scale dominance but does not remove outliers. Silhouette score is one evaluation aid, not proof of commercial value. Synthetic data demonstrates the workflow only; business decisions require validated real data and privacy controls.

## Conclusion
The project demonstrates an end-to-end customer segmentation workflow with preprocessing, clustering, profiling, visualization and exportable results. It provides an interactive basis for exploring customer behavior and planning segment-specific strategies.
