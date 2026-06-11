import streamlit as st
from PIL import Image

st.title("📈 Customer Insights")

st.subheader("Churn Distribution")

st.image(
    "reports/figures/churn_distribution.png",
    use_container_width=True
)

st.subheader("Correlation Heatmap")

st.image(
    "reports/figures/heatmap.png",
    use_container_width=True
)

st.subheader("Feature Importance")

st.image(
    "reports/figures/feature_importance.png",
    use_container_width=True
)

st.markdown("---")

st.subheader("Business Insights")

st.markdown("""
### Key Findings

- Customers with low tenure are more likely to churn.
- Active members show lower churn rates.
- Older customers demonstrate higher churn probability.
- Customers with fewer products are at higher risk.

### Recommendations

- Create loyalty programs for low-tenure customers.
- Increase engagement for inactive customers.
- Personalize retention campaigns.
- Monitor high-balance customers closely.
""")