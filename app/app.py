import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle

# ─── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="E-Commerce Intelligence",
    page_icon="🛒",
    layout="wide"
)

# ─── HELPER: Normalize Customer ID ─────────────────────────────
def normalize_cust_id(x):
    """Convert any form of customer ID to clean integer string: '12347.0' → '12347'"""
    try:
        return str(int(float(x)))
    except (ValueError, TypeError):
        return str(x)

# ─── LOAD DATA ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    df          = pd.read_csv('../data/processed/cleaned_data.csv')
    rfm         = pd.read_csv('../data/processed/rfm_segments.csv')
    segments    = pd.read_csv('../data/processed/customer_segments.csv')
    daily       = pd.read_csv('../data/processed/daily_revenue.csv')
    churn       = pd.read_csv('../data/processed/churn_scores.csv')
    cp_matrix   = pd.read_csv('../data/processed/customer_product_matrix.csv', index_col=0)

    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    daily['Date']     = pd.to_datetime(daily['Date'])

    # ── Normalize ALL Customer IDs to clean integer strings ('12347', not '12347.0') ──
    df['Customer ID']         = df['Customer ID'].apply(normalize_cust_id)
    churn['Customer ID']      = churn['Customer ID'].apply(normalize_cust_id)
    segments['Customer ID']   = segments['Customer ID'].apply(normalize_cust_id)
    cp_matrix.index           = [normalize_cust_id(i) for i in cp_matrix.index]
    cp_matrix.columns         = [str(c) for c in cp_matrix.columns]

    return df, rfm, segments, daily, churn, cp_matrix

@st.cache_resource
def load_models():
    with open('../models/forecast_model.pkl', 'rb') as f:
        forecast_model = pickle.load(f)
    with open('../models/churn_model.pkl', 'rb') as f:
        churn_model = pickle.load(f)
    with open('../models/churn_scaler.pkl', 'rb') as f:
        churn_scaler = pickle.load(f)
    with open('../models/recommendation_model.pkl', 'rb') as f:
        rec_model = pickle.load(f)
    with open('../models/item_similarity_model.pkl', 'rb') as f:
        item_model = pickle.load(f)

    # ── Normalize rec_model index and columns to clean integer strings ──
    rec_model.index   = [normalize_cust_id(i) for i in rec_model.index]
    rec_model.columns = [normalize_cust_id(c) for c in rec_model.columns]

    return forecast_model, churn_model, churn_scaler, rec_model, item_model

df, rfm, segments, daily, churn, cp_matrix = load_data()
forecast_model, churn_model, churn_scaler, rec_model, item_model = load_models()

# ─── SIDEBAR ───────────────────────────────────────────────────
st.sidebar.title("🛒 E-Commerce Intelligence")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "📊 Overview",
    "🔍 EDA",
    "👥 Customer Segments",
    "📈 Sales Forecast",
    "⚠️ Churn Predictor",
    "🎯 Recommendations"
])
st.sidebar.markdown("---")
st.sidebar.caption("Built with Python · Scikit-learn · Streamlit")

# ─── PAGE 1: OVERVIEW ──────────────────────────────────────────
if page == "📊 Overview":
    st.title("📊 Business Overview")
    st.markdown("Key performance indicators for the e-commerce store.")

    total_revenue   = df['TotalPrice'].sum()
    total_orders    = df['Invoice'].nunique()
    total_customers = df['Customer ID'].nunique()
    aov             = total_revenue / total_orders
    repeat_rate     = (rfm[rfm['Frequency'] > 1].shape[0] / total_customers) * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Revenue",        f"£{total_revenue:,.0f}")
    c2.metric("Total Orders",         f"{total_orders:,}")
    c3.metric("Unique Customers",     f"{total_customers:,}")
    c4.metric("Avg Order Value",      f"£{aov:,.2f}")
    c5.metric("Repeat Customer Rate", f"{repeat_rate:.1f}%")

    st.markdown("---")

    df['Month'] = df['InvoiceDate'].dt.to_period('M').astype(str)
    monthly = df.groupby('Month')['TotalPrice'].sum().reset_index()
    fig = px.line(monthly, x='Month', y='TotalPrice',
        title='Monthly Revenue Trend',
        labels={'TotalPrice': 'Revenue (£)', 'Month': 'Month'})
    fig.update_traces(line_color='steelblue', line_width=2)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        top_countries = df.groupby('Country')['TotalPrice'].sum()\
            .sort_values(ascending=False).head(10).reset_index()
        fig2 = px.bar(top_countries, x='TotalPrice', y='Country',
            orientation='h', title='Top 10 Countries by Revenue',
            labels={'TotalPrice': 'Revenue (£)'}, color='TotalPrice',
            color_continuous_scale='Blues')
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        top_products = df.groupby('Description')['TotalPrice'].sum()\
            .sort_values(ascending=False).head(10).reset_index()
        fig3 = px.bar(top_products, x='TotalPrice', y='Description',
            orientation='h', title='Top 10 Products by Revenue',
            labels={'TotalPrice': 'Revenue (£)'}, color='TotalPrice',
            color_continuous_scale='Greens')
        st.plotly_chart(fig3, use_container_width=True)

# ─── PAGE 2: EDA ───────────────────────────────────────────────
elif page == "🔍 EDA":
    st.title("🔍 Exploratory Data Analysis")

    col1, col2 = st.columns(2)
    with col1:
        day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        df['DayOfWeek'] = df['InvoiceDate'].dt.day_name()
        day_rev = df.groupby('DayOfWeek')['TotalPrice'].sum().reindex(day_order).reset_index()
        fig = px.bar(day_rev, x='DayOfWeek', y='TotalPrice',
            title='Revenue by Day of Week',
            labels={'TotalPrice': 'Revenue (£)', 'DayOfWeek': 'Day'},
            color='TotalPrice', color_continuous_scale='Oranges')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        df['Hour'] = df['InvoiceDate'].dt.hour
        hour_rev = df.groupby('Hour')['TotalPrice'].sum().reset_index()
        fig2 = px.bar(hour_rev, x='Hour', y='TotalPrice',
            title='Revenue by Hour of Day',
            labels={'TotalPrice': 'Revenue (£)', 'Hour': 'Hour'},
            color='TotalPrice', color_continuous_scale='Purples')
        st.plotly_chart(fig2, use_container_width=True)

    top_customers = df.groupby('Customer ID')['TotalPrice'].sum()\
        .sort_values(ascending=False).head(10).reset_index()
    fig3 = px.bar(top_customers, x='TotalPrice', y='Customer ID',
        orientation='h', title='Top 10 Customers by Revenue',
        labels={'TotalPrice': 'Revenue (£)', 'Customer ID': 'Customer'},
        color='TotalPrice', color_continuous_scale='Reds')
    fig3.update_yaxes(type='category')
    st.plotly_chart(fig3, use_container_width=True)

# ─── PAGE 3: CUSTOMER SEGMENTS ─────────────────────────────────
elif page == "👥 Customer Segments":
    st.title("👥 Customer Segments")

    col1, col2 = st.columns(2)
    with col1:
        seg_counts = segments['Cluster_Name'].value_counts().reset_index()
        seg_counts.columns = ['Segment', 'Count']
        fig = px.pie(seg_counts, names='Segment', values='Count',
            title='Customer Distribution by Segment',
            color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        seg_rev = segments.groupby('Cluster_Name')['Monetary'].sum()\
            .sort_values(ascending=False).reset_index()
        fig2 = px.bar(seg_rev, x='Monetary', y='Cluster_Name',
            orientation='h', title='Revenue by Segment',
            labels={'Monetary': 'Total Revenue (£)', 'Cluster_Name': 'Segment'},
            color='Monetary', color_continuous_scale='Teal')
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Segment Details")
    summary = segments.groupby('Cluster_Name').agg(
        Customers     = ('Customer ID', 'count'),
        Avg_Recency   = ('Recency', 'mean'),
        Avg_Frequency = ('Frequency', 'mean'),
        Avg_Spend     = ('Monetary', 'mean')
    ).round(2).reset_index()
    st.dataframe(summary, use_container_width=True)

# ─── PAGE 4: SALES FORECAST ────────────────────────────────────
elif page == "📈 Sales Forecast":
    st.title("📈 Sales Forecast")

    split_index  = int(len(daily) * 0.8)
    feature_cols = ['DayOfWeek', 'Month', 'WeekOfYear', 'IsWeekend', 'Lag7', 'Lag30', 'Rolling7']
    X_test = daily[feature_cols][split_index:]
    y_test = daily['Revenue'][split_index:]
    y_pred = forecast_model.predict(X_test)
    dates  = daily['Date'][split_index:]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=y_test.values,
        mode='lines', name='Actual Revenue',
        line=dict(color='steelblue', width=2)))
    fig.add_trace(go.Scatter(x=dates, y=y_pred,
        mode='lines', name='Predicted Revenue',
        line=dict(color='orange', width=2, dash='dash')))
    fig.update_layout(title='Actual vs Predicted Revenue',
        xaxis_title='Date', yaxis_title='Revenue (£)')
    st.plotly_chart(fig, use_container_width=True)

    mae  = np.mean(np.abs(y_test.values - y_pred))
    mape = np.mean(np.abs((y_test.values - y_pred) / y_test.values)) * 100
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE",      f"£{mae:,.2f}")
    c2.metric("MAPE",     f"{mape:.2f}%")
    c3.metric("Accuracy", f"{100 - mape:.2f}%")

# ─── PAGE 5: CHURN PREDICTOR ───────────────────────────────────
elif page == "⚠️ Churn Predictor":
    st.title("⚠️ Churn Predictor")
    st.markdown("Enter a Customer ID to see their churn risk.")
    st.caption("Example: 12347 or 12348 or 12349")

    customer_id = st.text_input("Customer ID", placeholder="e.g. 12347")

    if customer_id:
        try:
            normalized_id = normalize_cust_id(customer_id)
            match = churn[churn['Customer ID'] == normalized_id]

            if match.empty:
                st.error(f"Customer {customer_id} not found. Try IDs between 12346 and 18287.")
            else:
                row  = match.iloc[0]
                prob = row['Churn_Probability']
                risk = row['Churn_Risk']

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Recency (days)",    int(row['Recency']))
                c2.metric("Frequency",         int(row['Frequency']))
                c3.metric("Total Spend",       f"£{row['Monetary']:,.2f}")
                c4.metric("Churn Probability", f"{prob:.1%}")

                if risk == 'High':
                    st.error("🔴 HIGH RISK — This customer is very likely to churn.")
                elif risk == 'Medium':
                    st.warning("🟡 MEDIUM RISK — Monitor this customer closely.")
                else:
                    st.success("🟢 LOW RISK — This customer is likely to stay.")

        except ValueError:
            st.error("Please enter a valid numeric Customer ID.")

    st.markdown("---")
    st.markdown("### Top 10 Active Customers Most Likely to Churn")
    top_risk = churn[churn['Churned'] == 0].sort_values(
        'Churn_Probability', ascending=False).head(10)[
        ['Customer ID', 'Recency', 'Frequency', 'Monetary', 'Churn_Probability', 'Churn_Risk']]
    top_risk = top_risk.copy()
    top_risk['Churn_Probability'] = top_risk['Churn_Probability'].apply(lambda x: f"{x:.1%}")
    st.dataframe(top_risk, use_container_width=True)

# ─── PAGE 6: RECOMMENDATIONS ───────────────────────────────────
elif page == "🎯 Recommendations":
    st.title("🎯 Product Recommendations")

    tab1, tab2 = st.tabs(["Customer Based", "Item Based"])

    with tab1:
        st.markdown("Enter a Customer ID to get product recommendations.")
        st.caption("Example: 12347 or 12348 or 12349")
        cust_id = st.text_input("Customer ID", placeholder="e.g. 12347", key="cust")

        if cust_id:
            try:
                # ── FIX: normalize to clean integer string ('12347', not '12347.0') ──
                normalized_cust = normalize_cust_id(cust_id)

                rec_model_ids = rec_model.index.tolist()

                if normalized_cust not in rec_model_ids:
                    st.error(f"Customer {cust_id} not found in recommendation model. "
                             f"Try IDs like 12347, 12348, or 12349.")
                else:
                    # Get top 10 similar customers
                    similar = rec_model[normalized_cust]\
                        .sort_values(ascending=False)\
                        .drop(normalized_cust, errors='ignore')\
                        .head(10).index.tolist()

                    # Products this customer already bought
                    already_bought = set()
                    if normalized_cust in cp_matrix.index:
                        already_bought = set(
                            cp_matrix.loc[normalized_cust][
                                cp_matrix.loc[normalized_cust] > 0].index
                        )

                    # Aggregate products from similar customers
                    recommended = {}
                    for sc in similar:
                        if sc in cp_matrix.index:
                            for product, qty in cp_matrix.loc[sc].items():
                                if qty > 0 and product not in already_bought:
                                    recommended[product] = recommended.get(product, 0) + qty

                    if not recommended:
                        st.warning("No new recommendations found for this customer.")
                    else:
                        top_recs = sorted(recommended.items(),
                            key=lambda x: x[1], reverse=True)[:5]

                        st.markdown("### Recommended Products:")
                        for i, (product, _) in enumerate(top_recs, 1):
                            st.write(f"**{i}.** {product}")

            except ValueError:
                st.error("Please enter a valid numeric Customer ID.")

    with tab2:
        st.markdown("Select a product to find similar products.")
        all_products = sorted(item_model.index.tolist())
        selected = st.selectbox("Select a Product", all_products)
        if selected:
            similar_items = item_model[selected]\
                .sort_values(ascending=False)\
                .drop(selected, errors='ignore').head(5)
            st.markdown("### Customers who bought this also bought:")
            for i, (product, score) in enumerate(similar_items.items(), 1):
                st.write(f"**{i}.** {product}  —  similarity: {score:.2f}")