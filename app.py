import os
if not os.path.exists("phc_data.csv"):
    import generate_data
import time
import math
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# 1. Page Configuration
st.set_page_config(
    page_title="National Health Resource & Supply Chain Platform",
    page_icon="🏥",
    layout="wide"
)

# --- GEOSPATIAL HELPER ---
def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth in kilometers using the Haversine formula.
    """
    R = 6371.0  # Earth radius in kilometers
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return float(R * c)

# 2. Data Loading & Caching
@st.cache_data
def load_data():
    df = pd.read_csv("phc_data.csv")
    df['date'] = pd.to_datetime(df['date'])
    return df

@st.cache_data
def get_monthly_trends(df, districts):
    filtered = df[df['district'].isin(districts)]
    monthly = filtered.groupby(df['date'].dt.to_period('M'))['units_consumed'].sum().reset_index()
    monthly['date'] = monthly['date'].dt.to_timestamp()
    return monthly

@st.cache_data
def get_peak_stock(df):
    peak_stock = df.groupby(['phc_id', 'medicine'])['closing_stock'].max().reset_index()
    peak_stock.columns = ['phc_id', 'medicine', 'peak_stock']
    return peak_stock

# Load data
try:
    df = load_data()
    all_districts = sorted(df['district'].unique())
    all_medicines = sorted(df['medicine'].unique())
except FileNotFoundError:
    st.error("Dataset 'phc_data.csv' not found.")
    st.stop()

# 3. Sidebar Navigation & Filters
st.sidebar.title("🏥 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "PHC Explorer", "Demand Forecast", "Stock-out Alerts", "AI Redistribution", "Ask AI"]
)

st.sidebar.divider()
st.sidebar.title("Filters")
selected_districts = st.sidebar.multiselect(
    "Select Districts",
    options=all_districts,
    default=all_districts
)

# Global Constants
latest_date = df['date'].max()

# --- FORECAST LOGIC HELPER ---
def run_forecast_logic(ts_data, forecast_days=30):
    """Simple seasonal moving average forecast"""
    # Calculate Monthly Seasonal Factors
    ts_data['month'] = ts_data['date'].dt.month
    avg_consumption = ts_data['units_consumed'].mean()
    seasonal_factors = ts_data.groupby('month')['units_consumed'].mean() / avg_consumption
    
    # Baseline: Average of last 7 days
    baseline = ts_data['units_consumed'].tail(7).mean()
    
    # Project forward
    last_date = ts_data['date'].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=forecast_days)
    
    forecast_vals = []
    for d in future_dates:
        # Seasonality-adjusted forecast
        factor = seasonal_factors.get(d.month, 1.0)
        forecast_vals.append(baseline * factor)
        
    return pd.DataFrame({'date': future_dates, 'units_consumed': forecast_vals, 'type': 'Forecast'})

# --- GEMINI CALL HELPER WITH FAIL-FAST RETRY ---
def get_gemini_api_key():
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        try:
            key = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
        except Exception:
            pass
    return key

def call_gemini_flash(prompt):
    """
    Calls gemini-flash-lite-latest with fail-fast retry:
    Tries only twice, waiting 2 seconds between tries.
    If it fails twice, immediately displays a friendly busy message and stops.
    """
    api_key = get_gemini_api_key()
    max_retries = 2

    for attempt in range(max_retries):
        try:
            # Preferred SDK: google-genai
            try:
                from google import genai
                client = genai.Client(api_key=api_key) if api_key else genai.Client()
                response = client.models.generate_content(
                    model="gemini-flash-lite-latest",
                    contents=prompt
                )
                return response.text
            except ImportError:
                # Fallback SDK: google-generativeai
                import google.generativeai as legacy_genai
                if api_key:
                    legacy_genai.configure(api_key=api_key)
                model = legacy_genai.GenerativeModel(model_name="gemini-flash-lite-latest")
                response = model.generate_content(prompt)
                return response.text

        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            else:
                st.warning("⚠️ The AI assistant is briefly busy — everything else works, please try Ask AI again in a moment.")
                st.stop()

    return None

# 4. Page Logic
if page == "Overview":
    st.title("National Health Resource & Supply Chain Platform")
    df_latest = df[(df['district'].isin(selected_districts)) & (df['date'] == latest_date)]
    
    # KPIs
    m1, m2, m3, m4 = st.columns(4)
    total_phcs = df_latest['phc_id'].nunique()
    peak_df = get_peak_stock(df)
    stock_analysis = pd.merge(df_latest, peak_df, on=['phc_id', 'medicine'], how='left')
    low_stock_count = stock_analysis[(stock_analysis['closing_stock'] < (0.2 * stock_analysis['peak_stock']))].shape[0]
    
    phc_metrics = df_latest.groupby('phc_id').agg({'occupied_beds':'sum','total_beds':'sum','staff_present':'sum','staff_sanctioned':'sum'})
    m1.metric("Total PHCs", f"{total_phcs:,}")
    m2.metric("Critical Stock Alerts", f"{low_stock_count:,}")
    m3.metric("Avg Bed Occupancy", f"{(phc_metrics['occupied_beds'].sum()/phc_metrics['total_beds'].sum()*100):.1f}%")
    m4.metric("Avg Staff Attendance", f"{(phc_metrics['staff_present'].sum()/phc_metrics['staff_sanctioned'].sum()*100):.1f}%")

    st.subheader("Medicine Consumption Trend")
    trend_data = get_monthly_trends(df, selected_districts)
    fig_trend = px.line(trend_data, x='date', y='units_consumed', line_shape="spline")
    fig_trend.add_vrect(x0="2024-06-01", x1="2024-09-30", fillcolor="red", opacity=0.1, annotation_text="Monsoon Spike")
    st.plotly_chart(fig_trend, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("PHCs Needing Attention")
        st.dataframe(df_latest[['phc_name', 'medicine', 'closing_stock']].sort_values('closing_stock').head(15).style.background_gradient(cmap='Reds_r'), use_container_width=True, hide_index=True)
    with c2:
        st.subheader("Geographic Distribution")
        st.map(df_latest[['latitude', 'longitude']].drop_duplicates(), size=20)

elif page == "PHC Explorer":
    st.title("Individual PHC Explorer")
    available_phcs = sorted(df[df['district'].isin(selected_districts)]['phc_name'].unique())
    sel_phc = st.selectbox("Select PHC", available_phcs)
    phc_data = df[df['phc_name'] == sel_phc].sort_values('date')
    latest_phc = phc_data[phc_data['date'] == latest_date]
    
    s1, s2, s3 = st.columns(3)
    s1.metric("Beds", f"{latest_phc['occupied_beds'].iloc[0]}/{latest_phc['total_beds'].iloc[0]}")
    s2.metric("Staff", f"{latest_phc['staff_present'].iloc[0]}/{latest_phc['staff_sanctioned'].iloc[0]}")
    s3.metric("Daily Footfall", int(latest_phc['patient_footfall'].iloc[0]))
    
    st.table(latest_phc[['medicine', 'opening_stock', 'units_consumed', 'closing_stock']])
    fig_f = px.area(phc_data.groupby('date')['patient_footfall'].first().reset_index(), x='date', y='patient_footfall', title="Patient History")
    st.plotly_chart(fig_f, use_container_width=True)

elif page == "Demand Forecast":
    st.title("Medicine Demand Forecasting")
    
    f1, f2 = st.columns(2)
    with f1:
        phc_list = sorted(df[df['district'].isin(selected_districts)]['phc_name'].unique())
        sel_phc = st.selectbox("Select PHC for Forecast", phc_list)
    with f2:
        sel_med = st.selectbox("Select Medicine", all_medicines)
    
    # Prepare Time Series Data
    ts = df[(df['phc_name'] == sel_phc) & (df['medicine'] == sel_med)].sort_values('date')
    
    # 1. Forecast Generation
    forecast_df = run_forecast_logic(ts.copy())
    
    # 2. Back-testing for MAPE
    train_ts = ts.iloc[:-30].copy()
    test_actuals = ts.iloc[-30:].copy()
    if len(train_ts) > 60: # Ensure enough data to train
        backtest_forecast = run_forecast_logic(train_ts, forecast_days=30)
        merged = pd.merge(test_actuals[['date', 'units_consumed']], backtest_forecast[['date', 'units_consumed']], on='date', suffixes=('_act', '_pred'))
        # Avoid division by zero
        merged = merged[merged['units_consumed_act'] > 0]
        mape = np.mean(np.abs((merged['units_consumed_act'] - merged['units_consumed_pred']) / merged['units_consumed_act'])) * 100
    else:
        mape = 0
    
    # 3. Visualization
    history_90 = ts.tail(90).copy()
    history_90['type'] = 'Actual'
    
    plot_df = pd.concat([history_90[['date', 'units_consumed', 'type']], forecast_df])
    
    fig_forecast = px.line(plot_df, x='date', y='units_consumed', color='type',
                           color_discrete_map={'Actual': '#007BFF', 'Forecast': '#FF8C00'},
                           title=f"30-Day Demand Projection: {sel_med} at {sel_phc}")
    
    fig_forecast.update_layout(hovermode="x unified")
    st.plotly_chart(fig_forecast, use_container_width=True)
    
    # 4. Metrics & Explanation
    c1, c2 = st.columns([1, 3])
    with c1:
        st.metric("Forecast Accuracy (MAPE)", f"{mape:.2f}%", delta="Back-tested", delta_color="off")
    with c2:
        st.info(f"""
        **Forecast Summary:**
        The model uses a seasonal moving average to predict that **{sel_med}** consumption will likely follow 
        the historical {ts['date'].max().strftime('%B')} patterns for this PHC. 
        The forecast factors in a 7-day recent baseline of {ts['units_consumed'].tail(7).mean():.1f} units/day 
        weighted by monthly seasonality coefficients.
        """)

elif page == "Stock-out Alerts":
    st.title("Stock-out Alerts & Early Warning")

    # Apply district filter
    df_filtered = df[df['district'].isin(selected_districts)]

    if df_filtered.empty:
        st.warning("Please select at least one district with available data.")
    else:
        # 1. Most recent date's closing_stock as current_stock
        df_latest_stock = df_filtered[df_filtered['date'] == latest_date][
            ['phc_name', 'district', 'medicine', 'closing_stock']
        ].rename(columns={'closing_stock': 'current_stock'})

        # 2. Average daily consumption over the last 30 days
        cutoff_date = latest_date - pd.Timedelta(days=30)
        df_last_30 = df_filtered[df_filtered['date'] > cutoff_date]
        avg_cons = df_last_30.groupby(['phc_name', 'district', 'medicine'])['units_consumed'].mean().reset_index()
        avg_cons.rename(columns={'units_consumed': 'avg_daily_consumption'}, inplace=True)

        # Merge current stock and consumption
        alerts_df = pd.merge(df_latest_stock, avg_cons, on=['phc_name', 'district', 'medicine'], how='left')
        alerts_df['avg_daily_consumption'] = alerts_df['avg_daily_consumption'].fillna(0.0)

        # 3. Compute days_until_stockout logic
        def compute_stockout_info(row):
            stock = row['current_stock']
            cons = row['avg_daily_consumption']
            if stock <= 0:
                return 0.0, "STOCKED OUT"
            if cons <= 0:
                return float('inf'), "Safe"
            days = stock / cons
            return days, round(days, 1)

        res = alerts_df.apply(compute_stockout_info, axis=1)
        alerts_df['days_numeric'] = [r[0] for r in res]
        alerts_df['days_until_stockout'] = [r[1] for r in res]

        # 4. Display Metrics
        stocking_out_15 = (alerts_df['days_numeric'] <= 15).sum()
        already_zero = (alerts_df['current_stock'] <= 0).sum()

        m1, m2 = st.columns(2)
        m1.metric("Pairs Stocking Out Within 15 Days", f"{stocking_out_15:,}")
        m2.metric("Pairs Already at Zero Stock", f"{already_zero:,}")

        # 5. Slider for threshold
        threshold = st.slider("Show alerts within X days", min_value=1, max_value=60, value=15, step=1)

        # 6. Filter and sort table (most urgent first)
        table_df = alerts_df[alerts_df['days_numeric'] <= threshold].copy()
        table_df.sort_values(by=['days_numeric', 'current_stock'], ascending=[True, True], inplace=True)

        display_df = table_df[['phc_name', 'district', 'medicine', 'current_stock', 'avg_daily_consumption', 'days_until_stockout']].copy()
        display_df['avg_daily_consumption'] = display_df['avg_daily_consumption'].round(2)
        display_df['current_stock'] = display_df['current_stock'].astype(int)

        # 7. Row coloring: red for under 7 days, amber for 7-15 days
        def highlight_urgency(row):
            val = row['days_until_stockout']
            if val == "STOCKED OUT":
                return ['background-color: #ffcccc; color: #900c3f; font-weight: 500;'] * len(row)
            try:
                days = float(val)
                if days < 7:
                    return ['background-color: #ffcccc; color: #900c3f; font-weight: 500;'] * len(row)
                elif days <= 15:
                    return ['background-color: #fff2cc; color: #856404; font-weight: 500;'] * len(row)
            except (ValueError, TypeError):
                pass
            return [''] * len(row)

        if display_df.empty:
            st.info(f"No stock-out alerts found within {threshold} days for the selected district(s).")
        else:
            styled_table = display_df.style.apply(highlight_urgency, axis=1)
            st.dataframe(styled_table, use_container_width=True, hide_index=True)

elif page == "AI Redistribution":
    st.title("AI-Driven Inter-PHC Medicine Redistribution")

    # Initialize approval session state
    if "approved_transfers" not in st.session_state:
        st.session_state.approved_transfers = set()

    df_filtered = df[df['district'].isin(selected_districts)]

    if df_filtered.empty:
        st.warning("Please select at least one district with available data.")
    else:
        # 1. Latest stock data
        latest_df = df_filtered[df_filtered['date'] == latest_date].copy()

        # 2. Average daily consumption over the last 30 days
        cutoff_date = latest_date - pd.Timedelta(days=30)
        df_last_30 = df_filtered[df_filtered['date'] > cutoff_date]
        avg_cons = df_last_30.groupby(['phc_name', 'district', 'medicine'])['units_consumed'].mean().reset_index()
        avg_cons.rename(columns={'units_consumed': 'avg_daily_consumption'}, inplace=True)

        stock_df = pd.merge(
            latest_df[['phc_name', 'district', 'medicine', 'closing_stock']],
            avg_cons,
            on=['phc_name', 'district', 'medicine'],
            how='left'
        )
        stock_df['avg_daily_consumption'] = stock_df['avg_daily_consumption'].fillna(0.0)

        # PHC coordinates lookup
        coords = df[['phc_name', 'latitude', 'longitude']].dropna().drop_duplicates('phc_name').set_index('phc_name')

        # Baseline medicine consumption across filtered data (for items with 0 recent consumption)
        med_avg_dict = df_filtered.groupby('medicine')['units_consumed'].mean().to_dict()

        # 3. Generate Redistribution Recommendations
        recommendations = []
        for med in sorted(stock_df['medicine'].unique()):
            med_df = stock_df[stock_df['medicine'] == med]

            # Deficit: < 15 days supply or already stocked out (closing_stock <= 0)
            deficit_phcs = med_df[
                (med_df['closing_stock'] <= 0) |
                ((med_df['avg_daily_consumption'] > 0) & ((med_df['closing_stock'] / med_df['avg_daily_consumption']) < 15))
            ]

            # Surplus: > 60 days supply
            surplus_phcs = med_df[
                (med_df['avg_daily_consumption'] > 0) &
                ((med_df['closing_stock'] / med_df['avg_daily_consumption']) > 60)
            ]

            if deficit_phcs.empty or surplus_phcs.empty:
                continue

            for _, def_row in deficit_phcs.iterrows():
                dst_phc = def_row['phc_name']
                dst_district = def_row['district']
                dst_stock = def_row['closing_stock']
                dst_cons = def_row['avg_daily_consumption']

                # Effective consumption for calculating 30-day target
                eff_dst_cons = dst_cons if dst_cons > 0 else med_avg_dict.get(med, 1.0)
                if eff_dst_cons <= 0:
                    eff_dst_cons = 1.0

                needed = max(0.0, (30 * eff_dst_cons) - max(0.0, dst_stock))
                if needed <= 0 or dst_phc not in coords.index:
                    continue

                lat_dst, lon_dst = coords.loc[dst_phc, 'latitude'], coords.loc[dst_phc, 'longitude']

                # Find nearest surplus PHC with the same medicine
                best_surplus = None
                min_dist = float('inf')

                for _, sur_row in surplus_phcs.iterrows():
                    src_phc = sur_row['phc_name']
                    if src_phc == dst_phc or src_phc not in coords.index:
                        continue
                    lat_src, lon_src = coords.loc[src_phc, 'latitude'], coords.loc[src_phc, 'longitude']
                    dist = haversine(lat_dst, lon_dst, lat_src, lon_src)
                    if dist < min_dist:
                        min_dist = dist
                        best_surplus = sur_row

                if best_surplus is not None:
                    src_phc = best_surplus['phc_name']
                    src_stock = best_surplus['closing_stock']
                    src_cons = best_surplus['avg_daily_consumption']

                    # Surplus excess beyond 60 days
                    excess = max(0.0, src_stock - (60 * src_cons))
                    max_allowed = 0.5 * excess

                    transfer_qty = int(round(min(needed, max_allowed)))
                    if transfer_qty > 0:
                        days_prevented = round(transfer_qty / eff_dst_cons, 1)
                        rec_id = f"{src_phc} -> {dst_phc} : {med}"
                        recommendations.append({
                            'rec_id': rec_id,
                            'source_phc': src_phc,
                            'destination_phc': dst_phc,
                            'district': dst_district,
                            'medicine': med,
                            'transfer_quantity': transfer_qty,
                            'distance_km': round(min_dist, 1),
                            'days_prevented': days_prevented
                        })

        # 4. Display Metrics
        total_transfers = len(recommendations)
        total_units = sum(r['transfer_quantity'] for r in recommendations)
        approved_count = sum(1 for r in recommendations if r['rec_id'] in st.session_state.approved_transfers)

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Recommended Transfers", f"{total_transfers:,}")
        m2.metric("Total Units to Redistribute", f"{total_units:,}")
        m3.metric("Approved Transfers", f"{approved_count}/{total_transfers}")

        st.divider()

        if total_transfers == 0:
            st.info("No inter-PHC redistribution opportunities found under the current criteria for the selected district(s).")
        else:
            # 5. Approval Mechanism Controls
            ctrl1, ctrl2, ctrl3 = st.columns([2, 1, 1])
            with ctrl1:
                pending_options = [r['rec_id'] for r in recommendations if r['rec_id'] not in st.session_state.approved_transfers]
                if pending_options:
                    selected_rec = st.selectbox("Select Transfer to Approve Individually:", options=pending_options)
                    if st.button("Approve Selected Transfer"):
                        st.session_state.approved_transfers.add(selected_rec)
                        st.rerun()
                else:
                    st.success("🎉 All recommended transfers have been approved!")
            with ctrl2:
                st.write("")
                st.write("")
                if st.button("✅ Approve All Transfers", use_container_width=True):
                    for r in recommendations:
                        st.session_state.approved_transfers.add(r['rec_id'])
                    st.rerun()
            with ctrl3:
                st.write("")
                st.write("")
                if st.button("🔄 Reset Approvals", use_container_width=True):
                    st.session_state.approved_transfers.clear()
                    st.rerun()

            # 6. Recommendations Table
            table_records = []
            for r in recommendations:
                status = "Approved" if r['rec_id'] in st.session_state.approved_transfers else "Pending"
                table_records.append({
                    'source_phc': r['source_phc'],
                    'destination_phc': r['destination_phc'],
                    'district': r['district'],
                    'medicine': r['medicine'],
                    'transfer_quantity': r['transfer_quantity'],
                    'distance_km': r['distance_km'],
                    'days_prevented': r['days_prevented'],
                    'status': status
                })

            rec_df = pd.DataFrame(table_records)

            def highlight_status(row):
                if row['status'] == "Approved":
                    return ['background-color: #d4edda; color: #155724; font-weight: 500;'] * len(row)
                return ['background-color: #fff3cd; color: #856404; font-weight: 500;'] * len(row)

            styled_rec_table = rec_df.style.apply(highlight_status, axis=1)
            st.dataframe(styled_rec_table, use_container_width=True, hide_index=True)
            
elif page == "Ask AI":
    st.title("🤖 Ask AI - Health Resource & Supply Chain Intelligence")
    st.markdown("Generate automated health executive briefings or ask specific operational questions powered by Gemini.")

    # Context builder for Gemini
    df_filtered = df[df['district'].isin(selected_districts)]
    df_latest = df_filtered[df_filtered['date'] == latest_date]
    total_phcs = df_latest['phc_id'].nunique()
    peak_df = get_peak_stock(df)
    stock_analysis = pd.merge(df_latest, peak_df, on=['phc_id', 'medicine'], how='left')
    low_stock_count = stock_analysis[(stock_analysis['closing_stock'] < (0.2 * stock_analysis['peak_stock']))].shape[0]
    zero_stock_count = df_latest[df_latest['closing_stock'] <= 0].shape[0]

    top_critical_pairs = df_latest[df_latest['closing_stock'] <= 5][['phc_name', 'district', 'medicine', 'closing_stock']].head(10).to_dict(orient='records')
    top_consumed = df_filtered.groupby('medicine')['units_consumed'].sum().sort_values(ascending=False).head(5).to_dict()

    system_data_context = f"""
    You are an expert public health supply chain analyst for a national healthcare system.
    Current Analysis Date: {latest_date.strftime('%Y-%m-%d')}
    Selected Districts: {', '.join(selected_districts)}
    Total PHCs Analyzed: {total_phcs}
    PHC-Medicine pairs currently at ZERO stock: {zero_stock_count}
    PHC-Medicine pairs below 20% peak stock: {low_stock_count}
    Top 5 consumed medicines across districts: {top_consumed}
    Critical shortage snapshot: {top_critical_pairs}
    """

    # Section 1: Executive Briefing
    st.subheader("📋 Executive Briefing")
    st.markdown("Generate a real-time operational summary highlighting stock emergencies, utilization, and priority interventions.")

    if st.button("Generate Executive Briefing"):
        briefing_prompt = f"""
        {system_data_context}

        Please provide a concise and actionable Executive Briefing with the following sections:
        1. **Situation Overview**: High-level status of supply chain and facilities.
        2. **Critical Supply Bottlenecks**: Specific medicines and PHCs facing imminent or active stock-out risks.
        3. **Immediate Recommended Actions**: 3-4 concrete interventions for district health officers.
        """
        with st.spinner("Thinking..."):
            briefing_response = call_gemini_flash(briefing_prompt)
            if briefing_response:
                st.session_state["latest_briefing"] = briefing_response

    if "latest_briefing" in st.session_state:
        st.markdown(st.session_state["latest_briefing"])

    st.divider()

    # Section 2: Interactive Question Answering
    st.subheader("💬 Ask a Specific Question")
    st.markdown("Inquire about medicine stock levels, trends, bed occupancy, or redistribution advice.")

    # Preset quick questions
    quick_col1, quick_col2, quick_col3 = st.columns(3)
    preset_q = None
    if quick_col1.button("🚨 Which PHCs need immediate stock intervention?"):
        preset_q = "Which PHCs need immediate stock intervention and for which medicines?"
    if quick_col2.button("📈 Which medicines have the highest consumption spikes?"):
        preset_q = "Which medicines have the highest consumption spikes and how should districts prepare?"
    if quick_col3.button("🔄 Recommend stock redistribution options"):
        preset_q = "Recommend stock redistribution options between surplus and deficit PHCs."

    user_query = st.chat_input("Type your question here (e.g., 'What are the top risks in Jaipur district?')...")
    active_question = preset_q or user_query

    if active_question:
        st.chat_message("user").markdown(active_question)
        
        question_prompt = f"""
        {system_data_context}

        The user is asking:
        "{active_question}"

        Please provide a clear, factual, and actionable response based on the health system data provided.
        """
        with st.spinner("Thinking..."):
            answer = call_gemini_flash(question_prompt)
            if answer:
                with st.chat_message("assistant"):
                    st.markdown(answer)

# Custom Styling
st.markdown("""
    <style>
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #eee; }
    </style>
    """, unsafe_allow_html=True)