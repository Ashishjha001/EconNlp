import streamlit as st
import pandas as pd
import altair as alt

# Page setup
st.set_page_config(page_title="EconNLP Dashboard", layout="wide")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("data/economic_news_tagged.csv")

df = load_data()
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df.dropna(subset=["date"], inplace=True)

st.title("📈 EconNLP: Global Economic News Monitor")

# --- Sidebar filters ---
st.sidebar.header("🔍 Filter News")

event_options = df['event_type'].dropna().unique().tolist()
selected_event = st.sidebar.multiselect("Select Event Type", sorted(event_options), default=event_options)

country_options = df['country_region'].dropna().unique().tolist()
selected_country = st.sidebar.multiselect("Select Country", sorted(country_options), default=country_options)

# --- Apply filters ---
filtered_df = df[
    (df['event_type'].isin(selected_event)) &
    (df['country_region'].isin(selected_country))
]

# --- Metrics row ---
st.markdown("### 📊 Key Stats")
col1, col2, col3 = st.columns(3)
col1.metric("Total Articles", len(df))
col2.metric("Filtered Articles", len(filtered_df))
col3.metric("Event Types Tracked", df["event_type"].nunique())

# --- Chart: Time-Series ---
st.markdown("### 🗓 Event Trends Over Time")

if not filtered_df.empty:
    chart_data = (
        filtered_df.groupby(["date", "event_type"])
        .size()
        .reset_index(name="count")
    )
    line_chart = alt.Chart(chart_data).mark_line().encode(
        x="date:T",
        y="count:Q",
        color="event_type:N",
        tooltip=["date", "event_type", "count"]
    ).properties(width="container", height=400)

    st.altair_chart(line_chart, use_container_width=True)
else:
    st.warning("No data for selected filters.")

# --- Chart: Country-wise Bar Plot ---
st.markdown("### 🌍 Country-wise Article Distribution")

if not filtered_df.empty:
    country_count = (
        filtered_df['country_region'].value_counts()
        .reset_index().rename(columns={'index': 'country', 'country_region': 'count'})
    )
    bar_chart = alt.Chart(country_count).mark_bar().encode(
        x=alt.X('country:N', sort='-y', title="Country"),
        y=alt.Y('count:Q', title="Article Count"),
        tooltip=['country', 'count'],
        color=alt.Color("country:N", legend=None)
    ).properties(width="container", height=400)

    st.altair_chart(bar_chart, use_container_width=True)

# --- Table: News Preview ---
st.markdown("### 📰 Filtered News Table")
st.dataframe(filtered_df[["date", "title", "media_source", "event_type", "country_region"]], use_container_width=True)

# --- Insight Sample ---
st.markdown("### 🧠 Random Sample Insight")
if not filtered_df.empty:
    random_row = filtered_df.sample(1).iloc[0]
    st.info(f"**{random_row['date'].strftime('%b %d, %Y')}** — {random_row['event_type'].title()} in **{random_row['country_region']}**: *{random_row['title']}*")
else:
    st.write("No insights to show.")

# --- Download ---
st.markdown("### 📥 Download CSV")
st.download_button(
    label="Download Filtered Data",
    data=filtered_df.to_csv(index=False),
    file_name="filtered_econnlp_data.csv",
    mime="text/csv"
)

)



