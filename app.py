"""
YouTube Content Success Analytics — Interactive Dashboard
---------------------------------------------------------
Explore engagement drivers of ~36,500 UNIQUE trending YouTube videos (deduplicated from ~442k
trending appearances). Filter by category, duration and channel size; a separate tab shows
geographic spread at the appearance grain.

Data: outputs/youtube_videos.parquet (video level) + outputs/country_appearances.parquet
Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="YouTube Content Success Analytics", page_icon="📈", layout="wide")


@st.cache_data                               # load once, then cache
def load_videos():
    return pd.read_parquet("outputs/youtube_videos.parquet")


@st.cache_data
def load_appearances():
    return pd.read_parquet("outputs/country_appearances.parquet")


df = load_videos()          # video-level (primary)
appear = load_appearances()  # appearance-level (geography only)

# ---------------------------------------------------------------- header
st.title("📈 YouTube Content Success Analytics")
st.caption("Engagement drivers across ~36,500 unique trending videos "
           "(deduplicated from ~442k trending appearances).")

# ---------------------------------------------------------------- sidebar filters (all video-level)
st.sidebar.header("Filters")
cats = st.sidebar.multiselect("Category", sorted(df["video_category_id"].dropna().unique()),
                              default=sorted(df["video_category_id"].dropna().unique()))
durs = st.sidebar.multiselect("Duration format", [str(x) for x in df["duration_group"].cat.categories],
                              default=[str(x) for x in df["duration_group"].cat.categories])
tiers = st.sidebar.multiselect("Channel subscriber tier", [str(x) for x in df["channel_tier"].cat.categories],
                               default=[str(x) for x in df["channel_tier"].cat.categories])

mask = (df["video_category_id"].isin(cats)
        & df["duration_group"].astype(str).isin(durs)
        & df["channel_tier"].astype(str).isin(tiers))
d = df[mask]
st.sidebar.metric("Unique videos in selection", f"{len(d):,}")

if len(d) < 50:
    st.warning("Fewer than 50 videos selected — widen the filters.")
    st.stop()

# ---------------------------------------------------------------- KPI row
st.subheader("Key metrics for the current selection")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Unique videos", f"{len(d):,}")
c2.metric("Median views", f"{d['video_view_count'].median():,.0f}")
c3.metric("Mean like-rate", f"{d['like_rate'].mean():.2f}%")
c4.metric("Mean engagement-rate", f"{d['engagement_rate'].mean():.2f}%")
c5.metric("Median days-to-trend", f"{d['first_days_to_trend'].median():.0f}")

# ---------------------------------------------------------------- tabs
st.subheader("Explore the drivers")
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["By category", "By duration", "Publish timing", "Reach vs engagement", "Geographic spread"])

with tab1:
    agg = (d.groupby("video_category_id")
             .agg(median_views=("video_view_count", "median"),
                  mean_like_rate=("like_rate", "mean"), videos=("video_id", "count"))
             .reset_index().sort_values("median_views", ascending=False))
    col1, col2 = st.columns(2)
    col1.plotly_chart(px.bar(agg, x="median_views", y="video_category_id", orientation="h",
                             title="Median views by category"), use_container_width=True)
    col2.plotly_chart(px.bar(agg.sort_values("mean_like_rate"), x="mean_like_rate", y="video_category_id",
                             orientation="h", title="Mean like-rate (%) by category"),
                      use_container_width=True)

with tab2:
    agg = (d.groupby("duration_group", observed=True)
             .agg(median_views=("video_view_count", "median"),
                  mean_like_rate=("like_rate", "mean")).reset_index())
    col1, col2 = st.columns(2)
    col1.plotly_chart(px.bar(agg, x="duration_group", y="median_views",
                             title="Median views by video length"), use_container_width=True)
    col2.plotly_chart(px.bar(agg, x="duration_group", y="mean_like_rate",
                             title="Mean like-rate (%) by video length"), use_container_width=True)

with tab3:
    hr = d.groupby("publish_hour")["video_view_count"].median().reset_index()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dy = d.groupby("publish_day")["like_rate"].mean().reindex(day_order).reset_index()
    col1, col2 = st.columns(2)
    col1.plotly_chart(px.line(hr, x="publish_hour", y="video_view_count", markers=True,
                              title="Median views by publish hour (UTC)"), use_container_width=True)
    col2.plotly_chart(px.bar(dy, x="publish_day", y="like_rate",
                             title="Mean like-rate (%) by weekday"), use_container_width=True)

with tab4:
    samp = d.sample(min(6000, len(d)), random_state=1)
    fig = px.scatter(samp, x="video_view_count", y="video_like_count", log_x=True, log_y=True,
                     opacity=0.25, title="Likes vs Views (log-log)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Correlation log(views) ~ log(likes): "
               f"{np.corrcoef(d['log_views'], np.log1p(d['video_like_count']))[0,1]:.3f}")

with tab5:
    st.caption("Geographic spread uses the **appearance grain** — a video legitimately trends in "
               "many countries. (Filters above apply to video-level tabs only.)")
    col1, col2 = st.columns(2)
    cc = appear["video_trending_country"].value_counts().head(15).reset_index()
    cc.columns = ["country", "trending_appearances"]
    col1.plotly_chart(px.bar(cc, x="trending_appearances", y="country", orientation="h",
                             title="Top countries by trending appearances"), use_container_width=True)
    spread = d["countries_trended"].clip(1, 30)
    col2.plotly_chart(px.histogram(spread, nbins=30,
                      title="Countries a video trends in (selection)"), use_container_width=True)

# ---------------------------------------------------------------- takeaways
st.divider()
st.markdown(
    "**Key drivers (video level):** Shorts (<1 min) win reach (~6× the 5–20 min band); "
    "mid-to-long videos earn higher like-rates; Science & Tech / Comedy lead on views; "
    "HD beats SD on engagement (3.4% vs 1.6%); early-UTC uploads (03:00–05:00) trend highest. "
    "69% of trending videos trend in only one country."
)
