import streamlit as st
import os
import warnings
from pathlib import Path

st.set_page_config(
    page_title="YouTube Engagement Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* KPI cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e1e2f 0%, #2a2a40 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    }
    div[data-testid="stMetric"] label {
        color: #8b8fa3 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #e0e0ff !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(30,30,47,0.5);
        border-radius: 10px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #c4b5fd !important;
    }

    /* Section headers */
    h1, h2, h3 { letter-spacing: -0.02em; }

    /* Dividers */
    hr { border-color: rgba(255,255,255,0.06) !important; }

    /* Expander */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }

    /* Hide Streamlit menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Plotly template ─────────────────────────────────────────────────────────
PLOTLY_TEMPLATE = "plotly_dark"
COLOR_SEQ = px.colors.qualitative.Set2
COLOR_PRIMARY  = '#6366f1'
COLOR_ACCENT   = '#f43f5e'
COLOR_GREEN    = '#10b981'
COLOR_AMBER    = '#f59e0b'
COLOR_CYAN     = '#06b6d4'
COLOR_PURPLE   = '#8b5cf6'


# ═══════════════════════════════════════════════════════════════════════════
#  1.  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Loading data …")
def load_data():
    # Resolve the parquet next to this script so it works from any working directory
    file_path = str(Path(__file__).parent / "youtube_trending_videos_global.parquet")
    if not os.path.exists(file_path):
        return pd.DataFrame()

    useful_columns = [
        "video_id", "video_published_at", "video_trending__date",
        "video_trending_country", "video_category_id", "video_duration",
        "video_view_count", "video_like_count", "video_comment_count",
        "channel_id", "channel_title", "channel_country",
        "channel_subscriber_count", "channel_view_count",
        "channel_video_count", "video_title", "video_tags",
        "video_definition", "video_description", "video_licensed_content",
    ]
    import pyarrow.parquet as pq
    actual_cols = pq.ParquetFile(file_path).schema.names
    load_cols = [c for c in useful_columns if c in actual_cols]

    raw = pd.read_parquet(file_path, columns=load_cols)
    if len(raw) > 150_000:
        raw = raw.sample(n=150_000, random_state=42).copy()

    for col in ["video_view_count", "video_like_count",
                "video_comment_count", "channel_subscriber_count"]:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

    RENAME = {
        "video_views": "video_view_count",
        "video_likes": "video_like_count",
        "video_comments": "video_comment_count",
        "channel_view_count": "channel_total_views",
        "channel_subscriber_count": "channel_subscribers",
    }
    raw.rename(columns=RENAME, inplace=True)

    if 'video_published_at' in raw.columns:
        pub_dt = pd.to_datetime(raw['video_published_at'], errors='coerce', utc=True)
        raw['published_time'] = pub_dt.dt.strftime('%H:%M:%S')
        raw['published_day']  = pub_dt.dt.day_name()
    if 'video_trending__date' in raw.columns:
        td = pd.to_datetime(raw['video_trending__date'], errors='coerce', utc=True)
        raw['trending_month'] = td.dt.month
    if 'video_title' in raw.columns:
        raw['video_title_length'] = raw['video_title'].astype(str).str.len()
    if 'video_description' in raw.columns:
        raw['video_description_length'] = raw['video_description'].astype(str).str.len()
    if 'video_duration' in raw.columns:
        td_dur = pd.to_timedelta(raw['video_duration'], errors='coerce')
        raw['video_duration'] = td_dur.dt.total_seconds() / 60.0

    return raw

raw = load_data()
if raw.empty:
    st.error("⚠️  Parquet file not found.")
    st.stop()

VIEW_COL    = 'video_view_count'    if 'video_view_count'    in raw.columns else 'video_views'
LIKE_COL    = 'video_like_count'    if 'video_like_count'    in raw.columns else 'video_likes'
COMMENT_COL = 'video_comment_count' if 'video_comment_count' in raw.columns else 'video_comments'
SUBS_COL    = 'channel_subscribers' if 'channel_subscribers' in raw.columns else 'channel_subscriber_count'


# ═══════════════════════════════════════════════════════════════════════════
#  2.  DATA CLEANING
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Cleaning …")
def clean_data(_raw, vc, lc, cc):
    df = _raw.copy()
    df = df.dropna(subset=[vc, lc, cc, 'video_category_id'])
    df = df[df[vc] > 0]
    df = df[df[lc] <= df[vc]]
    df['video_published_at']   = pd.to_datetime(df['video_published_at'], errors='coerce', utc=True)
    df['video_trending__date'] = pd.to_datetime(df['video_trending__date'], errors='coerce')
    df = df.dropna(subset=['video_published_at', 'video_trending__date'])
    df['video_category_id'] = df['video_category_id'].astype(str).str.strip()
    df['video_definition']  = df['video_definition'].astype(str).str.strip().str.lower()
    df['channel_country']   = df['channel_country'].astype(str).str.strip()
    df['video_tags']        = df['video_tags'].fillna('')
    if 'video_duration' in df.columns:
        df['video_duration'] = pd.to_numeric(df['video_duration'], errors='coerce')
        df.loc[df['video_duration'] < 0, 'video_duration'] = np.nan
    return df

df = clean_data(raw, VIEW_COL, LIKE_COL, COMMENT_COL)


# ═══════════════════════════════════════════════════════════════════════════
#  3.  FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Engineering features …")
def engineer(_df, vc, lc, cc, sc):
    df = _df.copy()
    pub_norm = df['video_published_at'].dt.tz_localize(None).dt.normalize()
    df['days_to_trend'] = (df['video_trending__date'] - pub_norm).dt.days
    df.loc[df['days_to_trend'] < 0, 'days_to_trend'] = np.nan

    def dg(m):
        if pd.isna(m): return 'Unknown'
        if m <= 1:     return 'Shorts (≤1 min)'
        if m <= 5:     return 'Short (1-5 min)'
        if m <= 20:    return 'Mid (5-20 min)'
        if m <= 60:    return 'Long (20-60 min)'
        return 'Very Long (>60 min)'
    DUR_CATS = ['Shorts (≤1 min)','Short (1-5 min)','Mid (5-20 min)',
                'Long (20-60 min)','Very Long (>60 min)','Unknown']
    df['duration_group'] = pd.Categorical(df['video_duration'].map(dg),
                                          categories=DUR_CATS, ordered=True)

    df['like_rate']       = (df[lc] / df[vc]) * 100
    df['comment_rate']    = (df[cc] / df[vc]) * 100
    df['engagement_rate'] = df['like_rate'] + df['comment_rate']
    df['log_views']       = np.log1p(df[vc])
    df['title_length']    = df.get('video_title_length', df['video_title'].astype(str).str.len())
    df['tag_count']       = df['video_tags'].apply(
        lambda t: len([x for x in str(t).split(',') if x.strip()]) if t else 0)
    df['publish_hour'] = pd.to_datetime(
        df['published_time'], format='%H:%M:%S', errors='coerce').dt.hour
    df['publish_day']  = df['published_day']

    def st_tier(s):
        if pd.isna(s):       return 'Unknown'
        if s < 10_000:       return 'Nano (<10K)'
        if s < 100_000:      return 'Micro (10K-100K)'
        if s < 1_000_000:    return 'Mid (100K-1M)'
        if s < 10_000_000:   return 'Macro (1M-10M)'
        return 'Mega (>10M)'
    TIER_CATS = ['Nano (<10K)','Micro (10K-100K)','Mid (100K-1M)',
                 'Macro (1M-10M)','Mega (>10M)','Unknown']
    df['channel_tier'] = pd.Categorical(df[sc].map(st_tier),
                                        categories=TIER_CATS, ordered=True)
    if 'video_description_length' in df.columns:
        df['has_description'] = (df['video_description_length'] > 0).astype(int)
    MONTH_NAMES = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                   7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
    if 'trending_month' in df.columns:
        df['trending_month_name'] = df['trending_month'].map(MONTH_NAMES)
    return df

df = engineer(df, VIEW_COL, LIKE_COL, COMMENT_COL, SUBS_COL)

DAY_ORDER = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
MONTH_ORDER = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']


# ═══════════════════════════════════════════════════════════════════════════
#  4.  SIDEBAR FILTERS
# ═══════════════════════════════════════════════════════════════════════════
st.sidebar.markdown("# 🎬 Filters")
st.sidebar.markdown("---")

all_countries  = sorted(df['video_trending_country'].dropna().unique().tolist())
all_categories = sorted(df['video_category_id'].dropna().unique().tolist())
all_durations  = df['duration_group'].cat.categories.tolist()
all_tiers      = df['channel_tier'].cat.categories.tolist()
all_defs       = sorted(df['video_definition'].dropna().unique().tolist())

sel_countries = st.sidebar.multiselect("🌍 Country", all_countries, default=[],
                                        placeholder="All countries")
sel_categories = st.sidebar.multiselect("📂 Category", all_categories, default=[],
                                         placeholder="All categories")
sel_durations = st.sidebar.multiselect("⏱️ Duration", all_durations, default=[],
                                        placeholder="All durations")
sel_tiers = st.sidebar.multiselect("👥 Channel Tier", all_tiers, default=[],
                                    placeholder="All tiers")
sel_defs = st.sidebar.multiselect("📺 Definition", all_defs, default=[],
                                   placeholder="All (HD/SD)")

# Apply filters
fdf = df.copy()
if sel_countries:
    fdf = fdf[fdf['video_trending_country'].isin(sel_countries)]
if sel_categories:
    fdf = fdf[fdf['video_category_id'].isin(sel_categories)]
if sel_durations:
    fdf = fdf[fdf['duration_group'].isin(sel_durations)]
if sel_tiers:
    fdf = fdf[fdf['channel_tier'].isin(sel_tiers)]
if sel_defs:
    fdf = fdf[fdf['video_definition'].isin(sel_defs)]

st.sidebar.markdown("---")
st.sidebar.metric("Filtered Videos", f"{len(fdf):,}")
st.sidebar.caption(f"of {len(df):,} total")


# ═══════════════════════════════════════════════════════════════════════════
#  5.  HEADER
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="text-align:center; padding: 10px 0 5px 0;">
    <h1 style="margin:0; font-size:2.2rem; background: linear-gradient(90deg, #6366f1, #8b5cf6, #a78bfa);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        YouTube Trending Engagement Dashboard
    </h1>
    <p style="color:#8b8fa3; font-size:0.95rem; margin-top:4px;">
        Interactive analysis of trending video engagement across 20 key visualizations
    </p>
</div>
""", unsafe_allow_html=True)

# ── KPI row ─────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Videos", f"{len(fdf):,}")
k2.metric("Median Views", f"{fdf[VIEW_COL].median():,.0f}")
k3.metric("Avg Like Rate", f"{fdf['like_rate'].mean():.2f}%")
k4.metric("Avg Engagement", f"{fdf['engagement_rate'].mean():.2f}%")
k5.metric("Countries", f"{fdf['video_trending_country'].nunique()}")
k6.metric("Categories", f"{fdf['video_category_id'].nunique()}")

st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════
#  HELPER: common Plotly layout
# ═══════════════════════════════════════════════════════════════════════════
def _layout(fig, h=420):
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=h,
        margin=dict(l=20, r=20, t=50, b=20),
        font=dict(family="Inter, sans-serif", size=12),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  6.  TABS
# ═══════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "📊 Overview",
    "📂 Categories",
    "🎬 Content",
    "🕐 Timing",
    "🌍 Geography",
    "👥 Channels",
    "🔬 Statistics",
])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TAB: Overview
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[0]:
    col_a, col_b = st.columns(2)

    # V01 — Log Views Distribution
    with col_a:
        st.markdown("##### V01 · Distribution of log(1+Views)")
        fig1 = px.histogram(fdf, x='log_views', nbins=70,
                            color_discrete_sequence=[COLOR_PRIMARY],
                            opacity=0.85)
        med = fdf['log_views'].median()
        fig1.add_vline(x=med, line_dash="dash", line_color=COLOR_ACCENT,
                       annotation_text=f"Median={med:.2f}",
                       annotation_position="top right")
        fig1.update_layout(xaxis_title="log(1 + Views)", yaxis_title="Count",
                           bargap=0.05, showlegend=False)
        st.plotly_chart(_layout(fig1), use_container_width=True)

    # V04 — Likes vs Views
    with col_b:
        st.markdown("##### V04 · Likes vs Views (log-log)")
        samp = fdf.sample(min(8000, len(fdf)), random_state=42)
        fig4 = px.scatter(samp, x=VIEW_COL, y=LIKE_COL,
                          color='log_views', color_continuous_scale='Viridis',
                          opacity=0.3, log_x=True, log_y=True,
                          labels={VIEW_COL: 'Views', LIKE_COL: 'Likes',
                                  'log_views': 'log(1+Views)'},
                          hover_data=['video_category_id'])
        fig4.update_traces(marker_size=4)
        st.plotly_chart(_layout(fig4), use_container_width=True)

    # V05 — Correlation Heatmap
    st.markdown("##### V05 · Correlation Heatmap")
    num_cols = [VIEW_COL, LIKE_COL, COMMENT_COL, 'log_views', 'like_rate',
                'comment_rate', 'engagement_rate', 'video_duration',
                'days_to_trend', 'title_length', 'tag_count', SUBS_COL]
    num_cols = [c for c in num_cols if c in fdf.columns]
    corr = fdf[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    corr_masked = corr.where(~mask)

    fig5 = px.imshow(corr_masked, text_auto='.2f', color_continuous_scale='RdBu_r',
                     zmin=-1, zmax=1, aspect='auto',
                     labels=dict(color="Correlation"))
    fig5.update_layout(coloraxis_colorbar=dict(title="r"))
    st.plotly_chart(_layout(fig5, h=550), use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TAB: Categories
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[1]:
    cat_perf = fdf.groupby('video_category_id').agg(
        video_count       =(VIEW_COL, 'count'),
        median_views      =(VIEW_COL, 'median'),
        mean_like_rate    =('like_rate', 'mean'),
        mean_engagement   =('engagement_rate', 'mean'),
    ).round(2).sort_values('median_views', ascending=True).reset_index()

    c1, c2 = st.columns(2)

    # V02 — Median Views by Category
    with c1:
        st.markdown("##### V02 · Median Views by Category")
        fig2 = px.bar(cat_perf, y='video_category_id', x='median_views',
                      orientation='h', color='median_views',
                      color_continuous_scale='Blues',
                      text=cat_perf['median_views'].apply(lambda x: f'{x:,.0f}'))
        fig2.update_traces(textposition='outside', textfont_size=10)
        fig2.update_layout(yaxis_title="", xaxis_title="Median Views",
                           showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(_layout(fig2, h=500), use_container_width=True)

    # V03 — Mean Like-Rate by Category
    with c2:
        st.markdown("##### V03 · Mean Like-Rate by Category")
        cat_lr = cat_perf.sort_values('mean_like_rate', ascending=True)
        fig3 = px.bar(cat_lr, y='video_category_id', x='mean_like_rate',
                      orientation='h', color='mean_like_rate',
                      color_continuous_scale='Greens',
                      text=cat_lr['mean_like_rate'].apply(lambda x: f'{x:.2f}%'))
        fig3.update_traces(textposition='outside', textfont_size=10)
        fig3.update_layout(yaxis_title="", xaxis_title="Like Rate (%)",
                           showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(_layout(fig3, h=500), use_container_width=True)

    # V17 — Engagement Rate Box Plot
    st.markdown("##### V17 · Engagement Rate Distribution by Category")
    cats_sorted = (fdf.groupby('video_category_id')['engagement_rate']
                   .median().sort_values(ascending=False).index.tolist())
    box_data = fdf[fdf['engagement_rate'] <= fdf['engagement_rate'].quantile(0.98)]
    fig17 = px.box(box_data, x='video_category_id', y='engagement_rate',
                   color='video_category_id',
                   category_orders={'video_category_id': cats_sorted},
                   color_discrete_sequence=COLOR_SEQ)
    fig17.update_layout(xaxis_title="Category", yaxis_title="Engagement Rate (%)",
                        xaxis_tickangle=-40, showlegend=False)
    st.plotly_chart(_layout(fig17, h=480), use_container_width=True)

    # V19 — Category × Weekday Heatmap
    st.markdown("##### V19 · Median Views (M) by Category × Publish Weekday")
    pivot = fdf.pivot_table(
        index='video_category_id', columns='publish_day',
        values=VIEW_COL, aggfunc='median'
    ).reindex(columns=DAY_ORDER).dropna(how='all')
    fig19 = px.imshow(pivot / 1e6, text_auto='.1f',
                      color_continuous_scale='YlOrRd', aspect='auto',
                      labels=dict(x="Weekday", y="Category", color="Median Views (M)"))
    st.plotly_chart(_layout(fig19, h=520), use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TAB: Content Optimization
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[2]:
    dur_perf = fdf.groupby('duration_group', observed=True).agg(
        median_views     =(VIEW_COL, 'median'),
        mean_like_rate   =('like_rate', 'mean'),
    ).round(2).reset_index()
    dur_perf['duration_group'] = dur_perf['duration_group'].astype(str)

    c1, c2 = st.columns(2)

    # V06 — Median Views by Duration
    with c1:
        st.markdown("##### V06 · Median Views by Duration Format")
        fig6 = px.bar(dur_perf, x='duration_group', y='median_views',
                      color='duration_group', color_discrete_sequence=px.colors.sequential.Viridis,
                      text=dur_perf['median_views'].apply(lambda x: f'{x:,.0f}'))
        fig6.update_traces(textposition='outside', textfont_size=10)
        fig6.update_layout(xaxis_title="Duration", yaxis_title="Median Views",
                           showlegend=False)
        st.plotly_chart(_layout(fig6), use_container_width=True)

    # V07 — Mean Like-Rate by Duration
    with c2:
        st.markdown("##### V07 · Mean Like-Rate by Duration Format")
        fig7 = px.bar(dur_perf, x='duration_group', y='mean_like_rate',
                      color_discrete_sequence=[COLOR_AMBER],
                      text=dur_perf['mean_like_rate'].apply(lambda x: f'{x:.2f}%'))
        fig7.update_traces(textposition='outside', textfont_size=10)
        fig7.update_layout(xaxis_title="Duration", yaxis_title="Like Rate (%)",
                           showlegend=False)
        st.plotly_chart(_layout(fig7), use_container_width=True)

    c3, c4 = st.columns(2)

    # V11 — Mean Like-Rate by Title Length
    with c3:
        st.markdown("##### V11 · Mean Like-Rate by Title Length")
        fdf_tl = fdf.copy()
        fdf_tl['title_bin'] = pd.cut(fdf_tl['title_length'],
            bins=[0, 20, 40, 60, 80, 100, np.inf],
            labels=['0-20','20-40','40-60','60-80','80-100','100+'])
        tl = fdf_tl.groupby('title_bin', observed=True)['like_rate'].mean().reset_index()
        tl.columns = ['Title Length', 'Like Rate']
        fig11 = px.bar(tl, x='Title Length', y='Like Rate',
                       color_discrete_sequence=[COLOR_PURPLE],
                       text=tl['Like Rate'].apply(lambda x: f'{x:.2f}%'))
        fig11.update_traces(textposition='outside', textfont_size=10)
        fig11.update_layout(yaxis_title="Like Rate (%)", showlegend=False)
        st.plotly_chart(_layout(fig11), use_container_width=True)

    # V12 — Median Views by Tag Count
    with c4:
        st.markdown("##### V12 · Median Views by Number of Tags")
        fdf_tg = fdf.copy()
        fdf_tg['tag_bin'] = pd.cut(fdf_tg['tag_count'],
            bins=[-1, 0, 5, 15, 30, np.inf],
            labels=['0','1-5','6-15','16-30','30+'])
        tg = fdf_tg.groupby('tag_bin', observed=True)[VIEW_COL].median().reset_index()
        tg.columns = ['Tags', 'Median Views']
        fig12 = px.bar(tg, x='Tags', y='Median Views',
                       color_discrete_sequence=[COLOR_AMBER],
                       text=tg['Median Views'].apply(lambda x: f'{x:,.0f}'))
        fig12.update_traces(textposition='outside', textfont_size=10)
        fig12.update_layout(yaxis_title="Median Views", showlegend=False)
        st.plotly_chart(_layout(fig12), use_container_width=True)

    # V18 — Like-Rate by Description Length
    st.markdown("##### V18 · Mean Like-Rate by Description Length")
    fdf_dl = fdf.copy()
    if 'video_description_length' in fdf_dl.columns:
        fdf_dl['desc_bin'] = pd.cut(fdf_dl['video_description_length'],
            bins=[-1, 0, 100, 500, 1500, np.inf],
            labels=['None','1-100','101-500','501-1500','1500+'])
        desc_lr = fdf_dl.groupby('desc_bin', observed=True)['like_rate'].mean().reset_index()
        desc_lr.columns = ['Desc Length', 'Like Rate']
        fig18 = px.bar(desc_lr, x='Desc Length', y='Like Rate',
                       color_discrete_sequence=[COLOR_CYAN],
                       text=desc_lr['Like Rate'].apply(lambda x: f'{x:.2f}%'))
        fig18.update_traces(textposition='outside', textfont_size=10)
        fig18.update_layout(yaxis_title="Like Rate (%)", showlegend=False)
        st.plotly_chart(_layout(fig18), use_container_width=True)

    # V20 — Tag Count vs Engagement
    st.markdown("##### V20 · Tag Count vs Engagement Rate")
    samp2 = fdf.sample(min(8000, len(fdf)), random_state=99)
    fig20 = px.scatter(samp2,
                       x=samp2['tag_count'].clip(0, 40),
                       y=samp2['engagement_rate'].clip(0, 15),
                       color='log_views', color_continuous_scale='Viridis',
                       opacity=0.3, labels={
                           'x': 'Tag Count (clipped 40)',
                           'y': 'Engagement Rate % (clipped 15%)',
                           'log_views': 'log(1+Views)',
                       })
    fig20.update_traces(marker_size=4)
    st.plotly_chart(_layout(fig20), use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TAB: Timing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[3]:
    hour_p = fdf.groupby('publish_hour').agg(
        median_views=(VIEW_COL, 'median'),
        mean_like_rate=('like_rate', 'mean'),
        count=(VIEW_COL, 'count'),
    ).round(2).reset_index()

    day_p = fdf.groupby('publish_day').agg(
        median_views=(VIEW_COL, 'median'),
        mean_like_rate=('like_rate', 'mean'),
        count=(VIEW_COL, 'count'),
    ).round(2)
    day_p = day_p.reindex(DAY_ORDER).reset_index()

    c1, c2 = st.columns(2)

    # V08 — Median Views by Hour
    with c1:
        st.markdown("##### V08 · Median Views by Publish Hour (UTC)")
        fig8 = go.Figure()
        fig8.add_trace(go.Scatter(
            x=hour_p['publish_hour'], y=hour_p['median_views'],
            mode='lines+markers', fill='tozeroy',
            line=dict(color=COLOR_PRIMARY, width=2.5),
            marker=dict(size=7),
            fillcolor='rgba(99,102,241,0.12)',
            hovertemplate='Hour %{x}:00<br>Median Views: %{y:,.0f}<extra></extra>',
        ))
        fig8.update_layout(xaxis=dict(title="Publish Hour (UTC)", dtick=1),
                           yaxis_title="Median Views")
        st.plotly_chart(_layout(fig8), use_container_width=True)

    # V09 — Like-Rate by Weekday
    with c2:
        st.markdown("##### V09 · Mean Like-Rate by Publish Weekday")
        fig9 = px.bar(day_p, x='publish_day', y='mean_like_rate',
                      color_discrete_sequence=[COLOR_GREEN],
                      text=day_p['mean_like_rate'].apply(lambda x: f'{x:.2f}%'))
        fig9.update_traces(textposition='outside', textfont_size=10)
        fig9.update_layout(xaxis_title="Weekday", yaxis_title="Like Rate (%)",
                           showlegend=False)
        st.plotly_chart(_layout(fig9), use_container_width=True)

    c3, c4 = st.columns(2)

    # V10 — Days to Trend
    with c3:
        st.markdown("##### V10 · Days from Publish to Trending")
        clip_days = fdf['days_to_trend'].clip(0, 30).dropna()
        fig10 = px.histogram(clip_days, nbins=31,
                             color_discrete_sequence=[COLOR_ACCENT], opacity=0.85)
        med_d = clip_days.median()
        fig10.add_vline(x=med_d, line_dash="dash", line_color=COLOR_PRIMARY,
                        annotation_text=f"Median={med_d:.0f}d",
                        annotation_position="top right")
        fig10.update_layout(xaxis_title="Days to Trend", yaxis_title="Count",
                            showlegend=False, bargap=0.05)
        st.plotly_chart(_layout(fig10), use_container_width=True)

    # V16 — Monthly Trend Volume
    with c4:
        st.markdown("##### V16 · Trending Video Count by Month")
        monthly = fdf.groupby('trending_month').agg(
            video_count=(VIEW_COL, 'count')).reset_index()
        MONTH_MAP = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                     7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
        monthly['month_name'] = monthly['trending_month'].map(MONTH_MAP)
        monthly = monthly.sort_values('trending_month')

        fig16 = go.Figure()
        fig16.add_trace(go.Scatter(
            x=monthly['month_name'], y=monthly['video_count'],
            mode='lines+markers', fill='tozeroy',
            line=dict(color='#f97316', width=2.5),
            marker=dict(size=8, symbol='square'),
            fillcolor='rgba(249,115,22,0.10)',
            hovertemplate='%{x}<br>Videos: %{y:,}<extra></extra>',
        ))
        fig16.update_layout(xaxis_title="Month", yaxis_title="Videos Trending")
        st.plotly_chart(_layout(fig16), use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TAB: Geography & Definition
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[4]:
    # V13 — Top 15 Countries
    st.markdown("##### V13 · Top 15 Countries by Trending Video Count")
    cntry = fdf.groupby('video_trending_country').agg(
        video_count=(VIEW_COL, 'count'),
        median_views=(VIEW_COL, 'median'),
    ).sort_values('video_count', ascending=False).head(15).reset_index()
    cntry = cntry.sort_values('video_count', ascending=True)

    fig13 = px.bar(cntry, y='video_trending_country', x='video_count',
                   orientation='h', color='video_count',
                   color_continuous_scale='Blues',
                   text=cntry['video_count'].apply(lambda x: f'{x:,}'),
                   hover_data=['median_views'])
    fig13.update_traces(textposition='outside', textfont_size=10)
    fig13.update_layout(yaxis_title="", xaxis_title="Trending Videos",
                        showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(_layout(fig13, h=500), use_container_width=True)

    # V14 — HD vs SD
    c1, c2 = st.columns(2)

    hd_lr = fdf[fdf['video_definition'] == 'hd']['like_rate'].dropna()
    sd_lr = fdf[fdf['video_definition'] == 'sd']['like_rate'].dropna()

    with c1:
        st.markdown("##### V14a · Mean Like-Rate: HD vs SD")
        fig14a = go.Figure()
        fig14a.add_trace(go.Bar(
            x=['HD', 'SD'],
            y=[hd_lr.mean() if len(hd_lr) > 0 else 0,
               sd_lr.mean() if len(sd_lr) > 0 else 0],
            marker_color=[COLOR_GREEN, COLOR_ACCENT],
            text=[f'{hd_lr.mean():.2f}%' if len(hd_lr) > 0 else 'N/A',
                  f'{sd_lr.mean():.2f}%' if len(sd_lr) > 0 else 'N/A'],
            textposition='outside', textfont=dict(size=13, color='white'),
            width=0.4,
        ))
        fig14a.update_layout(yaxis_title="Mean Like Rate (%)")
        st.plotly_chart(_layout(fig14a), use_container_width=True)

    with c2:
        st.markdown("##### V14b · Like-Rate Distribution (Violin)")
        violin_data = pd.DataFrame({
            'Like Rate': pd.concat([hd_lr.clip(0, 15), sd_lr.clip(0, 15)]),
            'Definition': ['HD'] * len(hd_lr) + ['SD'] * len(sd_lr),
        })
        if len(violin_data) > 0:
            fig14b = px.violin(violin_data, x='Definition', y='Like Rate',
                               color='Definition',
                               color_discrete_map={'HD': COLOR_GREEN, 'SD': COLOR_ACCENT},
                               box=True, points=False)
            fig14b.update_layout(yaxis_title="Like Rate % (clipped 15%)",
                                 showlegend=False)
            st.plotly_chart(_layout(fig14b), use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TAB: Channels
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[5]:
    # V15 — Channel Tier
    st.markdown("##### V15 · Median Views by Channel Subscriber Tier")
    tier_p = fdf.groupby('channel_tier', observed=True).agg(
        median_views=(VIEW_COL, 'median'),
        mean_like_rate=('like_rate', 'mean'),
        count=(VIEW_COL, 'count'),
    ).round(2).reset_index()
    tier_p['channel_tier'] = tier_p['channel_tier'].astype(str)

    fig15 = px.bar(tier_p, x='channel_tier', y='median_views',
                   color='channel_tier', color_discrete_sequence=px.colors.sequential.Purples_r,
                   text=tier_p['median_views'].apply(lambda x: f'{x:,.0f}'),
                   hover_data=['mean_like_rate', 'count'])
    fig15.update_traces(textposition='outside', textfont_size=10)
    fig15.update_layout(xaxis_title="Channel Tier", yaxis_title="Median Views",
                        showlegend=False, xaxis_tickangle=-15)
    st.plotly_chart(_layout(fig15), use_container_width=True)

    # Bubble chart — Category performance
    st.markdown("##### Bonus · Category Performance Bubble Chart")
    cat_bubble = fdf.groupby('video_category_id').agg(
        video_count=(VIEW_COL, 'count'),
        median_views=(VIEW_COL, 'median'),
        mean_like_rate=('like_rate', 'mean'),
        mean_engagement=('engagement_rate', 'mean'),
    ).round(2).reset_index()
    fig_bub = px.scatter(cat_bubble, x='mean_like_rate', y='median_views',
                         size='video_count', color='video_category_id',
                         hover_name='video_category_id', size_max=55,
                         labels={'mean_like_rate': 'Mean Like-Rate (%)',
                                 'median_views': 'Median Views'},
                         color_discrete_sequence=COLOR_SEQ)
    fig_bub.update_layout(showlegend=True, legend=dict(font_size=9))
    st.plotly_chart(_layout(fig_bub, h=500), use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TAB: Statistics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[6]:
    st.markdown("##### Pearson Correlation Significance")
    corr_pairs = [
        ('log_views',      LIKE_COL,    'log(Views) vs Likes'),
        (VIEW_COL,         LIKE_COL,    'Views vs Likes'),
        (LIKE_COL,         COMMENT_COL, 'Likes vs Comments'),
        ('video_duration', 'like_rate', 'Duration vs Like-Rate'),
        ('title_length',   'like_rate', 'Title Length vs Like-Rate'),
        ('tag_count',      'log_views', 'Tag Count vs log(Views)'),
        (SUBS_COL,         VIEW_COL,    'Subscribers vs Views'),
        ('days_to_trend',  'like_rate', 'Days-to-Trend vs Like-Rate'),
    ]
    corr_results = []
    for a, b, label in corr_pairs:
        if a not in fdf.columns or b not in fdf.columns:
            continue
        sub = fdf[[a, b]].dropna()
        if len(sub) < 30:
            continue
        r, p = stats.pearsonr(sub[a], sub[b])
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
        corr_results.append({'Pair': label, 'r': round(r, 4),
                             'p-value': f'{p:.2e}', 'Sig.': sig, 'n': len(sub)})
    if corr_results:
        cdf = pd.DataFrame(corr_results)
        # Color bar for r values
        fig_corr = px.bar(cdf, y='Pair', x='r', orientation='h',
                          color='r', color_continuous_scale='RdYlGn',
                          range_color=[-1, 1],
                          text=cdf.apply(lambda row: f"r={row['r']:.3f} {row['Sig.']}", axis=1),
                          hover_data=['p-value', 'n'])
        fig_corr.update_traces(textposition='outside', textfont_size=10)
        fig_corr.update_layout(yaxis_title="", xaxis_title="Pearson r",
                               coloraxis_showscale=False)
        st.plotly_chart(_layout(fig_corr, h=380), use_container_width=True)

    c1, c2 = st.columns(2)

    # ANOVA
    with c1:
        st.markdown("##### One-Way ANOVA")
        anova_tests = [
            ('log_views',       'video_category_id', 'log(Views) by Category'),
            ('like_rate',       'video_category_id', 'Like-Rate by Category'),
            ('log_views',       'duration_group',    'log(Views) by Duration'),
            ('like_rate',       'duration_group',    'Like-Rate by Duration'),
            ('like_rate',       'publish_day',       'Like-Rate by Weekday'),
            ('engagement_rate', 'video_category_id', 'Engagement by Category'),
            ('log_views',       'channel_tier',      'log(Views) by Tier'),
        ]
        anova_rows = []
        for metric, grp, label in anova_tests:
            groups = [g[metric].dropna().values
                      for _, g in fdf.groupby(grp, observed=True)
                      if g[metric].dropna().shape[0] >= 5]
            if len(groups) < 2:
                continue
            F, p = stats.f_oneway(*groups)
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
            anova_rows.append({'Test': label, 'F-stat': round(F, 2),
                               'p-value': f'{p:.2e}', 'Sig.': sig})
        if anova_rows:
            st.dataframe(pd.DataFrame(anova_rows), hide_index=True, use_container_width=True)

    # t-test + Kruskal-Wallis
    with c2:
        st.markdown("##### Welch t-test · HD vs SD")
        hd = fdf[fdf['video_definition'] == 'hd']['like_rate'].dropna()
        sd = fdf[fdf['video_definition'] == 'sd']['like_rate'].dropna()
        if len(hd) > 1 and len(sd) > 1:
            t_stat, p_t = stats.ttest_ind(hd, sd, equal_var=False)
            outcome = ('HD significantly MORE engaging' if t_stat > 0 and p_t < 0.05
                       else 'SD significantly more engaging' if t_stat < 0 and p_t < 0.05
                       else 'No significant difference')
            st.dataframe(pd.DataFrame({
                'Metric': ['Mean Like-Rate', 'n', 't-stat', 'p-value', 'Result'],
                'HD': [f'{hd.mean():.3f}%', f'{len(hd):,}', f'{t_stat:.3f}', f'{p_t:.2e}', outcome],
                'SD': [f'{sd.mean():.3f}%', f'{len(sd):,}', '–', '–', '–'],
            }).set_index('Metric'), use_container_width=True)

        st.markdown("##### Kruskal-Wallis · Views by Category")
        kw_groups = [g[VIEW_COL].dropna().values
                     for _, g in fdf.groupby('video_category_id', observed=True)
                     if g[VIEW_COL].dropna().shape[0] >= 10]
        if len(kw_groups) >= 2:
            H, p_kw = stats.kruskal(*kw_groups)
            conclusion = ('Categories differ **significantly**' if p_kw < 0.05
                          else 'No significant difference')
            st.markdown(f"**H** = {H:.2f} · **p** = {p_kw:.2e} → {conclusion}")

    # OLS Regression
    st.markdown("##### OLS Regression (Target: like_rate)")
    feat_cols = ['video_duration', 'title_length', 'tag_count',
                 'days_to_trend', 'publish_hour']
    if SUBS_COL in fdf.columns:
        feat_cols.append(SUBS_COL)
    feat_cols = [c for c in feat_cols if c in fdf.columns]
    dd = fdf.dropna(subset=feat_cols + ['like_rate'])
    if len(dd) > 10 and len(feat_cols) > 0:
        model = LinearRegression().fit(dd[feat_cols], dd['like_rate'])
        r2 = r2_score(dd['like_rate'], model.predict(dd[feat_cols]))
        coef = pd.DataFrame({'Feature': feat_cols,
                             'Coefficient': model.coef_.round(6)})
        coef = coef.sort_values('Coefficient', key=abs, ascending=True)

        st.markdown(f"**R²** = {r2:.4f} · **Intercept** = {model.intercept_:.4f}")
        fig_reg = px.bar(coef, y='Feature', x='Coefficient', orientation='h',
                         color='Coefficient', color_continuous_scale='RdYlGn',
                         color_continuous_midpoint=0,
                         text=coef['Coefficient'].apply(lambda x: f'{x:.6f}'))
        fig_reg.update_traces(textposition='outside', textfont_size=10)
        fig_reg.update_layout(coloraxis_showscale=False, yaxis_title="")
        st.plotly_chart(_layout(fig_reg, h=300), use_container_width=True)

    # ── Business Recommendations ────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📋 Key Findings")

    cat_perf_full = fdf.groupby('video_category_id').agg(
        median_views=(VIEW_COL, 'median'),
        mean_like_rate=('like_rate', 'mean'),
    ).round(2)
    hour_perf_full = fdf.groupby('publish_hour').agg(median_views=(VIEW_COL, 'median')).round(2)
    day_perf_full = fdf.groupby('publish_day').agg(mean_like_rate=('like_rate', 'mean')).round(2)
    dur_perf_full = fdf.groupby('duration_group', observed=True).agg(
        median_views=(VIEW_COL, 'median')).round(2)

    r1, r2c, r3 = st.columns(3)
    with r1:
        st.markdown(f"""
        **🏆 Top Category (Views)**
        {cat_perf_full['median_views'].idxmax()}

        **💖 Top Category (Like-Rate)**
        {cat_perf_full['mean_like_rate'].idxmax()}
        """)
    with r2c:
        best_h = int(hour_perf_full['median_views'].idxmax()) if len(hour_perf_full) > 0 else 0
        best_d = day_perf_full['mean_like_rate'].idxmax() if len(day_perf_full) > 0 else 'N/A'
        st.markdown(f"""
        **🕐 Best Publish Hour**
        {best_h:02d}:00 UTC

        **📅 Best Publish Day**
        {best_d}
        """)
    with r3:
        best_dur = str(dur_perf_full['median_views'].idxmax()) if len(dur_perf_full) > 0 else 'N/A'
        st.markdown(f"""
        **⏱️ Best Duration**
        {best_dur}

        **🌍 Markets**
        {fdf['video_trending_country'].nunique()} countries
        """)

# ── Footer ──────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.caption(f"Dataset: {len(df):,} rows · {df.shape[1]} cols")
st.sidebar.caption("Built with Streamlit + Plotly")
