# YouTube Content Success Analytics

Capstone project analysing engagement drivers and content-performance trends across
~36,500 unique trending YouTube videos (deduplicated from ~450k trending appearances).

## Contents
- `youtube_content_success_analytics.ipynb` — cleaning, EDA, 16 visualizations, statistical tests
- `app2.py` — interactive Streamlit dashboard (7 tabs, 20 visualizations)
- `reports/` — Data Quality and EDA reports
- `presentation/` — final presentation deck
- `figures/`, `outputs/` — saved charts and summary tables

## Tools
pandas · numpy · scipy · scikit-learn · matplotlib · seaborn · plotly · streamlit

## Run the dashboard
```bash
pip install -r requirements.txt   # or: pandas numpy scipy scikit-learn plotly streamlit pyarrow
streamlit run app2.py
```

> Note: the raw dataset (`youtube_trending_3_countries.csv`, ~1 GB) is not committed —
> download it from Kaggle. The dashboard runs from the included `youtube_trending_videos_global.parquet`.
