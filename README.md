# March Madness Prediction Engine

Live NCAA tournament matchup explorer with KenPom-style ratings, dual prediction models, and real-time ESPN odds comparison.

**Live:** [madness.joecharland.dev](https://madness.joecharland.dev)

## Features

- **Live Matchups** — Select from real tournament games pulled via ESPN API. View team comparisons, derived matchup analytics, and model predictions side by side with DraftKings market odds.
- **Team Deep Dive** — Full KenPom profile for any team: efficiency ratings, four factors, and advanced stats.
- **Future Matchup Lab** — Pick any two teams for hypothetical head-to-head predictions.
- **Dual Models** — Deterministic formula model + trained ML model shown side by side. Agreement/disagreement between models highlights the most interesting games.

## Data Sources

| Source | Data | Auth |
|--------|------|------|
| [KenPom API](https://kenpom.com) | Team ratings, four factors, misc stats | Bearer token ($25/yr subscription) |
| ESPN Scoreboard API | Tournament schedule, scores, DraftKings odds, team logos | None (free) |
| Fallback CSVs | Demo data when APIs are unavailable | N/A |

## Quick Start

```bash
# Clone
git clone https://github.com/jdc141/march-madness-model.git
cd march-madness-model

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env and add your KenPom bearer token

# Run
streamlit run app.py
```

The app works without a KenPom token (falls back to bundled CSV data), but live data requires a [KenPom subscription](https://kenpom.com/register-kenpom.php). Your bearer token is at [kenpom.com/account.php](https://kenpom.com/account.php).

## ML Model Training (optional)

The app ships with a pre-trained model. To retrain:

```bash
# 1. Place historical tournament results at data/historical_results.csv
#    (download from Kaggle: "March Madness Historical DataSet")

# 2. Build training data (requires KenPom API access)
python scripts/build_training_data.py

# 3. Train and export model
python scripts/train_model.py
```

## Deployment

Deployed on Render as a Streamlit web service.

**Build command:** `pip install -r requirements.txt`
**Start command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`

Set `KENPOM_BEARER_TOKEN` as an environment variable in Render's dashboard.

## Tech Stack

Python · Streamlit · pandas · KenPom API · ESPN API · scikit-learn · XGBoost
