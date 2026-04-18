from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import pandas as pd

app = FastAPI()

# ✅ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load model
model = joblib.load("model.pkl")

# ✅ Load datasets
deliveries = pd.read_csv("deliveries.csv")
matches = pd.read_csv("matches.csv")

# Merge datasets
df = deliveries.merge(matches, left_on="match_id", right_on="id")

# Convert date
df["date"] = pd.to_datetime(df["date"])


# ---------------- FEATURE FUNCTION ----------------
def generate_features(player_name, team, opponent, venue, mean_runs_input, boundary_pct_input, strike_rate_input):
    player_df = df[df["batter"] == player_name]

    # If player not found
    if player_df.empty:
        raise ValueError(f"No data found for player: {player_name}")

    # Aggregate runs per match
    player_match = (
        player_df.groupby(["match_id", "date"])["batsman_runs"]
        .sum()
        .reset_index()
        .sort_values(by="date")
    )

    # Take last 5 matches
    last5 = player_match.tail(5)

    if len(last5) < 5:
    # use whatever matches available instead of failing
        last5 = player_match.tail(len(player_match))

    # -------- RUN FEATURES --------
    mean_runs = mean_runs_input
    print("Mean Runs from user:", mean_runs_input)
    print("Final mean_runs used:", mean_runs)
    std_runs = last5["batsman_runs"].std()
    min_runs = last5["batsman_runs"].min()
    max_runs = last5["batsman_runs"].max()

    # -------- BALL LEVEL FEATURES --------
    last5_ids = last5["match_id"].values
    balls_df = player_df[player_df["match_id"].isin(last5_ids)]

    balls = len(balls_df)
    runs = balls_df["batsman_runs"].sum()

    strike_rate = strike_rate_input if balls > 0 else 0

    fours = (balls_df["batsman_runs"] == 4).sum()
    sixes = (balls_df["batsman_runs"] == 6).sum()

    dots = (balls_df["batsman_runs"] == 0).sum()
    dot_pct = (dots / balls) if balls > 0 else 0

    boundary_pct = boundary_pct_input if balls > 0 else 0
      # -------- CAREER FEATURES --------
    career_avg = player_df["batsman_runs"].mean()
    matches_played = player_df["match_id"].nunique()

    # -------- CONTEXT FEATURES --------
    team_df = df[(df["batter"] == player_name) & (df["batting_team"] == team)]
    opp_df = df[(df["batter"] == player_name) & (df["bowling_team"] == opponent)]
    venue_df = df[(df["batter"] == player_name) & (df["venue"] == venue)]

    team_avg = team_df["batsman_runs"].mean() if not team_df.empty else career_avg
    opp_avg = opp_df["batsman_runs"].mean() if not opp_df.empty else career_avg
    venue_avg = venue_df["batsman_runs"].mean() if not venue_df.empty else career_avg

    # # -------- CAREER FEATURES --------
    # career_avg = player_df["batsman_runs"].mean()
    # matches_played = player_df["match_id"].nunique()

    # -------- FINAL FEATURE LIST --------
    features = [
        mean_runs,
        std_runs,
        min_runs,
        max_runs,
        strike_rate,
        balls,
        dot_pct,
        boundary_pct,
        fours,
        sixes,
        team_avg,
        opp_avg,
        venue_avg,
        career_avg,
        matches_played
    ]

    # Fill remaining features to match model input (28)

    return features


# ---------------- API ----------------
@app.get("/")
def home():
    return {"message": "API running"}


@app.post("/predict")
def predict(data: dict):
    try:
        player = data["player"]
        team = data["team"]
        opponent = data["opponent"]
        venue = data["venue"]

        mean_runs_input = data["mean_runs"]
        boundary_pct_input = data["boundary_pct"]
        strike_rate_input = data["strike_rate"]

        feature_list = generate_features(
            player,
            team,
            opponent,
            venue,
            mean_runs_input,
            boundary_pct_input,
            strike_rate_input
        )

        features = np.array([feature_list])

        prediction = model.predict(features)
        prediction = max(0, float(prediction[0]))

        # Explanation logic (same as before)
        explanation = []

# 🔹 Mean Runs Impact
        if mean_runs_input > 40:
            explanation.append("Strong recent form (High Impact)")
        elif mean_runs_input > 25:
            explanation.append("Decent recent form (Medium Impact)")
        else:
            explanation.append("Low recent form (Low Impact)")

# 🔹 Strike Rate Impact
        if strike_rate_input > 150:
            explanation.append("High strike rate (High Impact)")
        elif strike_rate_input > 120:
            explanation.append("Moderate strike rate (Medium Impact)")
        else:
            explanation.append("Low strike rate (Low Impact)")

# 🔹 Boundary % Impact
        if boundary_pct_input > 0.3:
            explanation.append("Aggressive boundary hitting (High Impact)")
        elif boundary_pct_input > 0.2:
            explanation.append("Balanced boundary hitting (Medium Impact)")
        else:
            explanation.append("Low boundary frequency (Low Impact)")

# 🔹 Match Context
        if team != opponent:
            explanation.append("Matchup considered")

        return {
            "success": True,
            "player": player,
            "predicted_runs": prediction,
            "explanation": explanation
}

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }