from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import base64
from math import pi
import plotly.graph_objects as go

app = Flask(__name__)
plt.switch_backend('Agg')  # Required for Flask

# Load and prepare data (same as before)
batting_odi = pd.read_csv("ODI data.csv")
bowling_odi = pd.read_csv("Bowling_ODI.csv")
fielding_odi = pd.read_csv("Fielding_ODI.csv")

batting_t20 = pd.read_csv("t20.csv")
bowling_t20 = pd.read_csv("Bowling_t20.csv")
fielding_t20 = pd.read_csv("Fielding_t20.csv")

odi_stats = batting_odi.merge(bowling_odi, on="Player", how="outer").merge(fielding_odi, on="Player", how="outer")
t20_stats = batting_t20.merge(bowling_t20, on="Player", how="outer").merge(fielding_t20, on="Player", how="outer")

def prepare_stats(df):
    numeric_cols = ["Runs_x", "Inns_x", "SR_x", "Wkts", "Econ", "Ct"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df.drop_duplicates("Player")

odi_stats = prepare_stats(odi_stats)
t20_stats = prepare_stats(t20_stats)

def get_player_stats(name, format_type):
    df = odi_stats if format_type == "ODI" else t20_stats
    player = df[df["Player"].str.lower().str.contains(name.lower())]
    if not player.empty:
        return {
            "Batting Average": player["Runs_x"].sum() / player["Inns_x"].sum() if player["Inns_x"].sum() > 0 else 0,
            "Total Runs": player["Runs_x"].sum(),
            "Total Wickets": player["Wkts"].sum(),
            "Bowling Economy": player["Econ"].mean(),
            "Batting Strike Rate": player["SR_x"].mean(),
            "Total Catches": player["Ct"].sum()
        }
    return None

def create_radar_chart(categories, values, players, colors):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories, color='grey', size=10)
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1], ["0.2","0.4","0.6","0.8","1"], color="grey", size=8)
    plt.ylim(0, 1)

    for i, (player, color) in enumerate(zip(players, colors)):
        vals = np.append(values[i], values[i][0])
        ax.plot(angles, vals, color=color, linewidth=2, label=player)
        ax.fill(angles, vals, color=color, alpha=0.25)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

def create_bar_chart(categories, values, players, colors):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set style
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.family'] = 'Inter'
    
    # Bar positioning
    bar_width = 0.35
    x_indices = np.arange(len(categories))
    
    # Create bars for each player
    for i, (player, color) in enumerate(zip(players, colors)):
        offset = bar_width * i
        ax.bar(x_indices + offset, values[i], bar_width, 
               label=player, color=color, alpha=0.9,
               edgecolor='white', linewidth=1.2)
    
    # Formatting
    ax.set_title('Player Comparison', fontsize=16, pad=20)
    ax.set_xticks(x_indices + bar_width/2)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.legend(frameon=True, facecolor='white')
    
    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add value labels
    for bars in ax.containers:
        ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=8)
    
    plt.tight_layout()
    
    # Save to base64
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=120, bbox_inches='tight')
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

@app.route('/', methods=['GET', 'POST'])
def index():
    single_stats = None
    compare_stats = []
    radar_img = None
    bar_img = None
    players = []
    colors = ['#2ecc71', '#e74c3c']
    
    if request.method == 'POST':
        # Single Player Analysis
        if 'single_submit' in request.form:
            player_name = request.form['single_player'].strip()
            format_type = request.form['single_format']
            single_stats = get_player_stats(player_name, format_type)
            if single_stats:
                max_val = max(single_stats.values()) or 1
                norm_stats = [v/max_val for v in single_stats.values()]
                radar_img = create_radar_chart(
                    list(single_stats.keys()), 
                    [norm_stats], 
                    [player_name], 
                    [colors[0]]
                )
                # Generate bar chart for single player
                bar_img = create_bar_chart(
                    list(single_stats.keys()),
                    [norm_stats],
                    [player_name],
                    [colors[0]]
                )

        # Player Comparison
        elif 'compare_submit' in request.form:
            format_type = request.form['compare_format']
            players = [
                request.form['player1'].strip(),
                request.form['player2'].strip()
            ]
            for player in players:
                stats = get_player_stats(player, format_type)
                if stats:
                    max_val = max(stats.values()) or 1
                    norm_stats = [v/max_val for v in stats.values()]
                    compare_stats.append(norm_stats)
                else:
                    compare_stats.append(None)
            
            if all(compare_stats):
                radar_img = create_radar_chart(
                    list(get_player_stats(players[0], format_type).keys()),
                    compare_stats,
                    players,
                    colors
                )
                # Generate bar chart for comparison
                bar_img = create_bar_chart(
                    list(get_player_stats(players[0], format_type).keys()),
                    compare_stats,
                    players,
                    colors
                )

    return render_template('index.html',
                         single_stats=single_stats,
                         compare_stats=compare_stats,
                         players=players,
                         radar_img=radar_img,
                         bar_img=bar_img,
                         colors=colors)

if __name__ == '__main__':
    app.run(debug=True)
