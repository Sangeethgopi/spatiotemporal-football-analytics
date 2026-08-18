import pandas as pd
import numpy as np
import argparse
SAMPLING_RATE_HZ = 25.0

def cv_to_metrica(cv_csv: str, output_home: str, output_away: str):
    """
    Converts the long-format output of the CV pipeline into the wide-format
    expected by the Metrica Sports analysis pipeline.
    """
    print(f"Translating {cv_csv} to Metrica schema...")
    df = pd.read_csv(cv_csv)
    
    # Heuristic: Find which cluster is which
    # Assuming ball is not clustered, and referees are fewest in number.
    players = df[df["class"] == "player"].copy()
    cluster_counts = players["team_cluster"].value_counts()
    
    # 0 and 1 are the two biggest clusters (Home/Away). 
    # Arbitrarily assign the largest to Home, second to Away.
    home_cluster = cluster_counts.index[0]
    away_cluster = cluster_counts.index[1]
    
    # Group by frame to create wide format
    frames = df["frame"].unique()
    frames = sorted(frames)
    
    home_data = []
    away_data = []
    
    for f in frames:
        f_df = df[df["frame"] == f]
        
        home_row = {"Period": 1, "Frame": f, "Time [s]": f / float(SAMPLING_RATE_HZ)}
        away_row = {"Period": 1, "Frame": f, "Time [s]": f / float(SAMPLING_RATE_HZ)}
        
        ball_df = f_df[f_df["class"] == "ball"]
        if not ball_df.empty:
            bx = ball_df.iloc[0]["pitch_x"]
            by = ball_df.iloc[0]["pitch_y"]
            home_row["Ball_x"] = bx
            home_row["Ball_y"] = by
            away_row["Ball_x"] = bx
            away_row["Ball_y"] = by
        else:
            home_row["Ball_x"] = np.nan
            home_row["Ball_y"] = np.nan
            away_row["Ball_x"] = np.nan
            away_row["Ball_y"] = np.nan
            
        p_df = f_df[f_df["class"] == "player"]
        for _, row in p_df.iterrows():
            tid = int(row["track_id"])
            px = row["pitch_x"]
            py = row["pitch_y"]
            cluster = row["team_cluster"]
            
            if cluster == home_cluster:
                home_row[f"Home_{tid}_x"] = px
                home_row[f"Home_{tid}_y"] = py
            elif cluster == away_cluster:
                away_row[f"Away_{tid}_x"] = px
                away_row[f"Away_{tid}_y"] = py
                
        home_data.append(home_row)
        away_data.append(away_row)
        
    df_home = pd.DataFrame(home_data)
    df_away = pd.DataFrame(away_data)
    
    def write_metrica_format(df_out, filepath, team_name):
        cols = list(df_out.columns)
        
        row0 = []
        row1 = []
        row2 = []
        
        for c in cols:
            if c in ["Period", "Frame", "Time [s]"]:
                row0.append("")
                row1.append("")
                row2.append(c)
            elif "Ball" in c:
                row0.append("")
                row1.append("")
                row2.append(c)
            else:
                # c is like Home_11_x or Away_25_y
                parts = c.split("_")
                t = parts[0]
                num = parts[1]
                coord = parts[2]
                row0.append(t)
                row1.append(num)
                row2.append(f"Player{num}" if coord == "x" else "")
                
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(",".join(row0) + "\n")
            f.write(",".join(row1) + "\n")
            f.write(",".join(row2) + "\n")
            
        # Write data without header
        df_out.to_csv(filepath, mode="a", index=False, header=False)

    write_metrica_format(df_home, output_home, "Home")
    write_metrica_format(df_away, output_away, "Away")
    
    print(f"Translation complete. Saved to {output_home} and {output_away}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/sample_final_2d.csv")
    parser.add_argument("--home", type=str, default="data/cv_home.csv")
    parser.add_argument("--away", type=str, default="data/cv_away.csv")
    args = parser.parse_args()
    
    cv_to_metrica(args.input, args.home, args.away)
