"""
Configuration and constants for the Pressing Intensity & Trigger Detection system.
Strict physical constraints, causal temporal parameters, and realistic thresholds.
"""

from dataclasses import dataclass

# Pitch dimensions in meters (Standard FIFA / Metrica coordinates standard)
PITCH_LENGTH_METERS: float = 105.0
PITCH_WIDTH_METERS: float = 68.0

# Frame rate of tracking data (Metrica standard is 25 Hz)
SAMPLING_RATE_HZ: int = 25
FRAME_INTERVAL_SEC: float = 1.0 / SAMPLING_RATE_HZ

# Full 90-minute Match Parameters
MATCH_DURATION_MINUTES: float = 90.0
MATCH_DURATION_SECONDS: float = MATCH_DURATION_MINUTES * 60.0  # 5,400 seconds
FULL_MATCH_FRAMES: int = int(MATCH_DURATION_SECONDS * SAMPLING_RATE_HZ)  # 135,000 frames
HALF_MATCH_FRAMES: int = FULL_MATCH_FRAMES // 2  # 67,500 frames per half (45 mins)

# Physical Kinematic Bounds
MAX_REALISTIC_PLAYER_SPEED_M_S: float = 10.5   # World-class top sprinting speed (~37.8 km/h)
MAX_REALISTIC_BALL_SPEED_M_S: float = 36.0     # Maximum shot/pass velocity (~130 km/h)
MAX_REALISTIC_ACCELERATION_M_S2: float = 4.5   # Maximum human acceleration

# Possession Hysteresis Parameters
POSSESSION_BALL_PROXIMITY_M: float = 2.0        # Player must be within 2.0m for direct ball control
POSSESSION_MIN_HOLD_FRAMES: int = 38           # Sustained possession required: 1.5s (38 frames at 25Hz)
TURNOVER_MIN_HOLD_FRAMES: int = 50             # New team must sustain control for 2.0s (50 frames)

# Feature extraction lookback window (Strictly Backward / Causal)
FEATURE_LOOKBACK_SECONDS: float = 1.5
FEATURE_LOOKBACK_FRAMES: int = int(FEATURE_LOOKBACK_SECONDS * SAMPLING_RATE_HZ)

# Target label definition: weak supervision lookahead window
TURNOVER_LOOKAHEAD_SECONDS: float = 4.0
TURNOVER_LOOKAHEAD_FRAMES: int = int(TURNOVER_LOOKAHEAD_SECONDS * SAMPLING_RATE_HZ)

# Tactical Pitch Thirds
DEFENSIVE_THIRD_THRESHOLD: float = PITCH_LENGTH_METERS * (1.0 / 3.0)   # 35m
MIDDLE_THIRD_THRESHOLD: float = PITCH_LENGTH_METERS * (2.0 / 3.0)      # 70m
ATTACKING_60_PCT_THRESHOLD: float = PITCH_LENGTH_METERS * 0.40         # 42m for traditional PPDA zone

@dataclass
class PitchDimensions:
    length: float = PITCH_LENGTH_METERS
    width: float = PITCH_WIDTH_METERS
    x_min: float = 0.0
    x_max: float = PITCH_LENGTH_METERS
    y_min: float = 0.0
    y_max: float = PITCH_WIDTH_METERS
