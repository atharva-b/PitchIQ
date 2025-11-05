Target Variable - pitch_type
    -> will be of the following values: FS, FF, SL (split-finger, 4-seam fastball, slider)
    -> want to be able to predict which of these pitches will be thrown next

Features:
release_speed (of the last pitch) -- velocity value of the last pitch thrown (MPH) 
batter -- ID of the batter
stand (handedness of the hitter) -- either L or R
balls & strikes (what's the count) -- range[0, 4] for balls, range[0, 3] for strikes
on\_3b, on\_2b, on\_1b (just to determine if anyone is on base) -- bool value for on-base
outs_when_up -- range[0, 3]
release_spin_rate (of the last pitch) -- rpm value of the last pitch thrown (usually in the thousands)
at_bat_number -- which AB it is in the game (between both teams) -- int value
pitch_number -- pitch number in the AB -- int value 
bat_score -- score of the batting team
field_score -- score of the fielding team (maybe both should be replaced by run differential, not sure)
n_priorpa_thisgame_player_at_bat -- how many PA the hitter had in the game before this -- int value
pitcher_days_since_prev_game -- days since prev. game -- int value



