**Target Variable** - `pitch_type`
    -> will be of the following values: FS, FF, SL (split-finger, 4-seam fastball, slider)
    -> want to be able to predict which of these pitches will be thrown next

**Features**:
`prev_pitch_type`: important factor in sequencing -- categorical

`prev_release_speed`: velocity value of the last pitch thrown (MPH) -- numerical

`batter`: MLBAM ID of the batter  -- numerical  

`stand` (handedness of the hitter): either L or R -- categorical 

`balls` & `strikes` (what's the count): `range[0, 4]` for balls, `range[0, 3]` for strikes -- numerical

`on\_3b`, `on\_2b`, `on\_1b` (just to determine if anyone is on base): bool value for on-base -- categorical

`outs_when_up`: range[0, 3] -- numerical 

`prev_release_spin_rate` (of the last pitch): rpm value of the last pitch thrown (usually in the thousands)  -- numerical

`at_bat_number`: which AB it is in the game (between both teams) -- numerical

`pitch_number`: pitch number in the AB  -- numerical

`bat_score`: score of the batting team -- numerical 

`field_score`: score of the fielding team (maybe both should be replaced by run differential, not sure) -- numerical

`n_priorpa_thisgame_player_at_bat`: how many PA the hitter had in the game before this -- int value -- numerical 

`pitcher_days_since_prev_game`: days since prev. game -- int value -- numerical 

`p_throws`: handedness of the pitcher, may not be necessary? but no other way for the model to know... -- categorical

