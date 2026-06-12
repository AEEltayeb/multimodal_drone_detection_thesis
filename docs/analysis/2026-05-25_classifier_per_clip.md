# Per-clip classifier evaluation (2026-05-25)

Each classifier evaluated on every drone-video-tests clip. `split` column shows whether that clip was in TRAIN or TEST for that specific classifier (per seed=42 GroupShuffleSplit of its own training CSV); `n/a` for 32-feat/40-feat which used a different training corpus.


## Per-clip accuracy

| clip | category | n | gt_pos | lean10 | lean13 | lean17 | lean19 | 32feat | 40feat |
|---|---|---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `video_airplanes_airplanes_compilation` | airplanes | 249 | 0 | 0.992🅣 | 1.000🅣 | 1.000🅣 | 1.000🅣 | 1.000 | 0.799 |
| `video_airplanes_distant_airplane_over_head_flying_away` | airplanes | 55 | 0 | 1.000🅣 | 1.000🅣 | 1.000🅣 | 0.727🅗 | 1.000 | 0.709 |
| `video_birds_birds_flying_overhead_various_sizes_short` | birds | 20 | 0 | 1.000🅣 | 1.000🅣 | 1.000🅣 | 1.000🅣 | 1.000 | 0.900 |
| `video_birds_birds_in_slow_motion_flying_various_sizes_compilation` | birds | 271 | 0 | 0.985🅣 | 1.000🅣 | 1.000🅣 | 1.000🅣 | 1.000 | 0.978 |
| `video_birds_distant_birds_flying_in_the_sky_short` | birds | 20 | 0 | 1.000🅣 | 1.000🅣 | 1.000🅣 | 1.000🅣 | 1.000 | 0.600 |
| `video_birds_flock_of_birds_flying_short` | birds | 21 | 0 | 1.000🅣 | 1.000🅣 | 1.000🅣 | 1.000🅣 | 1.000 | 0.810 |
| `video_birds_flock_of_birds_flying_sunset` | birds | 20 | 0 | 1.000🅣 | 1.000🅣 | 1.000🅣 | 1.000🅣 | 1.000 | 1.000 |
| `video_drone_drone_and_bird_sky_and_trees_short` | drone | 114 | 79 | 0.561🅗 | 0.377🅗 | 0.421🅗 | 0.596🅗 | 0.377 | 0.877 |
| `video_drone_drone_attacked_by_bird_mountain_side_view` | drone | 108 | 71 | 0.898🅣 | 1.000🅣 | 1.000🅣 | 0.991🅣 | 0.343 | 0.500 |
| `video_drone_drone_over_mountain_attacked_by_birds` | drone | 68 | 35 | 1.000🅣 | 1.000🅣 | 1.000🅣 | 1.000🅣 | 0.691 | 0.765 |
| `video_drone_drone_seagull_attack` | drone | 235 | 186 | 0.996🅣 | 0.996🅣 | 1.000🅣 | 0.566🅗 | 0.421 | 0.638 |
| `video_drone_drone_takeoff_from_ground_and_not_hand_short` | drone | 163 | 139 | 0.994🅣 | 0.994🅣 | 0.994🅣 | 0.994🅣 | 0.558 | 0.859 |
| `video_drone_drone_takeoff_short` | drone | 116 | 101 | 0.957🅣 | 1.000🅣 | 1.000🅣 | 1.000🅣 | 0.241 | 0.716 |
| `video_drone_drone_takeoff_short_trees_background_dji_air_3s_take_off_sho` | drone | 166 | 128 | 0.994🅣 | 0.994🅣 | 1.000🅣 | 0.994🅣 | 0.518 | 0.867 |
| `video_drone_flock_of_seagulls_attack_drone_beach` | drone | 239 | 165 | 0.933🅣 | 0.992🅣 | 1.000🅣 | 0.987🅣 | 0.310 | 0.632 |
| `video_drone_two_birds_drone` | drone | 150 | 140 | 0.387🅗 | 0.127🅗 | 0.120🅗 | 0.993🅣 | 0.067 | 0.513 |
| `video_helicopters_helicopter_compilation` | helicopters | 554 | 0 | 0.986🅣 | 1.000🅣 | 1.000🅣 | 0.998🅣 | 0.978 | 0.888 |
| `video_helicopters_helicopter_overhead_short` | helicopters | 20 | 0 | 0.950🅗 | 0.800🅗 | 0.750🅗 | 1.000🅣 | 0.800 | 0.500 |
| `video_helicopters_helicopter_overhead_very_small_airplane_in_background` | helicopters | 20 | 0 | 1.000🅣 | 1.000🅣 | 1.000🅣 | 1.000🅣 | 1.000 | 1.000 |

Legend: 🅣 = clip in classifier's TRAIN split  🅗 = clip in classifier's TEST split  — = clip not in classifier's training data  (blank) = classifier trained on a different corpus


## Per-clip F1-macro

| clip | category | n | gt_pos | lean10 | lean13 | lean17 | lean19 | 32feat | 40feat |
|---|---|---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `video_airplanes_airplanes_compilation` | airplanes | 249 | 0 | 0.498🅣 | 1.000🅣 | 1.000🅣 | 1.000🅣 | 1.000 | 0.222 |
| `video_airplanes_distant_airplane_over_head_flying_away` | airplanes | 55 | 0 | 1.000🅣 | 1.000🅣 | 1.000🅣 | 0.421🅗 | 1.000 | 0.415 |
| `video_birds_birds_flying_overhead_various_sizes_short` | birds | 20 | 0 | 1.000🅣 | 1.000🅣 | 1.000🅣 | 1.000🅣 | 1.000 | 0.316 |
| `video_birds_birds_in_slow_motion_flying_various_sizes_compilation` | birds | 271 | 0 | 0.496🅣 | 1.000🅣 | 1.000🅣 | 1.000🅣 | 1.000 | 0.494 |
| `video_birds_distant_birds_flying_in_the_sky_short` | birds | 20 | 0 | 1.000🅣 | 1.000🅣 | 1.000🅣 | 1.000🅣 | 1.000 | 0.250 |
| `video_birds_flock_of_birds_flying_short` | birds | 21 | 0 | 1.000🅣 | 1.000🅣 | 1.000🅣 | 1.000🅣 | 1.000 | 0.447 |
| `video_birds_flock_of_birds_flying_sunset` | birds | 20 | 0 | 1.000🅣 | 1.000🅣 | 1.000🅣 | 1.000🅣 | 1.000 | 1.000 |
| `video_drone_drone_and_bird_sky_and_trees_short` | drone | 114 | 79 | 0.458🅗 | 0.243🅗 | 0.277🅗 | 0.437🅗 | 0.326 | 0.615 |
| `video_drone_drone_attacked_by_bird_mountain_side_view` | drone | 108 | 71 | 0.918🅣 | 1.000🅣 | 1.000🅣 | 0.992🅣 | 0.170 | 0.379 |
| `video_drone_drone_over_mountain_attacked_by_birds` | drone | 68 | 35 | 1.000🅣 | 1.000🅣 | 1.000🅣 | 1.000🅣 | 0.512 | 0.675 |
| `video_drone_drone_seagull_attack` | drone | 235 | 186 | 0.996🅣 | 0.996🅣 | 1.000🅣 | 0.403🅗 | 0.258 | 0.443 |
| `video_drone_drone_takeoff_from_ground_and_not_hand_short` | drone | 163 | 139 | 0.993🅣 | 0.993🅣 | 0.993🅣 | 0.993🅣 | 0.348 | 0.769 |
| `video_drone_drone_takeoff_short` | drone | 116 | 101 | 0.943🅣 | 1.000🅣 | 1.000🅣 | 1.000🅣 | 0.214 | 0.513 |
| `video_drone_drone_takeoff_short_trees_background_dji_air_3s_take_off_sho` | drone | 166 | 128 | 0.994🅣 | 0.994🅣 | 1.000🅣 | 0.994🅣 | 0.387 | 0.650 |
| `video_drone_flock_of_seagulls_attack_drone_beach` | drone | 239 | 165 | 0.957🅣 | 0.994🅣 | 1.000🅣 | 0.992🅣 | 0.118 | 0.503 |
| `video_drone_two_birds_drone` | drone | 150 | 140 | 0.360🅗 | 0.098🅗 | 0.094🅗 | 0.996🅣 | 0.031 | 0.434 |
| `video_helicopters_helicopter_compilation` | helicopters | 554 | 0 | 0.331🅣 | 1.000🅣 | 1.000🅣 | 0.500🅣 | 0.247 | 0.235 |
| `video_helicopters_helicopter_overhead_short` | helicopters | 20 | 0 | 0.487🅗 | 0.296🅗 | 0.286🅗 | 1.000🅣 | 0.296 | 0.222 |
| `video_helicopters_helicopter_overhead_very_small_airplane_in_background` | helicopters | 20 | 0 | 1.000🅣 | 1.000🅣 | 1.000🅣 | 1.000🅣 | 1.000 | 1.000 |

## Drone clips where the model never saw any frame (true OOD)

| clip | n | lean10 | lean13 | lean17 | lean19 | 32feat | 40feat |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `video_drone_drone_and_bird_sky_and_trees_short` | 114 | 0.561 (test) | 0.377 (test) | 0.421 (test) | 0.596 (test) | _train_ | _train_ |
| `video_drone_drone_seagull_attack` | 235 | _train_ | _train_ | _train_ | 0.566 (test) | _train_ | _train_ |
| `video_drone_two_birds_drone` | 150 | 0.387 (test) | 0.127 (test) | 0.120 (test) | _train_ | _train_ | _train_ |