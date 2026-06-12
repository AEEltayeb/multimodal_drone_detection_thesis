# Drive eval_full_pipeline_singlepass.py over all 19 drone-detection-video-test clips.
# RGB-only dataset (no paired IR), so we use ir_grayscale as the IR fallback.
# Each clip runs all 3 RGB detectors x {no_classifier, sa32} + ir_grayscale.
# Scoring rule (IoP vs IoU) comes from the dataset's `scoring` field; the
# video_* datasets inherit the default ("iop") via enumerate_video_clips.

$clips = @(
  # ---- drone clips (have drone GT -> bbox-level scoring) ----
  "video_drone_drone_and_bird_sky_and_trees_short",
  "video_drone_drone_attacked_by_bird_mountain_side_view",
  "video_drone_drone_over_mountain_attacked_by_birds",
  "video_drone_drone_seagull_attack",
  "video_drone_drone_takeoff_from_ground_and_not_hand_short",
  "video_drone_drone_takeoff_short",
  "video_drone_drone_takeoff_short_trees_background_dji_air_3s_take_off_sho",
  "video_drone_flock_of_seagulls_attack_drone_beach",
  "video_drone_two_birds_drone",
  # ---- confuser-only clips (no drone GT -> frame-level scoring downstream) ----
  "video_birds_birds_flying_overhead_various_sizes_short",
  "video_birds_birds_in_slow_motion_flying_various_sizes_compilation",
  "video_birds_distant_birds_flying_in_the_sky_short",
  "video_birds_flock_of_birds_flying_short",
  "video_birds_flock_of_birds_flying_sunset",
  "video_airplanes_airplanes_compilation",
  "video_airplanes_distant_airplane_over_head_flying_away",
  "video_helicopters_helicopter_compilation",
  "video_helicopters_helicopter_overhead_short",
  "video_helicopters_helicopter_overhead_very_small_airplane_in_background"
)

$total = $clips.Count
$i = 0
foreach ($clip in $clips) {
  $i++
  Write-Host ""
  Write-Host "=== [$i / $total] $clip ===" -ForegroundColor Cyan
  python eval/eval_full_pipeline_singlepass.py `
    --dataset $clip `
    --rgb-detectors baseline retrained_v2 selcom_1280 `
    --ir-detectors ir_grayscale `
    --classifiers sa32
  if ($LASTEXITCODE -ne 0) {
    Write-Host "  FAILED on $clip (exit $LASTEXITCODE) -- continuing" -ForegroundColor Yellow
  }
}

Write-Host ""
Write-Host "All clips processed." -ForegroundColor Green
