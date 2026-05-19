import json
data = json.load(open("eval/results/pipeline_video_tests/pipeline_comparison.json"))
models = ["baseline_trained","retrained_v2","selcom_1280","selcom_640"]

for m in models:
    entries = [r for r in data if r["rgb_model"]==m]
    has_seg = all("seg_final" in r for r in entries)
    print(f"{m}: {len(entries)} entries, has_seg={has_seg}")

print("\nDRONE AGGREGATE:")
for mn in models:
    mr = [r for r in data if not r["is_negative"] and r["rgb_model"]==mn]
    if not mr: continue
    for stage in ["rgb_yolo","ir_yolo","after_classifier"]:
        tp=sum(r[stage]["TP"] for r in mr); fp=sum(r[stage]["FP"] for r in mr); fn=sum(r[stage]["FN"] for r in mr)
        p=tp/max(tp+fp,1); rec=tp/max(tp+fn,1); f1=2*p*rec/max(p+rec,1e-9)
        print(f"  {mn:18s} {stage:20s} TP={tp:4d} FP={fp:3d} FN={fn:4d} P={p:.3f} R={rec:.3f} F1={f1:.3f}")
    for seg in ["seg_temporal","seg_final"]:
        tp=sum(r[seg]["TP"] for r in mr); fp=sum(r[seg]["FP"] for r in mr); fn=sum(r[seg]["FN"] for r in mr)
        ns=sum(r[seg]["segments"] for r in mr)
        p=tp/max(tp+fp,1); rec=tp/max(tp+fn,1); f1=2*p*rec/max(p+rec,1e-9)
        print(f"  {mn:18s} {seg:20s} segs={ns:4d} TP={tp:4d} FP={fp:3d} FN={fn:4d} P={p:.3f} R={rec:.3f} F1={f1:.3f}")
    al=sum(r["alert_events"] for r in mr); ve=sum(r["vetoed_alerts"] for r in mr)
    print(f"  {mn:18s} alerts: passed={al} vetoed={ve} total={al+ve}")
    print()

print("CONFUSER AGGREGATE:")
for mn in models:
    mr = [r for r in data if r["is_negative"] and r["rgb_model"]==mn]
    if not mr: continue
    tot=sum(r["total_frames"] for r in mr)
    for stage in ["rgb_yolo","ir_yolo","after_classifier"]:
        fp=sum(r[stage]["FP"] for r in mr)
        print(f"  {mn:18s} {stage:20s} FP={fp:4d}/{tot} FPR={fp/tot:.3f}")
    for seg in ["seg_temporal","seg_final"]:
        fp=sum(r[seg]["FP"] for r in mr); ns=sum(r[seg]["segments"] for r in mr)
        print(f"  {mn:18s} {seg:20s} FP_segs={fp:4d}/{ns} FPR={fp/max(ns,1):.3f}")
    al=sum(r["alert_events"] for r in mr); ve=sum(r["vetoed_alerts"] for r in mr)
    raw=sum(r["rgb_yolo"]["FP"] for r in mr)
    red=(1-al/max(raw,1))*100
    print(f"  {mn:18s} alerts: passed={al} vetoed={ve} reduction={red:.1f}%")
    print()
