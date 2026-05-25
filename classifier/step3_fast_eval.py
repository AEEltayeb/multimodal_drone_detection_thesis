"""Fast eval: sa32 vs realgray classifier using cached detections (no YOLO)."""
import sys, json, time
from pathlib import Path
from collections import Counter
import cv2, numpy as np, joblib
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "ir_gui"))
from fusion.features import compute_global_features, compute_target_features, TARGET_NAMES

CACHE = REPO / "docs" / "analysis" / "full_pipeline_ablations" / "cache"
VID_ROOT = REPO / "datasets" / "drone detection video tests" / "rgb"
EXTS = {".jpg",".jpeg",".png",".bmp"}
MAX = 500  # per dataset

def load_clf(p):
    o = joblib.load(str(p))
    return o["model"], o.get("features") or o.get("feat_cols") or []

sa32_m, sa32_f = load_clf(REPO/"classifier"/"fusion_models"/"scene_aware_v3more_32feat"/"model.joblib")
gray_m, gray_f = load_clf(REPO/"classifier"/"runs"/"reliability"/"fusion"/"fusion_no_fn_v3more_realgray_model.joblib")

def read_gt(lp, w, h):
    boxes = []
    if not lp.exists(): return boxes
    for line in lp.read_text().strip().split("\n"):
        p = line.strip().split()
        if len(p)<5: continue
        cx,cy,bw,bh = float(p[1]),float(p[2]),float(p[3]),float(p[4])
        boxes.append([(cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h])
    return boxes

def iop_match(dets, gts, thr=0.5):
    tp=fp=fn=0
    gt_matched=[False]*len(gts)
    for d in sorted(dets, key=lambda x:-x[4]):
        hit=False
        for gi,g in enumerate(gts):
            if gt_matched[gi]: continue
            x1=max(d[0],g[0]);y1=max(d[1],g[1]);x2=min(d[2],g[2]);y2=min(d[3],g[3])
            inter=max(0,x2-x1)*max(0,y2-y1)
            da=max(1e-6,(d[2]-d[0])*(d[3]-d[1]))
            if inter/da>=thr:
                tp+=1; gt_matched[gi]=True; hit=True; break
        if not hit: fp+=1
    fn=sum(1 for m in gt_matched if not m)
    return tp,fp,fn

def build_feats(rgb_dets, ir_dets, gray, w, h, ir_gray=None):
    if ir_gray is None: ir_gray = gray
    f = {}
    rc=[d[4] for d in rgb_dets]; ic=[d[4] for d in ir_dets]
    f["rgb_max_conf"]=max(rc) if rc else 0.0; f["rgb_mean_conf"]=float(np.mean(rc)) if rc else 0.0
    f["ir_max_conf"]=max(ic) if ic else 0.0; f["ir_mean_conf"]=float(np.mean(ic)) if ic else 0.0
    g=compute_global_features(gray)
    f.update({f"rgb_{k}":v for k,v in g.items()})
    ig=compute_global_features(ir_gray)
    f.update({f"ir_{k}":v for k,v in ig.items()})
    for pfx,dets,gy in [("rgb",rgb_dets,gray),("ir",ir_dets,ir_gray)]:
        if not dets:
            f.update({f"{pfx}_best_{k}":0.0 for k in TARGET_NAMES})
        else:
            b=max(dets,key=lambda d:d[4])
            tf=compute_target_features(gy,b[:4],w,h)
            f.update({f"{pfx}_best_{k}":v for k,v in tf.items()})
    return f

def to_x(f, cols):
    return np.array([[f.get(c,0.0) for c in cols]],dtype=np.float32)

# Datasets with caches
datasets = []
# Paired: svanstrom, antiuav
for tag,rgb_cache_tag,ir_cache_tag,rgb_dir,lbl_dir,ir_dir,ir_lbl_dir in [
    ("svanstrom","svanstrom_baseline_sz1280","svanstrom_ir_model_sz640",
     Path("G:/drone/svanstrom_paired/RGB/images"),Path("G:/drone/svanstrom_paired/RGB/labels"),
     Path("G:/drone/svanstrom_paired/IR/images"),Path("G:/drone/svanstrom_paired/IR/labels")),
    ("antiuav","antiuav_baseline_sz1280","antiuav_ir_native_sz640",
     Path("G:/drone/Anti-UAV-RGBT_yolo_converted/test/RGB/images"),
     Path("G:/drone/Anti-UAV-RGBT_yolo_converted/test/RGB/labels"),
     Path("G:/drone/Anti-UAV-RGBT_yolo_converted/test/IR/images"),
     Path("G:/drone/Anti-UAV-RGBT_yolo_converted/test/IR/labels")),
]:
    rc=CACHE/f"{rgb_cache_tag}.json"; ic=CACHE/f"{ir_cache_tag}.json"
    if rc.exists() and ic.exists():
        datasets.append({"name":tag,"type":"paired","rgb_cache":rc,"ir_cache":ic,
                         "rgb_dir":rgb_dir,"lbl_dir":lbl_dir,"ir_dir":ir_dir,"ir_lbl_dir":ir_lbl_dir})

# Grayscale video clips
for cat in ("drone","birds","airplanes","helicopters"):
    cd=VID_ROOT/cat
    if not cd.exists(): continue
    for clip in sorted(cd.iterdir()):
        if not clip.is_dir(): continue
        tag=f"video_{cat}_{clip.name}"
        rc=CACHE/f"{tag}_baseline_sz1280.json"; ic=CACHE/f"{tag}_ir_grayscale_sz640.json"
        img_dir=clip/"images"/"test" if (clip/"images"/"test").exists() else clip/"images"
        lbl_dir=clip/"labels"/"test" if (clip/"labels"/"test").exists() else clip/"labels"
        if rc.exists() and ic.exists() and img_dir.exists():
            datasets.append({"name":tag,"type":"grayscale","rgb_cache":rc,"ir_cache":ic,
                             "rgb_dir":img_dir,"lbl_dir":lbl_dir})

import re
def strip_mod(s): return re.sub(r"_(visible|infrared)","",s)

results = {}
for ds in datasets:
    rc=json.load(open(ds["rgb_cache"]))["dets"]
    ic=json.load(open(ds["ir_cache"]))["dets"]
    
    imgs=sorted(p for p in ds["rgb_dir"].iterdir() if p.suffix.lower() in EXTS)
    if len(imgs)>MAX: imgs=imgs[::max(1,len(imgs)//MAX)]
    
    counters={n:{"labels":Counter(),"tp":0,"fp":0,"fn":0} for n in ("sa32","realgray")}
    nf=0; t0=time.time()
    
    for img_path in imgs:
        stem=img_path.stem
        rd=rc.get(stem,[]); ird=ic.get(stem,[])
        img=cv2.imread(str(img_path))
        if img is None: continue
        h,w=img.shape[:2]; gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        rgb_dets=[d for d in rd if d[4]>=0.25]
        ir_dets=[d for d in ird if d[4]>=0.40]
        gt=read_gt(ds["lbl_dir"]/f"{stem}.txt",w,h)
        
        # For paired, get IR image
        ir_gray=gray
        ir_gts=gt
        if ds["type"]=="paired":
            base=strip_mod(stem)
            # Find IR image
            for ext in (".jpg",".png",".bmp",".jpeg"):
                candidates=[ds["ir_dir"]/f"{base}_infrared{ext}",ds["ir_dir"]/f"{base}{ext}"]
                for cp in candidates:
                    if cp.exists():
                        irim=cv2.imread(str(cp))
                        if irim is not None:
                            ir_gray=cv2.cvtColor(irim,cv2.COLOR_BGR2GRAY)
                            ir_gts=read_gt(ds["ir_lbl_dir"]/f"{cp.stem}.txt",irim.shape[1],irim.shape[0])
                        break
        
        feats=build_feats(rgb_dets,ir_dets,gray,w,h,ir_gray)
        nf+=1
        
        for cn,model,fc in [("sa32",sa32_m,sa32_f),("realgray",gray_m,gray_f)]:
            x=to_x(feats,fc)
            try: label=int(model.predict(x)[0])
            except: label=3
            counters[cn]["labels"][label]+=1
            
            if ds["type"]=="paired":
                if label in (1,3):
                    tp,fp,fn=iop_match(rgb_dets,gt); counters[cn]["tp"]+=tp; counters[cn]["fp"]+=fp; counters[cn]["fn"]+=fn
                elif label==0:
                    counters[cn]["fn"]+=len(gt)
                if label in (2,3):
                    tp,fp,fn=iop_match(ir_dets,ir_gts); counters[cn]["tp"]+=tp; counters[cn]["fp"]+=fp; counters[cn]["fn"]+=fn
                elif label==0:
                    counters[cn]["fn"]+=len(ir_gts)
            else:
                if label==0: kept=[]
                elif label==1: kept=rgb_dets
                elif label==2: kept=ir_dets
                else: kept=rgb_dets+ir_dets
                tp,fp,fn=iop_match(kept,gt); counters[cn]["tp"]+=tp; counters[cn]["fp"]+=fp; counters[cn]["fn"]+=fn
    
    el=time.time()-t0
    print(f"\n{'='*70}\n{ds['name']} ({nf} frames, {el:.0f}s)\n{'='*70}")
    ds_r={"n":nf,"type":ds["type"]}
    for cn in ("sa32","realgray"):
        c=counters[cn]; tp,fp,fn=c["tp"],c["fp"],c["fn"]
        p=tp/(tp+fp) if tp+fp else 0; r=tp/(tp+fn) if tp+fn else 0; f1=2*p*r/(p+r) if p+r else 0
        rej=c["labels"][0]; rp=rej/nf*100 if nf else 0
        print(f"  {cn:12s} TP={tp:>5d} FP={fp:>5d} FN={fn:>5d} P={p:.4f} R={r:.4f} F1={f1:.4f} rej={rej}/{nf} ({rp:.1f}%)")
        ds_r[cn]={"tp":tp,"fp":fp,"fn":fn,"P":round(p,4),"R":round(r,4),"F1":round(f1,4),"rej":round(rp/100,4)}
    results[ds["name"]]=ds_r

# Aggregates
print(f"\n{'='*70}\nAGGREGATE\n{'='*70}")
for grp in ("paired","grayscale"):
    dl=[k for k,v in results.items() if v["type"]==grp]
    if not dl: continue
    for cn in ("sa32","realgray"):
        tp=sum(results[d][cn]["tp"] for d in dl); fp=sum(results[d][cn]["fp"] for d in dl); fn=sum(results[d][cn]["fn"] for d in dl)
        n=sum(results[d]["n"] for d in dl)
        p=tp/(tp+fp) if tp+fp else 0; r=tp/(tp+fn) if tp+fn else 0; f1=2*p*r/(p+r) if p+r else 0
        print(f"  {grp:10s} {cn:12s} n={n:>5d} P={p:.4f} R={r:.4f} F1={f1:.4f}")

out=REPO/"eval"/"results"/"realgray_clf_comparison.json"
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(results,indent=2))
print(f"\nSaved: {out}")
