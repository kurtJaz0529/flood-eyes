import sys, json, urllib.request, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import rasterio, numpy as np
from rasterio.windows import from_bounds
from rasterio.warp import transform as wt

def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
        headers={"Content-Type":"application/json","User-Agent":"Mozilla/5.0"})
    return json.load(urllib.request.urlopen(req, timeout=45))

BBOX = [84.9, 27.8, 85.8, 28.7]
r = post("https://planetarycomputer.microsoft.com/api/stac/v1/search",
         {"collections":["sentinel-1-rtc"], "bbox":BBOX, "datetime":"2026-08-10T00:00:00Z/2026-09-05T00:00:00Z", "limit":40})
feats = r["features"]
tok = json.load(urllib.request.urlopen("https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-1-rtc", timeout=30))["token"]

def pick(date):
    return min(feats, key=lambda f: abs((np.datetime64(f["properties"]["datetime"][:10]) - np.datetime64(date)) / np.timedelta64(1,"D")))

LON, LAT, N = 85.368, 28.247, 2048   # 20.5 km 窗口
res = {}
for tag, date in (("pre","2026-08-16"), ("post","2026-08-28")):
    it = pick(date)
    print(f"{tag}: {it['id'][:52]}  {it['properties']['datetime'][:16]}")
    for pol in ("vv","vh"):
        href = it["assets"][pol]["href"] + "?" + tok
        t0=time.time()
        with rasterio.open("/vsicurl/" + href) as src:
            xs, ys = wt("EPSG:4326", src.crs, [LON],[LAT])
            win = from_bounds(xs[0]-N*5, ys[0]-N*5, xs[0]+N*5, ys[0]+N*5, src.transform)
            a = src.read(1, window=win).astype(np.float32)
            crs, tr = src.crs, src.window_transform(win)
        db = 10*np.log10(np.maximum(a, 1e-6))
        res[(tag,pol)] = db
        print(f"   {pol.upper()} {a.shape} dB: 1%={np.percentile(db,1):.1f} 中位={np.median(db):.1f} 99%={np.percentile(db,99):.1f}  {time.time()-t0:.1f}s")

print("\n=== 变化检测（VV）===")
pre, post = res[("pre","vv")], res[("post","vv")]
drop = post - pre
print(f"ΔdB: 1%={np.percentile(drop,1):.1f} 中位={np.median(drop):.2f} | Δ<-4 占比 {100*(drop<-4).mean():.2f}%")
for th, d in [(-16,-4),(-18,-4),(-20,-4),(-16,-6)]:
    m = (post < th) & (drop < d)
    print(f"  post<{th} & Δ<{d}: 候选 {100*m.mean():.2f}%")
