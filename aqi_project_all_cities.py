"""
AQI Analysis & Prediction - ALL INDIAN CITIES - Data Science Mini Project
Dataset: Kaggle 'Air Quality Data in India (2015-2020)' -> city_day.csv
Statistics, feature selection, PCA, optimisation and regression are coded from scratch.
numpy/pandas only for arrays and IO, sqlite3 for SQL, matplotlib for plots.
"""
import numpy as np, pandas as pd, math, sqlite3
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
np.random.seed(7)

# ===== UNIT III: PIPELINE  extract -> batch ingest -> warehouse (SQL) -> query
raw = pd.concat(pd.read_csv("city_day.csv", chunksize=5000))
con = sqlite3.connect(":memory:"); raw.to_sql("air", con, index=False)
total = len(raw)
df = pd.read_sql("SELECT * FROM air WHERE AQI IS NOT NULL", con)
df["Date"] = pd.to_datetime(df.Date); df = df.sort_values(["City", "Date"]).reset_index(drop=True)
print(f"Raw rows: {total} | rows with AQI: {len(df)} | cities: {df.City.nunique()} | {df.Date.min().date()} to {df.Date.max().date()}")
print("Duplicates (City, Date):", int(df.duplicated(["City", "Date"]).sum()))
miss = (df.isna().mean()*100).round(1); print("Missing %:\n", miss[miss > 0].to_string())

# ===== CLEANING: drop >35% missing, per-city median impute, clip at 3*IQR
pol = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"]
drop = [c for c in pol if miss[c] > 35]; pol = [c for c in pol if c not in drop]; print("Dropped columns:", drop)
df = df.drop(columns=drop+["AQI_Bucket"])
def mean(x): return sum(x)/len(x)
def var(x): m = mean(x); return sum((v-m)**2 for v in x)/(len(x)-1)
def std(x): return math.sqrt(var(x))
def moment(x, k): m, s = mean(x), std(x); return sum(((v-m)/s)**k for v in x)/len(x)
def quant(x, q):
    x = np.sort(np.asarray(x)); k = (len(x)-1)*q; f = int(k); c = min(f+1, len(x)-1); return x[f]+(x[c]-x[f])*(k-f)
def corr(x, y):
    mx, my = mean(x), mean(y)
    return sum((a-mx)*(b-my) for a, b in zip(x, y))/math.sqrt(sum((a-mx)**2 for a in x)*sum((b-my)**2 for b in y))
for c in pol:
    df[c] = df[c].fillna(df.groupby("City")[c].transform("median")).fillna(quant(df[c].dropna().values, .5))
    q1, q3 = quant(df[c], .25), quant(df[c], .75); df[c] = df[c].clip(q1-3*(q3-q1), q3+3*(q3-q1))
print("Missing after cleaning:", int(df.isna().sum().sum()))

# ===== UNIT II: STATISTICS
print("\nVariable   mean    std   skew  kurt")
for c in ["PM2.5", "PM10", "NO2", "CO", "AQI"]:
    v = df[c].values; print(f"{c:8s} {mean(v):7.1f} {std(v):6.1f} {moment(v,3):5.2f} {moment(v,4):5.2f}")
cm = df.groupby("City").AQI.mean().sort_values(ascending=False)
print("Most polluted:", cm.head(3).round(0).to_dict(), "| Cleanest:", cm.tail(3).round(0).to_dict())
def ncdf(z): return 0.5*(1+math.erf(z/math.sqrt(2)))
N = len(df); gm = mean(df.AQI.values)
sm = [mean(np.random.choice(df.AQI.values, 30)) for _ in range(2000)]
print(f"CLT: std of sample means={std(sm):.2f} vs sigma/sqrt(n)={std(df.AQI.values)/math.sqrt(30):.2f}")
df["Month"] = df.Date.dt.month; df["Year"] = df.Date.dt.year
df["Season"] = df.Month.map(lambda m: "Winter" if m in (11, 12, 1, 2) else ("Monsoon" if m in (7, 8, 9) else "Other"))
w, mo = df[df.Season == "Winter"].AQI.values, df[df.Season == "Monsoon"].AQI.values
t = (mean(w)-mean(mo))/math.sqrt(var(w)/len(w)+var(mo)/len(mo))
print(f"One-tailed Welch t-test (winter > monsoon): t={t:.2f}, p={1-ncdf(t):.4g}")
def anova1(groups):
    k = len(groups); n = sum(len(g) for g in groups); g_ = mean(np.concatenate(groups))
    ssb = sum(len(g)*(mean(g)-g_)**2 for g in groups); ssw = sum(sum((v-mean(g))**2 for v in g) for g in groups)
    return (ssb/(k-1))/(ssw/(n-k)), k-1, n-k
F, d1, d2 = anova1([g.AQI.values for _, g in df.groupby("Season")]); print(f"One-way ANOVA (season): F({d1},{d2})={F:.1f}")
F, d1, d2 = anova1([g.AQI.values for _, g in df.groupby("City")]); print(f"One-way ANOVA (city):   F({d1},{d2})={F:.1f}")
df["Period"] = np.where(df.Year >= 2020, "2020", "Before")
cell = [g.AQI.values for _, g in df.groupby(["Season", "Period"])]; sse = sum(sum((v-mean(g))**2 for v in g) for g in cell); dfe = N-len(cell)
sa = sum(len(g)*(mean(g.AQI.values)-gm)**2 for _, g in df.groupby("Season")); sb = sum(len(g)*(mean(g.AQI.values)-gm)**2 for _, g in df.groupby("Period"))
print(f"Two-way ANOVA: F(Season)={(sa/2)/(sse/dfe):.1f}, F(Period)={sb/(sse/dfe):.1f}")
obs = np.array([len(g) for _, g in df.groupby("Season")]); ex = obs.sum()/3; print(f"Chi-square (equal seasons): {sum((obs-ex)**2/ex):.1f}")
print(f"P(AQI>300)={mean(df.AQI.values>300):.3f} | P(AQI>300|Winter)={mean(w>300):.3f}")

# ===== UNIT IV: FEATURE ENGINEERING
df["PM_ratio"] = df["PM2.5"]/(df.PM10+1e-6); df["logCO"] = np.log1p(df.CO); df["NOx_NO2"] = df.NOx-df.NO2
df["IsWinter"] = (df.Season == "Winter").astype(float)
df["m_sin"] = np.sin(2*np.pi*df.Month/12); df["m_cos"] = np.cos(2*np.pi*df.Month/12)
df["PM25_lag1"] = df.groupby("City")["PM2.5"].shift(1).fillna(df["PM2.5"])          # per-city lag
cities = sorted(df.City.unique())
for c in cities[1:]: df["C_"+c] = (df.City == c).astype(float)                       # one-hot (manual)
num = pol+["PM_ratio", "logCO", "NOx_NO2", "IsWinter", "m_sin", "m_cos", "PM25_lag1"]
filt = [c for c in num if abs(corr(df[c].values, df.AQI.values)) > 0.2]; cand = filt+["C_"+c for c in cities[1:]]
print(f"\nFilter-selected numeric: {len(filt)} of {len(num)} | + {len(cities)-1} city dummies = {len(cand)} candidates")
df = df.sort_values("Date", kind="stable").reset_index(drop=True)                      # time order for split
cut = int(.8*N); tr, te = np.arange(cut), np.arange(cut, N); y = df.AQI.values
print("Train:", df.Date[0].date(), "to", df.Date[cut-1].date(), "| Test:", df.Date[cut].date(), "to", df.Date[N-1].date())
cand = [c for c in cand if df[c].values[tr].std() > 0.05]            # drop dummies with ~no train rows
Xall = df[cand].values.astype(float); Z = (Xall-Xall[tr].mean(0))/(Xall[tr].std(0)+1e-9)
def ols(X, y): return np.linalg.solve(X.T@X, X.T@y)
def r2(yt, yp): return 1-sum((yt-yp)**2)/sum((yt-mean(yt))**2)
sub, val = tr[:int(.8*cut)], tr[int(.8*cut):]; chosen = []
for _ in range(8):                                                                    # forward wrapper selection
    best = None
    for j in range(len(cand)):
        if j in chosen: continue
        Xa = np.c_[np.ones(N), Z[:, chosen+[j]]]; s = r2(y[val], Xa[val]@ols(Xa[sub], y[sub]))
        if best is None or s > best[0]: best = (s, j)
    chosen.append(best[1]); print(f"  forward step -> {cand[best[1]]:14s} val R2={best[0]:.3f}")
sel = [cand[j] for j in chosen]
ev = np.sort(np.linalg.eigh(np.cov(Z[:, :len(filt)].T))[0])[::-1]; ev /= ev.sum()
print("PCA (numeric features) top 3:", np.round(ev[:3], 3), "cum:", round(ev[:3].sum(), 3))

# ===== UNIT IV/V: OPTIMISATION + LINEAR REGRESSION
XA = np.c_[np.ones(N), Z[:, chosen]]; Xtr, Xte, ytr, yte = XA[tr], XA[te], y[tr], y[te]
th_ne = ols(Xtr, ytr)
def gd(X, y, lr=0.1, ep=800, lam=0.0):
    th = np.zeros(X.shape[1]); h = []
    for _ in range(ep):
        e = X@th-y; th -= lr*(X.T@e+lam*np.r_[0, th[1:]])/len(y); h.append(e@e/(2*len(y)))
    return th, h
th_gd, hist = gd(Xtr, ytr); th_rg, _ = gd(Xtr, ytr, lam=20)
def rand_search(X, y, it=12000):
    b, bl, st = np.zeros(X.shape[1]), 1e18, 20.0
    for i in range(it):
        c = b+np.random.normal(0, st, len(b)); l = mean((X@c-y)**2)
        if l < bl: b, bl = c, l
        if i % 3000 == 2999: st *= .4
    return b
th_rs = rand_search(Xtr, ytr)
print("\nModel            RMSE     MAE     R2    AdjR2")
for n_, th in [("Normal Eqn", th_ne), ("Grad Descent", th_gd), ("Ridge", th_rg), ("Random Search", th_rs)]:
    r = yte-Xte@th; R = r2(yte, Xte@th); p_ = len(sel)
    print(f"{n_:13s} {math.sqrt(mean(r**2)):8.2f} {mean(np.abs(r)):7.2f} {R:7.3f} {1-(1-R)*(len(yte)-1)/(len(yte)-p_-1):7.3f}")
print(f"Baseline (train mean) R2 = {r2(yte, np.full(len(yte), mean(ytr))):.3f}")
print("Coefficients:", {c: round(float(v), 1) for c, v in zip(["bias"]+sel, th_ne)})
print("||grad|| at optimum:", float(np.linalg.norm(Xtr.T@(Xtr@th_ne-ytr))/len(ytr)))
bucket = lambda a: next((n for l, n in [(50,"Good"),(100,"Satisfactory"),(200,"Moderate"),(300,"Poor"),(400,"Very Poor")] if a <= l), "Severe")
pred = Xte@th_ne; print(f"Category accuracy: {mean([bucket(a)==bucket(b) for a, b in zip(yte, pred)])*100:.1f}%")
tc = df.City.values[te]; pc = {c: math.sqrt(mean((yte[tc == c]-pred[tc == c])**2)) for c in cities if (tc == c).sum() > 30}
pc = dict(sorted(pc.items(), key=lambda z: z[1])); print("Best/worst city RMSE:", list(pc.items())[:2], list(pc.items())[-2:])

# ===== VISUALISATION
fig, ax = plt.subplots(2, 3, figsize=(16, 9))
ax[0,0].barh(cm.index[::-1], cm.values[::-1], color="teal"); ax[0,0].set_title("Mean AQI by city"); ax[0,0].tick_params(labelsize=6)
ax[0,1].boxplot([df[df.Month == m].AQI for m in range(1, 13)]); ax[0,1].set_title("AQI by month (all cities)")
ax[0,2].hist(df.AQI, bins=40, color="teal"); ax[0,2].set_title("AQI distribution")
cols = ["PM2.5", "PM10", "NO2", "CO", "SO2", "O3", "AQI"]; C = np.array([[corr(df[a].values, df[b].values) for b in cols] for a in cols])
ax[1,0].imshow(C, cmap="coolwarm", vmin=-1, vmax=1); ax[1,0].set_xticks(range(7)); ax[1,0].set_xticklabels(cols, rotation=90, fontsize=7)
ax[1,0].set_yticks(range(7)); ax[1,0].set_yticklabels(cols, fontsize=7); ax[1,0].set_title("Correlation heatmap")
ax[1,1].plot(hist); ax[1,1].set_title("Gradient descent cost"); ax[1,1].set_xlabel("epoch")
ax[1,2].scatter(yte, pred, s=3, alpha=.4); ax[1,2].plot([0, 500], [0, 500], "r--"); ax[1,2].set_title("Actual vs Predicted (test)")
plt.tight_layout(); plt.savefig("aqi_plots_all_cities.png", dpi=100)
