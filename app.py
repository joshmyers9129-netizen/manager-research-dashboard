"""
Manager Research Dashboard
═════════════════════════════════════
Production-quality factor regime analysis tool for manager evaluation.
Single-file Streamlit app. Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import re, io, os, tempfile
from typing import Tuple, List, Dict, Any
from scipy import stats as sp_stats

try:
    import statsmodels.api as sm
    HAS_SM = True
except ImportError:
    HAS_SM = False
try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
try:
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:
    HAS_FPDF = False


# ═══════════════════════════════════════════════════════════════════════════════
#  BRAND + FACTOR CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

C = {  # Brand color palette
    "navy_dk": "#000000", "navy": "#1A1A1A", "navy_md": "#2293BD",
    "blue": "#2293BD", "blue_lt": "#4AB0D0",
    "cream": "#F0E6DD", "cream_lt": "#FBF7F3", "wh": "#FFFFFF",
    "txt": "#222222", "txt_lt": "#A09080",
    "pos": "#2293BD", "neg": "#D9532B", "warn": "#FAA51A",
    "neut": "#7B7265", "grid": "#E8E0D8", "card": "#FBF7F3",
    "red": "#D9532B", "gold": "#FAA51A",
}
BC4 = [C["neg"], C["warn"], C["blue_lt"], C["blue"]]
BC3 = [C["neg"], C["warn"], C["blue"]]
FC = [C["blue"], C["red"], C["gold"], C["navy_dk"], C["cream"],
      C["blue_lt"], C["neut"], "#6B4C8A"]

FDISP = {"style": "Growth vs Value", "yield": "Yield", "momentum": "Momentum",
         "quality": "Quality", "volatility": "Volatility", "liquidity": "Liquidity",
         "size": "Size", "cash": "Cash", "growth": "Growth", "value": "Value"}
FSIGN = {
    "style": "Positive = Growth leading Value. Negative = Value leading Growth.",
    "yield": "Positive = high-dividend stocks outperforming.",
    "momentum": "Positive = recent winners outperforming losers.",
    "quality": "Positive = high-quality outperforming low-quality.",
    "volatility": "Positive = high-vol outperforming low-vol.",
    "liquidity": "Positive = illiquid outperforming liquid.",
    "size": "Positive = small caps outperforming large caps.",
    "cash": "Positive = risk-free rate rising.",
}
METHODOLOGY = (
    "Regime analysis: each month, factor returns are sorted into quartiles (Q1=bottom=out-of-favor, "
    "Q4=top=in-favor). Strategy excess returns are averaged within each bucket. Spread = Q4 avg minus Q1 avg. "
    "Regressions: OLS of monthly excess returns on factor returns. Standardized beta = raw beta x (factor SD / excess SD). "
    "Impact = raw beta x factor SD (expected excess return per 1-SD factor shock). "
    "Significance tested at 5% level. HAC (Newey-West) errors available for autocorrelation adjustment."
)

def fl(raw):
    """Factor display label."""
    return FDISP.get(raw.lower().strip(), raw.title())


def _abbrev_strategy(full_name):
    """Turn 'Acme Capital Management | US Large-Cap Core Growth Equity' into 'Acme LC Core Grw'."""
    if "|" not in full_name:
        return full_name[:30]
    firm, product = full_name.split("|", 1)
    firm, product = firm.strip(), product.strip()
    # First word of firm (drop common suffixes)
    firm_first = firm.split()[0] if firm else ""
    for sfx in ["Capital", "Asset", "Investment", "Management", "Financial", "Global", "Advisors", "Partners"]:
        if firm_first.lower() == sfx.lower() and len(firm.split()) > 1:
            firm_first = firm.split()[0]
            break
    # Shorten common product-name words
    _abbr = {
        "United States": "US", "Large-Cap": "LC", "Large Cap": "LC", "LargeCap": "LC",
        "Small-Cap": "SC", "Small Cap": "SC", "SmallCap": "SC",
        "Mid-Cap": "MC", "Mid Cap": "MC", "MidCap": "MC",
        "International": "Intl", "Emerging": "Emrg", "Markets": "Mkts", "Market": "Mkt",
        "Growth": "Grw", "Value": "Val", "Equity": "", "Equities": "",
        "Fixed Income": "FI", "High Yield": "HY",
        "Corporate": "Corp", "Government": "Govt", "Municipal": "Muni",
        "Aggregate": "Agg", "Treasury": "Treas", "Investment": "Inv",
        "Management": "Mgmt", "Opportunities": "Opp", "Opportunity": "Opp",
        "Technology": "Tech", "Healthcare": "Hlth", "Consumer": "Cnsm",
        "Concentrated": "Conc", "Sustainable": "Sust", "Fundamental": "Fndm",
        "Enhanced": "Enh", "Strategic": "Strat", "Systematic": "Sys",
        "Quantitative": "Quant", "Dividend": "Div", "Income": "Inc",
        "Portfolio": "", "Fund": "", "Strategy": "",
    }
    short = product
    for long, abbr in _abbr.items():
        short = re.sub(r'\b' + re.escape(long) + r'\b', abbr, short, flags=re.IGNORECASE)
    short = re.sub(r'\s+', ' ', short).strip().strip("-").strip()
    return f"{firm_first} {short}".strip()


# ═══════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def _norm_cols(df):
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
    return df

def _detect_units(s):
    c = s.dropna()
    if len(c) == 0: return "unknown", s
    if c.abs().max() > 1.0 or (c.abs().quantile(0.99) if len(c)>10 else c.abs().max()) > 0.5 or c.abs().median() > 0.08:
        return "percent", s / 100.0
    return "decimal", s

def _monthly(df, col="date"):
    df = df.copy(); df[col] = pd.to_datetime(df[col])
    df[col] = df[col].dt.to_period("M").dt.to_timestamp("M")
    return df.sort_values(col).reset_index(drop=True)

def _align(sdf, fdf):
    s, f = set(sdf["date"].dropna()), set(fdf["date"].dropna())
    common = s & f
    if not common:
        raise ValueError(f"No overlap. Strategy: {min(s):%Y-%m}–{max(s):%Y-%m}, Factors: {min(f):%Y-%m}–{max(f):%Y-%m}")
    return (sdf[sdf["date"].isin(common)].sort_values("date").reset_index(drop=True),
            fdf[fdf["date"].isin(common)].sort_values("date").reset_index(drop=True),
            {"common": len(common), "s_only": len(s-f), "f_only": len(f-s),
             "pct": round(len(common)/max(len(s),1)*100, 1)})

_EQ = ["russell","s&p 500","msci","nasdaq","reit","equity","ftse emerging","ftse nareit","spliced equity"]
_FI = ["bloomberg","barclays","ice bofaml","ice bofa","aggregate","high yield","municipal","treasury",
       "govt/credit","leveraged loan","morningstar lsta","cs leveraged","corporate","jp embi","jpm","wgbi","tips"]

def _classify(b):
    if pd.isna(b) or not b: return "Unknown"
    bl = b.lower()
    if any(k in bl for k in ["target","freedom","gtaa"]): return "Multi-Asset"
    if any(k in bl for k in ["t-bill","t bill","treasuries bellwether"]): return "Cash"
    if any(k in bl for k in _FI): return "Fixed Income"
    if any(k in bl for k in _EQ): return "Equity"
    if "cpi" in bl: return "Real Assets"
    return "Other"

def _window_filter(df, window, date_col="date"):
    """Apply global analysis window filter."""
    if window == "Full Period" or df.empty:
        return df
    latest = df[date_col].max()
    if "10" in window:
        cutoff = latest - pd.DateOffset(years=10)
    else:
        cutoff = latest - pd.DateOffset(years=5)
    return df[df[date_col] >= cutoff].reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PARSERS
# ═══════════════════════════════════════════════════════════════════════════════

def parse_evestment(fp):
    xls = pd.ExcelFile(fp)
    sheet = [s for s in xls.sheet_names if s.lower()!="disclaimer"]
    sheet = sheet[0] if sheet else 0
    raw = pd.read_excel(fp, sheet_name=sheet, header=None)
    hi = None
    for i in range(min(30,len(raw))):
        if "Firm Name" in [str(v).strip() for v in raw.iloc[i] if pd.notna(v)]:
            hi=i; break
    if hi is None: raise ValueError("No 'Firm Name' header found.")
    df = pd.read_excel(fp, sheet_name=sheet, header=hi); df.columns = df.columns.str.strip()
    dp = re.compile(r"\((\d{1,2}/\d{4})\s*-")
    cd = {}
    for c in df.columns:
        if "Excess Return" in str(c):
            m = dp.search(str(c))
            if m:
                try: cd[c] = pd.to_datetime(m.group(1), format="%m/%Y").to_period("M").to_timestamp("M")
                except: pass
    if not cd: raise ValueError("No Excess Return columns with dates.")
    recs = []
    for _, row in df.iterrows():
        fm, pr, bm = row.get("Firm Name",""), row.get("Product Name",""), row.get("Benchmark","")
        if pd.isna(fm) or pd.isna(pr): continue
        fm, pr = str(fm).strip(), str(pr).strip()
        bm = str(bm).strip() if pd.notna(bm) else ""
        sn = f"{fm} | {pr}"
        for c, dt in cd.items():
            v = row.get(c)
            if pd.notna(v):
                try: recs.append({"date":dt,"strategy":sn,"excess_return":float(v),"firm_name":fm,"product_name":pr,"benchmark":bm})
                except: pass
    if not recs: raise ValueError("No data extracted.")
    r = pd.DataFrame(recs); r["date"]=pd.to_datetime(r["date"]); return r

def parse_risk_premiums(fp, sn="Multiple Risk Premiums"):
    try: raw = pd.read_excel(fp, sheet_name=sn, header=None)
    except: xls = pd.ExcelFile(fp); raw = pd.read_excel(fp, sheet_name=xls.sheet_names[0], header=None)
    known = {"quality","momentum","style","yield","volatility","liquidity","size","cash","growth","value","low_vol","beta"}
    hi = None
    for i in range(min(30,len(raw))):
        vals = {str(v).strip().lower().replace(" ","_") for v in raw.iloc[i] if pd.notna(v)}
        if len(vals & known)>=2: hi=i; break
    if hi is None: raise ValueError("No factor header found.")
    df = pd.read_excel(fp, sheet_name=sn, header=hi)
    df.columns = df.columns.str.strip().str.lower().str.replace(r"\s+","_",regex=True)
    df = df.rename(columns={df.columns[0]:"date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce"); df = df.dropna(subset=["date"])
    fc = []
    for c in df.columns:
        if c=="date": continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[c].notna().sum()>0: fc.append(c)
    df = df[["date"]+fc]; df["date"] = df["date"].dt.to_period("M").dt.to_timestamp("M")
    return df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

def parse_clean_strategy(fp):
    fn = str(getattr(fp,"name",fp)).lower()
    df = pd.read_csv(fp) if fn.endswith(".csv") else pd.read_excel(fp)
    df = _norm_cols(df)
    if "date" not in df.columns: raise ValueError(f"Missing 'date'.")
    if "strategy" not in df.columns: raise ValueError(f"Missing 'strategy'.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce"); df = _monthly(df)
    if "excess_return" not in df.columns:
        if "strategy_return" in df.columns and "benchmark_return" in df.columns:
            df["excess_return"] = df["strategy_return"] - df["benchmark_return"]
        elif "strategy_return" in df.columns: df["excess_return"] = df["strategy_return"]
        else: raise ValueError("Need excess_return or strategy_return+benchmark_return.")
    for c in ["excess_return","strategy_return","benchmark_return"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["firm_name","product_name","benchmark"]:
        if c not in df.columns: df[c] = ""
    return df

def parse_clean_factors(fp):
    fn = str(getattr(fp,"name",fp)).lower()
    df = pd.read_csv(fp) if fn.endswith(".csv") else pd.read_excel(fp)
    df = _norm_cols(df)
    if "date" not in df.columns: raise ValueError("Missing 'date'.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce"); df = _monthly(df)
    for c in df.columns:
        if c!="date": df[c] = pd.to_numeric(df[c], errors="coerce")
    fc = [c for c in df.columns if c!="date" and df[c].notna().sum()>0]
    if not fc: raise ValueError("No numeric factor columns.")
    return df[["date"]+fc]

# ═══════════════════════════════════════════════════════════════════════════════
#  ANALYTICS ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def _blabels(n):
    if n==3: return ["T1 (Bottom)","T2 (Mid)","T3 (Top)"]
    if n==4: return ["Q1 (Bottom)","Q2","Q3","Q4 (Top)"]
    return [f"B{i+1}" for i in range(n)]

def _favor(b):
    if "Bottom" in b: return "Out-of-Favor"
    if "Top" in b: return "In-Favor"
    return "Neutral"

def regime_one(excess, fseries, fname, nb=4):
    tmp = pd.DataFrame({"e":excess.values,"f":fseries.values}).dropna()
    if len(tmp)<nb*2: return pd.DataFrame()
    labels = _blabels(nb)
    tmp["bucket"] = pd.qcut(tmp["f"], q=nb, labels=labels, duplicates="drop")
    agg = tmp.groupby("bucket", observed=False)["e"].agg(
        avg_excess="mean", median_excess="median", std_excess="std",
        hit_rate=lambda x:(x>0).mean()*100, count="count").reset_index()
    agg["factor"]=fname; agg["environment"]=agg["bucket"].apply(_favor)
    agg["se"]=agg["std_excess"]/np.sqrt(agg["count"])
    return agg[["factor","bucket","environment","avg_excess","median_excess","std_excess","se","hit_rate","count"]]

def all_regimes(excess, fdf, nb=4):
    parts = [regime_one(excess, fdf[c], c, nb) for c in fdf.columns if c!="date"]
    parts = [p for p in parts if len(p)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

def spread_summary(rt, nb=4):
    if rt.empty: return pd.DataFrame()
    labels = _blabels(nb)
    top = rt[rt["bucket"]==labels[-1]].set_index("factor")
    bot = rt[rt["bucket"]==labels[0]].set_index("factor")
    sp = (top["avg_excess"]-bot["avg_excess"]).rename("spread").reset_index()
    ts, ps = [], []
    for _, row in sp.iterrows():
        tr, br = top.loc[row["factor"]], bot.loc[row["factor"]]
        se = np.sqrt(tr["se"]**2+br["se"]**2)
        t = row["spread"]/se if se>0 else 0
        dfa = tr["count"]+br["count"]-2
        p = 2*(1-sp_stats.t.cdf(abs(t), df=max(dfa,1)))
        ts.append(t); ps.append(p)
    sp["t_stat"]=ts; sp["p_value"]=ps; sp["significant"]=sp["p_value"]<0.05
    sp["abs_spread"]=sp["spread"].abs()
    sp = sp.sort_values("abs_spread", ascending=False).drop(columns="abs_spread").reset_index(drop=True)
    def _interp(r):
        f = fl(r["factor"])
        if r["factor"]=="style":
            return "Outperforms in Value-led markets" if r["spread"]<0 else "Outperforms in Growth-led markets"
        return f"Outperforms when {f} IN-FAVOR" if r["spread"]>0 else f"Outperforms when {f} OUT-OF-FAVOR"
    sp["interpretation"] = sp.apply(_interp, axis=1)
    return sp


# ── Notable quarters & years ─────────────────────────────────────────────────

def _drivers(grp, full, fcols, z_thresh=1.0):
    out = []
    for fc in fcols:
        mu, sig = full[fc].mean(), full[fc].std()
        if sig<=0: continue
        z = (grp[fc].mean()-mu)/sig
        if fc=="style":
            if z>=z_thresh: out.append(f"Growth over Value (z={z:+.1f})")
            elif z<=-z_thresh: out.append(f"Value over Growth (z={z:+.1f})")
        else:
            f_ = fl(fc)
            if z>=z_thresh: out.append(f"{f_} in-favor (z={z:+.1f})")
            elif z<=-z_thresh: out.append(f"{f_} out-of-favor (z={z:+.1f})")
    return "; ".join(out) if out else "No extreme factor regimes"

def notable_quarters(dates, excess, fdf, n=5):
    df = pd.DataFrame({"date":dates.values,"excess":excess.values})
    fc = [c for c in fdf.columns if c!="date"]
    for c in fc: df[c]=fdf[c].values
    df = df.dropna(subset=["excess"]).sort_values("date").reset_index(drop=True)
    df["qtr"]=df["date"].dt.to_period("Q")
    recs = []
    for q, g in df.groupby("qtr"):
        cum=(1+g["excess"]).prod()-1
        recs.append({"period":str(q),"start":g["date"].iloc[0],"end":g["date"].iloc[-1],
                      "excess_return":cum,"months":len(g),"factor_drivers":_drivers(g,df,fc)})
    qdf=pd.DataFrame(recs)
    best=qdf.nlargest(n,"excess_return").copy(); best["label"]="Best"
    worst=qdf.nsmallest(n,"excess_return").copy(); worst["label"]="Worst"
    return pd.concat([best,worst]).sort_values("start").reset_index(drop=True)

def notable_years(dates, excess, fdf, n=3):
    df = pd.DataFrame({"date":dates.values,"excess":excess.values})
    fc = [c for c in fdf.columns if c!="date"]
    for c in fc: df[c]=fdf[c].values
    df = df.dropna(subset=["excess"]).sort_values("date").reset_index(drop=True)
    df["year"]=df["date"].dt.year
    recs = []
    for yr, g in df.groupby("year"):
        if len(g)<6: continue
        cum=(1+g["excess"]).prod()-1
        recs.append({"period":str(yr),"excess_return":cum,"months":len(g),
                      "factor_drivers":_drivers(g,df,fc,z_thresh=0.8)})
    ydf=pd.DataFrame(recs)
    nt=min(n, max(1, len(ydf)//2))
    best=ydf.nlargest(nt,"excess_return").copy(); best["label"]="Best"
    worst=ydf.nsmallest(nt,"excess_return").copy(); worst["label"]="Worst"
    return pd.concat([best,worst]).sort_values("period").reset_index(drop=True)


# ── Regressions ───────────────────────────────────────────────────────────────

def _ols(y, x, hac=None):
    mask=~(np.isnan(y)|np.isnan(x)); y,x=y[mask],x[mask]; n=len(y)
    if n<10: return {"alpha":np.nan,"beta":np.nan,"t":np.nan,"p":np.nan,"r2":np.nan,"n":n}
    if HAS_SM:
        X=sm.add_constant(x)
        m=sm.OLS(y,X).fit(cov_type="HAC",cov_kwds={"maxlags":hac}) if hac else sm.OLS(y,X).fit()
        return {"alpha":m.params[0],"beta":m.params[1],"t":m.tvalues[1],"p":m.pvalues[1],"r2":m.rsquared,"n":int(m.nobs)}
    sl,ic,r,p,se=sp_stats.linregress(x,y)
    return {"alpha":ic,"beta":sl,"t":sl/se if se else np.nan,"p":p,"r2":r**2,"n":n}

def run_sf(excess, fdf, hac=None):
    rows=[]
    for c in [col for col in fdf.columns if col!="date"]:
        r=_ols(excess.values, fdf[c].values, hac); r["factor"]=c
        sx, sy = fdf[c].std(), excess.std()
        r["std_beta"]=r["beta"]*sx/sy if sy>0 else np.nan
        r["impact"]=r["beta"]*sx
        rows.append(r)
    return pd.DataFrame(rows)[["factor","alpha","beta","std_beta","impact","t","p","r2","n"]]

def run_mf(excess, fdf, fcols=None, hac=None):
    if not HAS_SM: return pd.DataFrame()
    if fcols is None: fcols=[c for c in fdf.columns if c!="date"]
    tmp=pd.DataFrame({"y":excess.values})
    for fc in fcols: tmp[fc]=fdf[fc].values
    tmp=tmp.dropna()
    if len(tmp)<len(fcols)+5: return pd.DataFrame()
    X=sm.add_constant(tmp[fcols])
    m=sm.OLS(tmp["y"],X).fit(cov_type="HAC",cov_kwds={"maxlags":hac}) if hac else sm.OLS(tmp["y"],X).fit()
    rows=[{"var":"alpha","coef":m.params.get("const",np.nan),"t":m.tvalues.get("const",np.nan),"p":m.pvalues.get("const",np.nan)}]
    for fc in fcols:
        rows.append({"var":fc,"coef":m.params.get(fc,np.nan),"t":m.tvalues.get(fc,np.nan),"p":m.pvalues.get(fc,np.nan)})
    result=pd.DataFrame(rows)
    result.attrs.update(r2=m.rsquared, adj_r2=m.rsquared_adj, n=int(m.nobs), f=m.fvalue, fp=m.f_pvalue)
    return result

def rolling_betas(dates, excess, fdf, window=36):
    parts=[]
    for fc in [c for c in fdf.columns if c!="date"]:
        tmp=pd.DataFrame({"date":dates.values,"y":excess.values,"x":fdf[fc].values}).dropna().sort_values("date").reset_index(drop=True)
        if len(tmp)<window: continue
        recs=[]
        for i in range(window, len(tmp)+1):
            ch=tmp.iloc[i-window:i]; r=_ols(ch["y"].values,ch["x"].values)
            recs.append({"date":ch["date"].iloc[-1],"factor":fc,"beta":r["beta"],"t":r["t"],"r2":r["r2"]})
        parts.append(pd.DataFrame(recs))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

def subperiod_betas(excess, fdf, dates):
    pds=[("Pre-2020",dates<"2020-01-01"),("2020-2021",(dates>="2020-01-01")&(dates<"2022-01-01")),("2022+",dates>="2022-01-01")]
    fc=[c for c in fdf.columns if c!="date"]; recs=[]
    for lbl,mask in pds:
        if mask.sum()<12: continue
        for f in fc:
            r=_ols(excess[mask].values, fdf[f][mask].values)
            recs.append({"period":lbl,"factor":f,"beta":r["beta"],"t":r["t"],"p":r["p"],"n_obs":int(mask.sum())})
    return pd.DataFrame(recs)


# ── Peer comparison compute ───────────────────────────────────────────────────

def compute_peer_table(strategy_df, factor_df, benchmark, window_label, hac=None):
    """Compute multi-factor betas + R-squared for all strategies sharing a benchmark."""
    peers = strategy_df[strategy_df["benchmark"]==benchmark]["strategy"].unique()
    rows = []
    for strat in peers:
        sd = strategy_df[strategy_df["strategy"]==strat][["date","excess_return"]].dropna()
        sd = _window_filter(sd, window_label)
        try:
            sa, fa, info = _align(sd, factor_df)
        except ValueError:
            continue
        if info["common"] < 24: continue
        ex = sa["excess_return"].reset_index(drop=True)
        fac = fa.reset_index(drop=True)
        sf = run_sf(ex, fac, hac=hac)
        mf = run_mf(ex, fac, hac=hac)
        row = {"strategy": strat, "months": info["common"]}
        if not mf.empty:
            row["r2"] = mf.attrs.get("r2", np.nan)
            row["adj_r2"] = mf.attrs.get("adj_r2", np.nan)
            alpha_row = mf[mf["var"]=="alpha"]
            if len(alpha_row):
                row["alpha_bps"] = alpha_row.iloc[0]["coef"]*10000
                row["alpha_p"] = alpha_row.iloc[0]["p"]
            for _, fr in mf[mf["var"]!="alpha"].iterrows():
                row[f"mf_{fr['var']}"] = fr["coef"]
                row[f"mf_{fr['var']}_p"] = fr["p"]
        else:
            # Fallback: use sum of single-factor R² capped at 1 as rough proxy
            top_r2 = sf.sort_values("r2", ascending=False).iloc[0]["r2"] if len(sf) else np.nan
            row["r2"] = top_r2
            row["adj_r2"] = np.nan
        for _, sr in sf.iterrows():
            row[f"std_{sr['factor']}"] = sr["std_beta"]
            row[f"imp_{sr['factor']}"] = sr["impact"]*10000
            if mf.empty:
                row[f"mf_{sr['factor']}"] = sr["beta"]
                row[f"mf_{sr['factor']}_p"] = sr["p"]
        rows.append(row)
    return pd.DataFrame(rows)


def compute_peer_corr(strategy_df, benchmark, window_label):
    """Build a correlation matrix of monthly excess returns across peer strategies."""
    peers = strategy_df[strategy_df["benchmark"]==benchmark]["strategy"].unique()
    series = {}
    for strat in peers:
        sd = strategy_df[strategy_df["strategy"]==strat][["date","excess_return"]].dropna()
        sd = _window_filter(sd, window_label)
        if len(sd) < 24: continue
        sd = sd.drop_duplicates(subset="date").set_index("date")["excess_return"]
        series[strat] = sd
    if len(series) < 2: return pd.DataFrame()
    wide = pd.DataFrame(series)
    return wide.corr()


def compute_blend_stats(strategy_df, s1, s2, window_label):
    """Compute 50/50 blended historical performance stats for a strategy pair."""
    d1 = strategy_df[strategy_df["strategy"]==s1][["date","excess_return"]].dropna()
    d2 = strategy_df[strategy_df["strategy"]==s2][["date","excess_return"]].dropna()
    d1 = _window_filter(d1, window_label).drop_duplicates(subset="date").set_index("date")
    d2 = _window_filter(d2, window_label).drop_duplicates(subset="date").set_index("date")
    merged = d1.join(d2, lsuffix="_1", rsuffix="_2", how="inner").dropna()
    if len(merged) < 12: return None
    e1, e2 = merged["excess_return_1"], merged["excess_return_2"]
    blend = 0.5 * e1 + 0.5 * e2
    # Annualised stats
    ann_ret_1 = (1 + e1).prod() ** (12 / len(e1)) - 1
    ann_ret_2 = (1 + e2).prod() ** (12 / len(e2)) - 1
    ann_ret_b = (1 + blend).prod() ** (12 / len(blend)) - 1
    ann_vol_1 = e1.std() * np.sqrt(12)
    ann_vol_2 = e2.std() * np.sqrt(12)
    ann_vol_b = blend.std() * np.sqrt(12)
    # Max drawdown (cumulative excess)
    def _mdd(s):
        cum = (1 + s).cumprod()
        peak = cum.cummax()
        dd = (cum - peak) / peak
        return dd.min()
    mdd_1, mdd_2, mdd_b = _mdd(e1), _mdd(e2), _mdd(blend)
    # Hit rate
    hit_1 = (e1 > 0).mean() * 100
    hit_2 = (e2 > 0).mean() * 100
    hit_b = (blend > 0).mean() * 100
    return {
        "months": len(merged),
        "ann_ret": (ann_ret_1, ann_ret_2, ann_ret_b),
        "ann_vol": (ann_vol_1, ann_vol_2, ann_vol_b),
        "max_dd": (mdd_1, mdd_2, mdd_b),
        "hit_rate": (hit_1, hit_2, hit_b),
    }


def _tilt_label(factor_name, beta):
    """Generate a plain-English tilt label for a factor exposure."""
    f_raw = factor_name.lower()
    if "style" in f_raw or "growth" in f_raw and "value" in f_raw:
        return "Growth-tilted" if beta > 0 else "Value-tilted"
    if "momentum" in f_raw:
        return "Momentum-tilted" if beta > 0 else "Contrarian-tilted"
    if "quality" in f_raw:
        return "Quality-tilted" if beta > 0 else "Low-quality-tilted"
    if "volatility" in f_raw or "vol" in f_raw:
        return "High-vol-tilted" if beta > 0 else "Low-vol-tilted"
    if "size" in f_raw:
        return "Small-cap-tilted" if beta > 0 else "Large-cap-tilted"
    if "yield" in f_raw or "dividend" in f_raw:
        return "High-yield-tilted" if beta > 0 else "Low-yield-tilted"
    if "liquidity" in f_raw:
        return "Illiquidity-tilted" if beta > 0 else "Liquidity-tilted"
    if "cash" in f_raw:
        return "Risk-off-tilted" if beta > 0 else "Risk-on-tilted"
    return f"Positive {factor_name}" if beta > 0 else f"Negative {factor_name}"


def find_pairings(corr_df, peer_df, factor_cols, n=5):
    """Find strategy pairs with low/negative excess correlation AND different factor profiles."""
    if corr_df.empty or len(corr_df) < 2: return []
    strats = list(corr_df.columns)
    pairs = []
    for i in range(len(strats)):
        for j in range(i+1, len(strats)):
            s1, s2 = strats[i], strats[j]
            rho = corr_df.loc[s1, s2]
            if np.isnan(rho): continue
            # Compute factor profile distance
            r1 = peer_df[peer_df["strategy"]==s1]
            r2 = peer_df[peer_df["strategy"]==s2]
            if r1.empty or r2.empty: continue
            r1, r2 = r1.iloc[0], r2.iloc[0]
            diffs = []
            diff_details = []
            for fc in factor_cols:
                mf_col = f"mf_{fc}"
                v1 = r1.get(mf_col, np.nan)
                v2 = r2.get(mf_col, np.nan)
                if pd.notna(v1) and pd.notna(v2):
                    diffs.append((v1 - v2)**2)
                    if abs(v1 - v2) > 0.01:
                        diff_details.append({"factor": fl(fc), "raw_factor": fc,
                                             "s1_beta": v1, "s2_beta": v2})
            factor_dist = np.sqrt(sum(diffs)) if diffs else 0
            # Score: lower correlation + higher factor distance = better pair
            score = (1 - rho) * 0.6 + min(factor_dist * 5, 1) * 0.4
            # Find opposing factor exposures for narrative
            opposing = [d for d in diff_details if d["s1_beta"] * d["s2_beta"] < 0]
            same_sign = [d for d in diff_details if d["s1_beta"] * d["s2_beta"] > 0
                         and abs(d["s1_beta"] - d["s2_beta"]) > 0.02]
            pairs.append({
                "s1": s1, "s2": s2, "correlation": rho, "factor_distance": factor_dist,
                "score": score, "opposing": opposing, "different_magnitude": same_sign
            })
    pairs.sort(key=lambda x: x["score"], reverse=True)
    return pairs[:n]


def get_recommended_partners(anchor, corr_df, peer_df, factor_cols, n=3):
    """For a given anchor strategy, find the top N complementary partners using the same
    scoring logic as find_pairings(): 60% weight on low correlation, 40% on factor distance."""
    if corr_df.empty or anchor not in corr_df.columns: return []
    r1 = peer_df[peer_df["strategy"] == anchor]
    if r1.empty: return []
    r1 = r1.iloc[0]
    candidates = []
    for s2 in corr_df.columns:
        if s2 == anchor: continue
        rho = corr_df.loc[anchor, s2] if anchor in corr_df.index else corr_df.loc[s2, anchor]
        if np.isnan(rho): continue
        r2 = peer_df[peer_df["strategy"] == s2]
        if r2.empty: continue
        r2 = r2.iloc[0]
        diffs = []
        for fc in factor_cols:
            mf_col = f"mf_{fc}"
            v1 = r1.get(mf_col, np.nan)
            v2 = r2.get(mf_col, np.nan)
            if pd.notna(v1) and pd.notna(v2):
                diffs.append((v1 - v2) ** 2)
        factor_dist = np.sqrt(sum(diffs)) if diffs else 0
        score = (1 - rho) * 0.6 + min(factor_dist * 5, 1) * 0.4
        candidates.append({"strategy": s2, "score": score})
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return [c["strategy"] for c in candidates[:n]]


# ═══════════════════════════════════════════════════════════════════════════════
#  SUMMARY GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def make_summary(sf, rt, sp, mf, n_months, nq, ny, window_label):
    """Return structured summary dict for the card-based UI."""
    out = {"kpis": [], "factors": [], "narrative": "", "outlook": "", "outlook_watch": "",
           "best_q": None, "worst_q": None, "window": window_label, "n_months": n_months}
    sig = sf[sf["p"]<0.05].sort_values("r2", ascending=False)

    # ── Build regime lookup early (used by KPIs and factor cards) ────────────
    regime_lookup = {}
    if not rt.empty:
        labels_4 = _blabels(4)
        for fname_r in rt["factor"].unique():
            sub = rt[rt["factor"]==fname_r]
            top_row = sub[sub["bucket"]==labels_4[-1]]
            bot_row = sub[sub["bucket"]==labels_4[0]]
            if not top_row.empty and not bot_row.empty:
                regime_lookup[fname_r] = {
                    "top_bps": top_row.iloc[0]["avg_excess"] * 10000,
                    "bot_bps": bot_row.iloc[0]["avg_excess"] * 10000,
                    "hit_top": top_row.iloc[0]["hit_rate"],
                    "hit_bot": bot_row.iloc[0]["hit_rate"],
                }

    # ── KPI: R-squared ───────────────────────────────────────────────────────
    r2_val = 0
    r2_src = ""
    if not mf.empty:
        r2_val = mf.attrs.get("r2", 0) * 100
        r2_src = "multi-factor"
    elif len(sig) > 0:
        r2_val = sig.iloc[0]["r2"] * 100
        r2_src = "single-factor"
    out["kpis"].append({"label": "Factor Explained", "value": f"{r2_val:.0f}%",
                        "sub": f"{100-r2_val:.0f}% driven by stock selection & timing",
                        "color": "teal" if r2_val >= 20 else "neut"})

    # ── KPI: Alpha ───────────────────────────────────────────────────────────
    if not mf.empty:
        ar = mf[mf["var"]=="alpha"]
        if len(ar):
            abps = ar.iloc[0]["coef"] * 10000
            ap = ar.iloc[0]["p"]
            sig_tag = "Significant" if ap < 0.05 else "Not significant"
            out["kpis"].append({"label": "Monthly Alpha", "value": f"{abps:+.0f} bps",
                                "sub": sig_tag, "color": "pos" if abps > 0 else "neg"})

    # ── KPI: Strongest relationship ──────────────────────────────────────────
    if len(sig) > 0:
        t = sig.iloc[0]
        fname_k = t["factor"]
        rl_k = regime_lookup.get(fname_k, {})
        if rl_k:
            spread_k = abs(rl_k.get("top_bps", 0) - rl_k.get("bot_bps", 0))
            sub_k = f"{spread_k:.0f} bps spread between regimes"
        else:
            sub_k = f"{abs(t['impact'] * 10000):.0f} bps impact per 1-SD move"
        out["kpis"].append({"label": "Strongest Link", "value": fl(fname_k),
                            "sub": sub_k, "color": "teal"})

    # ── KPI: Total significant factors ───────────────────────────────────────
    n_sig = len(sig)
    n_total = len(sf)
    out["kpis"].append({"label": "Significant Factors", "value": f"{n_sig} of {n_total}",
                        "sub": "at 95% confidence level",
                        "color": "teal" if n_sig > 0 else "neut"})

    # ── Factor cards ─────────────────────────────────────────────────────────
    for _, row in sf.sort_values("p").iterrows():
        is_sig = row["p"] < 0.05
        imp_bps = row["impact"] * 10000
        r2_pct = row["r2"] * 100
        fname = row["factor"]

        if fname == "style":
            if row["beta"] > 0:
                direction = "Favors Growth-led markets"
            else:
                direction = "Favors Value-led markets"
        elif row["beta"] > 0:
            direction = f"Benefits when {fl(fname)} is rewarded"
        else:
            direction = f"Benefits when {fl(fname)} is out of favor"

        # Regime quartile averages (concrete, intuitive)
        rl = regime_lookup.get(fname, {})
        top_bps = rl.get("top_bps", None)
        bot_bps = rl.get("bot_bps", None)
        hit_top = rl.get("hit_top", None)
        hit_bot = rl.get("hit_bot", None)

        # Strength label based on spread
        if top_bps is not None and bot_bps is not None:
            spread = abs(top_bps - bot_bps)
            if spread >= 60: strength = "Strong"
            elif spread >= 25: strength = "Moderate"
            else: strength = "Weak"
        else:
            abs_imp = abs(imp_bps)
            if abs_imp >= 30: strength = "Strong"
            elif abs_imp >= 15: strength = "Moderate"
            else: strength = "Weak"

        out["factors"].append({
            "name": fl(fname), "raw": fname, "significant": is_sig,
            "impact_bps": imp_bps, "r2_pct": r2_pct, "p": row["p"],
            "direction": direction, "strength": strength, "beta": row["beta"],
            "top_bps": top_bps, "bot_bps": bot_bps,
            "hit_top": hit_top, "hit_bot": hit_bot,
        })

    # ── Narrative ────────────────────────────────────────────────────────────
    if len(sig) == 0:
        out["narrative"] = ("This strategy's excess returns show no statistically significant "
                           "relationship to the factors analyzed. Performance appears driven by "
                           "stock selection, sector positioning, or exposures outside this factor set.")
    else:
        t = sig.iloc[0]; f_ = fl(t["factor"])
        if t["factor"] == "style":
            env = "Growth-led environments" if t["beta"] > 0 else "Value-led environments"
        elif t["beta"] > 0:
            env = f"periods when {f_} is being rewarded by the market"
        else:
            env = f"periods when {f_} is out of favor"
        parts = [f"The dominant driver of this strategy's excess returns is {f_}, "
                f"which explains {t['r2']*100:.0f}% of the variation. "
                f"The strategy tends to outperform in {env}."]
        if len(sig) > 1:
            others = [fl(r["factor"]) for _, r in sig.iloc[1:].iterrows()]
            parts.append(f" Secondary relationships exist with {', '.join(others)}.")
        out["narrative"] = "".join(parts)

    # ── Outlook ──────────────────────────────────────────────────────────────
    if len(sig) > 0:
        t = sig.iloc[0]; f_ = fl(t["factor"])
        if t["factor"] == "style":
            out["outlook"] = "Expect strength in Value rotations and relative weakness during Growth-led rallies." if t["beta"] < 0 else "Expect strength during Growth-led rallies and relative weakness in Value rotations."
        elif t["beta"] < 0:
            out["outlook"] = f"Expect relative strength when {f_} reverses and potential headwinds when {f_} is being rewarded."
        else:
            out["outlook"] = f"Expect relative strength when {f_} is being rewarded and potential headwinds when {f_} reverses."
        out["outlook_watch"] = f"Monitor the rolling 36-month {f_} beta for sign changes or weakening significance."

    # ── Best/worst quarters ──────────────────────────────────────────────────
    if not nq.empty:
        bq = nq[nq["label"]=="Best"].nlargest(1, "excess_return")
        wq = nq[nq["label"]=="Worst"].nsmallest(1, "excess_return")
        if len(bq): out["best_q"] = bq.iloc[0].to_dict()
        if len(wq): out["worst_q"] = wq.iloc[0].to_dict()

    return out

# ═══════════════════════════════════════════════════════════════════════════════
#  PLOTLY CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

def _lay(fig, title=""):
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=C["navy"], family="Georgia, Times New Roman, serif")),
        font=dict(family="Arial, Helvetica Neue, sans-serif", size=12, color=C["txt"]),
        plot_bgcolor=C["wh"], paper_bgcolor=C["wh"],
        margin=dict(l=60,r=30,t=55,b=50),
        xaxis=dict(gridcolor=C["grid"], zeroline=True, zerolinecolor="#D5CFC8"),
        yaxis=dict(gridcolor=C["grid"], zeroline=True, zerolinecolor="#D5CFC8"))
    return fig

def ch_heatmap(rt, metric="avg_excess"):
    if not HAS_PLOTLY or rt.empty: return None
    pivot=rt.pivot(index="factor", columns="bucket", values=metric)
    pivot=pivot[sorted(pivot.columns)]
    pivot.index=[fl(f) for f in pivot.index]
    fmt=(lambda x: f"{x:.1f}%") if metric=="hit_rate" else (lambda x: f"{x*10000:.0f}")
    text=np.vectorize(fmt)(pivot.values)
    fig=go.Figure(go.Heatmap(z=pivot.values, x=[str(c) for c in pivot.columns], y=list(pivot.index),
        colorscale=[[0,C["neg"]],[0.5,C["wh"]],[1,C["blue"]]], zmid=0,
        text=text, texttemplate="%{text}", textfont=dict(size=12, color=C["txt"]),
        colorbar=dict(title="Hit %" if metric=="hit_rate" else "Bps/Mo"),
        hovertemplate="Factor: %{y}<br>Bucket: %{x}<br>Value: %{z:.4f}<extra></extra>"))
    _lay(fig, f"Factor Regime Analysis: {metric.replace('_',' ').title()}")
    fig.update_layout(height=max(300, len(pivot)*55+130), yaxis=dict(autorange="reversed"))
    fig.add_annotation(x=0,y=-0.12,xref="paper",yref="paper",text="\u25c0 Out-of-Favor",
                       showarrow=False, font=dict(size=10, color=C["neut"]), xanchor="left")
    fig.add_annotation(x=1,y=-0.12,xref="paper",yref="paper",text="In-Favor \u25b6",
                       showarrow=False, font=dict(size=10, color=C["neut"]), xanchor="right")
    return fig

def ch_bar(rt, fname):
    if not HAS_PLOTLY: return None
    sub=rt[rt["factor"]==fname]
    if sub.empty: return None
    colors=BC4 if len(sub)==4 else BC3
    fig=go.Figure(go.Bar(x=sub["bucket"], y=sub["avg_excess"], marker_color=colors[:len(sub)],
        text=[f"{v*10000:.0f}" for v in sub["avg_excess"]], textposition="outside",
        error_y=dict(type="data", array=sub["se"].values*1.96, visible=True, color=C["neut"], thickness=1.5),
        hovertemplate="Bucket: %{x}<br>Avg: %{y:.4f}<br>Hit: %{customdata[0]:.1f}%<br>N=%{customdata[1]}<extra></extra>",
        customdata=sub[["hit_rate","count"]].values))
    _lay(fig, f"{fl(fname)}: Avg Excess by Regime (bps, 95% CI)")
    fig.update_layout(height=380, yaxis_title="Avg Excess Return")
    return fig

def ch_sf(sf):
    if not HAS_PLOTLY or sf.empty: return None
    df=sf.sort_values("impact", ascending=True)
    colors=[C["blue"] if b>0 else C["neg"] for b in df["impact"]]
    fig=go.Figure(go.Bar(y=[fl(f) for f in df["factor"]], x=df["impact"]*10000, orientation="h",
        marker_color=colors,
        text=[f"{v*10000:+.0f}{'*' if p<.05 else ''}" for v,p in zip(df["impact"],df["p"])],
        textposition="outside",
        hovertemplate="%{y}<br>Impact: %{x:.0f} bps/1-SD<br>Beta: %{customdata[0]:.3f}<br>t=%{customdata[1]:.2f}<br>p=%{customdata[2]:.4f}<extra></extra>",
        customdata=df[["beta","t","p"]].values))
    _lay(fig, "Expected Excess per 1-SD Factor Move (bps) (* = significant)")
    fig.update_layout(height=max(300, len(df)*42+120), xaxis_title="Bps per 1-SD")
    return fig

def ch_mf(mf):
    if not HAS_PLOTLY or mf.empty: return None
    df=mf[mf["var"]!="alpha"].sort_values("coef", ascending=True)
    colors=[C["blue"] if c>0 else C["neg"] for c in df["coef"]]
    fig=go.Figure(go.Bar(y=[fl(f) for f in df["var"]], x=df["coef"], orientation="h",
        marker_color=colors, text=[f"{c:.3f}{'*' if p<.05 else ''}" for c,p in zip(df["coef"],df["p"])],
        textposition="outside"))
    _lay(fig, "Multi-Factor Coefficients (* = significant)")
    fig.update_layout(height=max(300, len(df)*42+120))
    return fig

def ch_rolling(rdf, window=36):
    if not HAS_PLOTLY or rdf.empty: return None
    fig=go.Figure()
    for i, fc in enumerate(rdf["factor"].unique()):
        sub=rdf[rdf["factor"]==fc].sort_values("date")
        fig.add_trace(go.Scatter(x=sub["date"],y=sub["beta"],name=fl(fc),mode="lines",
                                  line=dict(color=FC[i%len(FC)], width=2)))
    fig.add_hline(y=0, line_dash="dash", line_color="#D5CFC8")
    _lay(fig, f"Rolling {window}-Month Factor Betas")
    fig.update_layout(height=420, legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
    return fig

def ch_cum(dates, excess, name):
    if not HAS_PLOTLY: return None
    tmp=pd.DataFrame({"d":dates.values,"e":excess.values}).dropna().sort_values("d")
    tmp["c"]=(1+tmp["e"]).cumprod()-1
    fig=go.Figure(go.Scatter(x=tmp["d"],y=tmp["c"],mode="lines",fill="tozeroy",
        line=dict(color=C["blue"],width=2), fillcolor="rgba(34,147,189,0.08)",
        hovertemplate="Date: %{x}<br>Cumulative: %{y:.3%}<extra></extra>"))
    fig.add_hline(y=0, line_dash="dash", line_color="#D5CFC8")
    _lay(fig, f"Cumulative Excess Return: {name}")
    fig.update_layout(height=320, yaxis_tickformat=".1%")
    return fig

def ch_corr(fdf):
    if not HAS_PLOTLY: return None
    fc=[c for c in fdf.columns if c!="date"]; corr=fdf[fc].corr()
    fig=go.Figure(go.Heatmap(z=corr.values, x=[fl(c) for c in corr.columns], y=[fl(c) for c in corr.index],
        colorscale=[[0,C["neg"]],[0.5,C["wh"]],[1,C["blue"]]], zmid=0, zmin=-1, zmax=1,
        text=np.round(corr.values,2), texttemplate="%{text}", textfont=dict(size=10)))
    _lay(fig, "Factor Correlation Matrix")
    fig.update_layout(height=max(350, len(fc)*45+120), yaxis=dict(autorange="reversed"))
    return fig

def ch_peer_corr(corr_df):
    """Heatmap of pairwise excess return correlations across peer strategies."""
    if not HAS_PLOTLY or corr_df.empty: return None
    labels = [_abbrev_strategy(s) for s in corr_df.columns]
    fig = go.Figure(go.Heatmap(
        z=corr_df.values, x=labels, y=labels,
        colorscale=[[0, C["neg"]], [0.5, C["wh"]], [1, C["blue"]]],
        zmid=0.5, zmin=-0.2, zmax=1,
        text=np.round(corr_df.values, 2), texttemplate="%{text}",
        textfont=dict(size=9),
        colorbar=dict(title="Correlation"),
        hovertemplate="Strategy 1: %{y}<br>Strategy 2: %{x}<br>Correlation: %{z:.3f}<extra></extra>"))
    _lay(fig, "Excess Return Correlations (Pairwise)")
    n = len(corr_df)
    fig.update_layout(
        height=max(400, n * 40 + 150),
        yaxis=dict(autorange="reversed"),
        xaxis_tickangle=-45)
    return fig


def ch_notable_q(nq):
    if not HAS_PLOTLY or nq.empty: return None
    df=nq.sort_values("start")
    colors=[C["blue"] if l=="Best" else C["neg"] for l in df["label"]]
    fig=go.Figure(go.Bar(x=df["period"],y=df["excess_return"],marker_color=colors,
        text=[f"{v:+.1%}" for v in df["excess_return"]], textposition="outside",
        hovertemplate="%{x}<br>Excess: %{y:.2%}<br>%{customdata}<extra></extra>",
        customdata=df["factor_drivers"].values))
    _lay(fig, "Best & Worst Quarters: Excess Return")
    fig.update_layout(height=400, xaxis_tickangle=-45, yaxis_tickformat=".1%")
    return fig

def ch_notable_y(ny):
    if not HAS_PLOTLY or ny.empty: return None
    df=ny.sort_values("period")
    colors=[C["blue"] if l=="Best" else C["neg"] for l in df["label"]]
    fig=go.Figure(go.Bar(x=df["period"],y=df["excess_return"],marker_color=colors,
        text=[f"{v:+.1%}" for v in df["excess_return"]], textposition="outside",
        hovertemplate="%{x}<br>%{y:.2%}<br>%{customdata}<extra></extra>",
        customdata=df["factor_drivers"].values))
    _lay(fig, "Best & Worst Calendar Years: Excess Return")
    fig.update_layout(height=380, yaxis_tickformat=".1%")
    return fig

def ch_explained(r2):
    """Simple donut chart for explained vs unexplained."""
    if not HAS_PLOTLY: return None
    vals=[r2*100, (1-r2)*100]
    fig=go.Figure(go.Pie(values=vals, labels=["Factor-Explained","Residual (Unexplained)"],
        hole=0.55, marker=dict(colors=[C["blue"], C["grid"]]),
        textinfo="label+percent", textfont=dict(size=12),
        hovertemplate="%{label}: %{value:.1f}%<extra></extra>"))
    fig.update_layout(showlegend=False, height=250, width=300,
                      margin=dict(l=10,r=10,t=30,b=10),
                      font=dict(family="Arial, Helvetica Neue, sans-serif"),
                      annotations=[dict(text=f"{r2*100:.0f}%",x=0.5,y=0.5,font_size=24,
                                        font_color=C["navy"],showarrow=False)])
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  PDF EXPORT ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class ReportPDF(FPDF if HAS_FPDF else object):
    """Professional landscape PDF report generator."""
    # Brand colors (from corporate PPT template)
    _BLACK = (0, 0, 0)
    _BLUE = (34, 147, 189)    # #2293BD — primary brand blue
    _RED = (217, 83, 43)      # #D9532B — brand vermillion
    _GOLD = (250, 165, 26)    # #FAA51A — brand amber
    _CREAM = (240, 230, 221)  # #F0E6DD — warm cream
    _CREAM_LT = (251, 247, 243)  # #FBF7F3 — off-white
    _TXT = (34, 34, 34)       # #222222 — primary text
    _NEUT = (123, 114, 101)   # #7B7265 — muted text
    _CARD = (251, 247, 243)   # #FBF7F3 — card bg
    _NEG = (217, 83, 43)      # same as _RED
    _WH = (255, 255, 255)
    _GRID = (232, 224, 216)   # #E8E0D8 — warm grid
    # Landscape dimensions
    PW = 297  # page width
    PH = 210  # page height
    M = 12    # margin

    def __init__(self, strategy="", benchmark="", window="", as_of=""):
        if not HAS_FPDF: return
        super().__init__(orientation="L", format="A4")
        self.strategy = strategy
        self.benchmark = benchmark
        self.window = window
        self.as_of = as_of
        self.add_font("Arial", "", r"C:\Windows\Fonts\arial.ttf")
        self.add_font("Arial", "B", r"C:\Windows\Fonts\arialbd.ttf")
        self.add_font("Arial", "I", r"C:\Windows\Fonts\ariali.ttf")
        self.add_font("Arial", "BI", r"C:\Windows\Fonts\arialbi.ttf")
        self.add_font("Georgia", "", r"C:\Windows\Fonts\georgia.ttf")
        self.add_font("Georgia", "B", r"C:\Windows\Fonts\georgiab.ttf")
        self.add_font("Georgia", "I", r"C:\Windows\Fonts\georgiai.ttf")
        self.add_font("Courier", "", r"C:\Windows\Fonts\cour.ttf")
        self.add_font("Courier", "B", r"C:\Windows\Fonts\courbd.ttf")
        self.set_auto_page_break(auto=True, margin=18)

    @property
    def cw(self):
        """Content width (page minus margins)."""
        return self.PW - 2 * self.M

    def header(self):
        self.set_fill_color(*self._BLACK)
        self.rect(0, 0, self.PW, 10, "F")
        self.set_draw_color(*self._BLUE)
        self.line(0, 10, self.PW, 10)
        self.set_font("Georgia", "B", 7)
        self.set_text_color(*self._WH)
        self.set_xy(self.M, 2)
        self.cell(0, 6, "Manager Research  |  Factor Regime Analysis", align="L")
        self.set_font("Arial", "", 6.5)
        self.set_text_color(*self._CREAM)
        self.set_xy(self.PW - 100, 2)
        self.cell(88, 6, f"{self.window}  |  {self.as_of}", align="R")
        self.ln(12)

    def footer(self):
        self.set_y(-13)
        self.set_draw_color(*self._GRID)
        self.line(self.M, self.get_y(), self.PW - self.M, self.get_y())
        self.ln(2)
        self.set_font("Arial", "", 6)
        self.set_text_color(*self._NEUT)
        hw = self.cw / 2
        self.cell(hw, 3, f"{self.strategy}  |  {self.benchmark}", align="L")
        self.cell(hw, 3, f"Page {self.page_no()}", align="R")

    def section_title(self, title):
        self.set_font("Georgia", "B", 11)
        self.set_text_color(*self._BLACK)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*self._BLUE)
        self.line(self.M, self.get_y(), self.M + 55, self.get_y())
        self.ln(3)

    def body_text(self, text):
        self.set_font("Arial", "", 8.5)
        self.set_text_color(*self._TXT)
        clean = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        clean = re.sub(r'\*(.+?)\*', r'\1', clean)
        self.multi_cell(0, 4, clean)
        self.ln(1)

    def add_chart(self, fig, w=None, h=None):
        w = w or self.cw
        h = h or 70
        if fig is None: return
        # Check if chart would overflow the page; if so, start new page
        if self.get_y() + h > 178:
            self.add_page()
        try:
            imgb = fig.to_image(format="png", width=int(w * 4), height=int(h * 4), scale=2)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                f.write(imgb); tp = f.name
            self.image(tp, x=self.M, w=w)
            self.ln(3)
            os.unlink(tp)
        except Exception as exc:
            self.set_font("Arial", "I", 8)
            self.cell(0, 5, f"(Chart could not be rendered: {exc})",
                     new_x="LMARGIN", new_y="NEXT")

    # ── Summary page helpers ─────────────────────────────────────────────────
    def _kpi_box(self, x, y, w, label, value, sub, accent=None):
        accent = accent or self._BLUE
        self.set_fill_color(*self._WH)
        self.set_draw_color(*self._GRID)
        self.rect(x, y, w, 22, "DF")
        self.set_fill_color(*accent)
        self.rect(x, y, w, 2.5, "F")
        self.set_xy(x + 4, y + 5)
        self.set_font("Arial", "B", 6.5)
        self.set_text_color(*self._NEUT)
        self.cell(w - 8, 3, label.upper())
        self.set_xy(x + 4, y + 9.5)
        self.set_font("Georgia", "B", 15)
        self.set_text_color(*self._BLACK)
        self.cell(w - 8, 6, value)
        self.set_xy(x + 4, y + 17)
        self.set_font("Arial", "", 6.5)
        self.set_text_color(*self._NEUT)
        self.cell(w - 8, 3, sub)

    def _factor_row(self, y, name, direction, top_label, top_bps, top_hit,
                    bot_label, bot_bps, bot_hit, significant):
        row_w = self.cw
        self.set_fill_color(*self._WH)
        self.set_draw_color(*self._GRID)
        self.rect(self.M, y, row_w, 20, "DF")
        # Left: factor name + direction
        self.set_xy(self.M + 4, y + 2)
        self.set_font("Georgia", "B", 9)
        self.set_text_color(*self._BLACK)
        self.cell(70, 5, name)
        self.set_font("Arial", "", 6)
        if significant:
            self.set_text_color(*self._BLUE)
            self.cell(30, 5, "SIGNIFICANT")
        else:
            self.set_text_color(*self._NEUT)
            self.cell(30, 5, "NOT SIGNIFICANT")
        self.set_xy(self.M + 4, y + 8)
        self.set_font("Arial", "", 7)
        self.set_text_color(*self._NEUT)
        self.cell(120, 4, direction)
        # Top quartile box
        bx1 = self.M + row_w - 120
        self.set_fill_color(220, 240, 248)  # light blue tint
        self.rect(bx1, y + 2, 55, 16, "F")
        self.set_xy(bx1 + 3, y + 3)
        self.set_font("Arial", "B", 5.5)
        self.set_text_color(*self._BLUE)
        self.cell(49, 3, top_label.upper()[:40])
        self.set_xy(bx1 + 3, y + 7)
        self.set_font("Georgia", "B", 12)
        self.cell(49, 5, f"{top_bps:+.0f} bps/mo")
        self.set_xy(bx1 + 3, y + 13)
        self.set_font("Arial", "", 5.5)
        self.set_text_color(*self._NEUT)
        self.cell(49, 3, f"{top_hit:.0f}% months positive" if top_hit is not None else "")
        # Bottom quartile box
        bx2 = bx1 + 60
        self.set_fill_color(252, 234, 228)  # light vermillion tint
        self.rect(bx2, y + 2, 55, 16, "F")
        self.set_xy(bx2 + 3, y + 3)
        self.set_font("Arial", "B", 5.5)
        self.set_text_color(*self._RED)
        self.cell(49, 3, bot_label.upper()[:40])
        self.set_xy(bx2 + 3, y + 7)
        self.set_font("Georgia", "B", 12)
        self.cell(49, 5, f"{bot_bps:+.0f} bps/mo")
        self.set_xy(bx2 + 3, y + 13)
        self.set_font("Arial", "", 5.5)
        self.set_text_color(*self._NEUT)
        self.cell(49, 3, f"{bot_hit:.0f}% months positive" if bot_hit is not None else "")

    def _quarter_box(self, x, y, w, label, period, ret, drivers, is_best=True):
        accent = self._BLUE if is_best else self._RED
        self.set_fill_color(*self._WH)
        self.set_draw_color(*self._GRID)
        self.rect(x, y, w, 22, "DF")
        self.set_fill_color(*accent)
        self.rect(x, y, w, 2.5, "F")
        self.set_xy(x + 4, y + 5)
        self.set_font("Arial", "B", 6.5)
        self.set_text_color(*accent)
        self.cell(w - 8, 3, label.upper())
        self.set_xy(x + 4, y + 9.5)
        self.set_font("Georgia", "B", 13)
        self.set_text_color(*accent)
        self.cell(35, 5, f"{ret:+.1%}")
        self.set_font("Georgia", "B", 9)
        self.set_text_color(*self._BLACK)
        self.cell(w - 43, 5, period, align="R")
        self.set_xy(x + 4, y + 16)
        self.set_font("Arial", "", 6.5)
        self.set_text_color(*self._NEUT)
        self.cell(w - 8, 3, drivers[:100])

    def _outlook_box(self, x, y, w, h, label, text, dark=False):
        # Truncate text to avoid overflowing the box
        max_chars = int((w - 10) / 2.0 * (h - 9) / 4)  # rough estimate
        display_text = text[:max_chars] + "..." if len(text) > max_chars else text
        if dark:
            self.set_fill_color(*self._BLACK)
            self.rect(x, y, w, h, "F")
            self.set_xy(x + 5, y + 3)
            self.set_font("Arial", "B", 6.5)
            self.set_text_color(*self._BLUE)
            self.cell(w - 10, 3, label.upper())
            self.set_xy(x + 5, y + 9)
            self.set_font("Arial", "", 8)
            self.set_text_color(*self._CREAM)
            self.multi_cell(w - 10, 4, display_text)
        else:
            self.set_fill_color(*self._WH)
            self.set_draw_color(*self._GRID)
            self.rect(x, y, w, h, "DF")
            self.set_xy(x + 5, y + 3)
            self.set_font("Arial", "B", 6.5)
            self.set_text_color(*self._NEUT)
            self.cell(w - 10, 3, label.upper())
            self.set_xy(x + 5, y + 9)
            self.set_font("Arial", "", 8)
            self.set_text_color(*self._TXT)
            self.multi_cell(w - 10, 4, display_text)

    def add_footnotes(self):
        self.ln(5)
        self.set_font("Arial", "", 6.5)
        self.set_text_color(*self._NEUT)
        self.multi_cell(0, 3.5, METHODOLOGY)
        self.ln(2)
        self.multi_cell(0, 3.5, "Factor sign conventions: " +
                       " | ".join(f"{fl(k)}: {v}" for k, v in FSIGN.items() if k in FDISP))


def _summary_to_bullets(summary):
    """Convert structured summary dict to flat bullet strings for PDF/Excel export."""
    bullets = []
    for kpi in summary.get("kpis", []):
        bullets.append(f"{kpi['label']}: {kpi['value']} - {kpi['sub']}")
    if summary.get("narrative"):
        bullets.append(summary["narrative"])
    for f in summary.get("factors", []):
        tag = "Significant" if f["significant"] else "Not significant"
        bullets.append(f"{f['name']}: {f['impact_bps']:+.0f} bps impact ({tag}). {f['direction']}.")
    if summary.get("outlook"):
        bullets.append(f"Outlook: {summary['outlook']}")
    if summary.get("outlook_watch"):
        bullets.append(f"Watch: {summary['outlook_watch']}")
    bq, wq = summary.get("best_q"), summary.get("worst_q")
    if bq:
        bullets.append(f"Best quarter: {bq['period']} ({bq['excess_return']:+.1%}) - {bq['factor_drivers']}")
    if wq:
        bullets.append(f"Worst quarter: {wq['period']} ({wq['excess_return']:+.1%}) - {wq['factor_drivers']}")
    bullets.append(f"Analysis window: {summary.get('window','N/A')}. Based on {summary.get('n_months','N/A')} months.")
    return bullets


def build_strategy_pdf(sel_strat, benchmark, window_label, as_of, summary_dict,
                        sf, mf, rt, sp, nq, ny, fig_cum, fig_sf, fig_hm, fig_nq,
                        fig_mf=None, fig_roll=None, fig_ny=None):
    if not HAS_FPDF: return None
    pdf = ReportPDF(strategy=sel_strat, benchmark=benchmark, window=window_label, as_of=as_of)
    s = summary_dict
    M = pdf.M
    CW = pdf.cw  # content width ~273

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 1: STRATEGY SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    pdf.add_page()

    # Title block
    y0 = pdf.get_y()
    pdf.set_fill_color(*pdf._BLACK)
    pdf.rect(M, y0, CW, 16, "F")
    pdf.set_xy(M + 5, y0 + 2)
    pdf.set_font("Georgia", "B", 15)
    pdf.set_text_color(*pdf._WH)
    pdf.cell(0, 6, "Factor Analysis Summary")
    pdf.set_xy(M + 5, y0 + 9)
    pdf.set_font("Arial", "", 8.5)
    pdf.set_text_color(*pdf._CREAM)
    pdf.cell(0, 5, f"{sel_strat}  |  Benchmark: {benchmark}")
    pdf.set_y(y0 + 19)

    # KPI row
    kpis = s.get("kpis", [])
    if kpis:
        n_kpis = len(kpis)
        gap = 5
        kpi_w = (CW - (n_kpis - 1) * gap) / n_kpis
        ky = pdf.get_y()
        for i, kpi in enumerate(kpis):
            kx = M + i * (kpi_w + gap)
            accent = pdf._BLUE
            if kpi.get("color") == "neg": accent = pdf._RED
            elif kpi.get("color") == "neut": accent = pdf._NEUT
            pdf._kpi_box(kx, ky, kpi_w, kpi["label"], kpi["value"], kpi["sub"], accent)
        pdf.set_y(ky + 26)

    # Narrative
    narr = s.get("narrative", "")
    if narr:
        ny0 = pdf.get_y()
        pdf.set_fill_color(*pdf._WH)
        pdf.set_draw_color(*pdf._GRID)
        pdf.rect(M, ny0, CW, 18, "DF")
        pdf.set_xy(M + 5, ny0 + 3)
        pdf.set_font("Arial", "B", 6.5)
        pdf.set_text_color(*pdf._NEUT)
        pdf.cell(0, 3, "ANALYSIS OVERVIEW")
        pdf.set_xy(M + 5, ny0 + 8)
        pdf.set_font("Arial", "", 8.5)
        pdf.set_text_color(*pdf._TXT)
        pdf.multi_cell(CW - 10, 4, narr)
        actual_bottom = pdf.get_y() + 2
        if actual_bottom > ny0 + 18:
            pdf.set_draw_color(*pdf._GRID)
            pdf.rect(M, ny0, CW, actual_bottom - ny0, "D")
        pdf.set_y(actual_bottom + 2)

    # Factor regime rows
    facs = s.get("factors", [])
    for f in facs:
        fy = pdf.get_y()
        if fy + 22 > 178:  # factor row is 20mm + 2mm gap; usable ≈ 178mm
            pdf.add_page()
            fy = pdf.get_y()
        if f["raw"] == "style":
            top_label = "Top Quartile (Growth-Led)"
            bot_label = "Bottom Quartile (Value-Led)"
        else:
            top_label = f'Top Quartile ({f["name"]} In Favor)'
            bot_label = f'Bottom Quartile ({f["name"]} Out of Favor)'
        top_bps = f.get("top_bps", 0) or 0
        bot_bps = f.get("bot_bps", 0) or 0
        pdf._factor_row(fy, f["name"], f["direction"], top_label, top_bps,
                        f.get("hit_top"), bot_label, bot_bps, f.get("hit_bot"),
                        f["significant"])
        pdf.set_y(fy + 22)

    # Outlook + quarter cards row (side by side in landscape)
    outlook = s.get("outlook", "")
    watch = s.get("outlook_watch", "")
    bq, wq = s.get("best_q"), s.get("worst_q")
    oy = pdf.get_y() + 1
    if oy + 25 > 178:  # outlook boxes are 22mm + 3mm gap
        pdf.add_page()
        oy = pdf.get_y()
    half = (CW - 5) / 2
    # Left side: outlook
    if outlook and watch:
        pdf._outlook_box(M, oy, half, 22, "What to Expect", outlook, dark=True)
        pdf._outlook_box(M + half + 5, oy, half, 22, "What to Watch", watch, dark=False)
    elif outlook:
        pdf._outlook_box(M, oy, CW, 22, "What to Expect", outlook, dark=True)
    if outlook or watch:
        pdf.set_y(oy + 25)

    # Best/worst quarter cards
    if bq or wq:
        qy = pdf.get_y()
        if qy + 25 > 178:  # quarter cards are 22mm + 3mm gap
            pdf.add_page()
            qy = pdf.get_y()
        if bq:
            pdf._quarter_box(M, qy, half, "Best Quarter", bq["period"],
                             bq["excess_return"], bq["factor_drivers"], True)
        if wq:
            pdf._quarter_box(M + half + 5, qy, half, "Worst Quarter", wq["period"],
                             wq["excess_return"], wq["factor_drivers"], False)
        pdf.set_y(qy + 25)

    # Data context bar
    dy = pdf.get_y()
    if dy + 10 > 178: pdf.add_page(); dy = pdf.get_y()
    pdf.set_fill_color(*pdf._CARD)
    pdf.rect(M, dy, CW, 7, "F")
    pdf.set_xy(M + 4, dy + 1.5)
    pdf.set_font("Arial", "", 6.5)
    pdf.set_text_color(*pdf._NEUT)
    n_mo = s.get("n_months", "")
    win = s.get("window", "")
    n_sig = len([f for f in facs if f["significant"]])
    pdf.cell(0, 4, f"{n_mo} months  |  {n_sig} significant factors  |  Window: {win}")

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 2: CUMULATIVE EXCESS RETURN
    # ══════════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section_title("Cumulative Excess Return")
    pdf.add_chart(fig_cum, h=70)

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 3: REGIME HEATMAP
    # ══════════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section_title("Factor Regime Analysis")
    pdf.add_chart(fig_hm, h=75)

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 4: FACTOR ANALYSIS CHARTS
    # ══════════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section_title("Factor Impact")
    pdf.add_chart(fig_sf, h=70)

    if fig_mf:
        pdf.section_title("Multi-Factor Coefficients")
        pdf.add_chart(fig_mf, h=55)
    elif not mf.empty:
        pdf.section_title("Multi-Factor Regression")
        pdf.set_font("Arial", "", 8.5)
        pdf.set_text_color(*pdf._TXT)
        pdf.cell(0, 5, f"R\u00b2 = {mf.attrs.get('r2',0):.4f}  |  Adj R\u00b2 = {mf.attrs.get('adj_r2',0):.4f}  |  N = {mf.attrs.get('n',0)}",
                 new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Courier", "", 7.5)
        for _, row in mf.iterrows():
            lbl = fl(row["var"]) if row["var"]!="alpha" else "Alpha"
            sig = " *" if row["p"] < 0.05 else ""
            pdf.cell(0, 4.5, f"  {lbl:20s} {row['coef']:+8.4f}  t={row['t']:6.2f}  p={row['p']:.4f}{sig}",
                     new_x="LMARGIN", new_y="NEXT")

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 5: NOTABLE QUARTERS
    # ══════════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section_title("Notable Quarters")
    pdf.add_chart(fig_nq, h=70)

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 6: NOTABLE CALENDAR YEARS
    # ══════════════════════════════════════════════════════════════════════════
    if fig_ny:
        pdf.add_page()
        pdf.section_title("Notable Calendar Years")
        pdf.add_chart(fig_ny, h=70)

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 7: ROLLING BETAS (if available)
    # ══════════════════════════════════════════════════════════════════════════
    if fig_roll:
        pdf.add_page()
        pdf.section_title("Rolling Factor Betas (36-Month Window)")
        pdf.add_chart(fig_roll, h=80)

    # ══════════════════════════════════════════════════════════════════════════
    # FINAL: METHODOLOGY
    # ══════════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section_title("Methodology & Disclosures")
    pdf.add_footnotes()

    return bytes(pdf.output())

# ═══════════════════════════════════════════════════════════════════════════════
#  STREAMLIT APP
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Manager Research Dashboard",
                   layout="wide", initial_sidebar_state="expanded")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""<style>
.block-container {{ padding-top: 1rem; font-family: Arial, 'Helvetica Neue', sans-serif; }}
h1 {{ color: {C["navy"]}; font-family: Georgia, 'Times New Roman', serif; font-weight: 700; }}
h2 {{ color: {C["blue"]}; font-family: Georgia, 'Times New Roman', serif; font-weight: 600;
      border-bottom: 2px solid {C["blue"]}; padding-bottom: 6px; }}
h3 {{ color: {C["txt"]}; font-family: Georgia, 'Times New Roman', serif; font-weight: 600; }}
.stMetric > div {{ background: {C["card"]}; border-radius: 8px; padding: 8px 14px;
                   border-left: 3px solid {C["blue"]}; }}
div[data-testid="stExpander"] {{ border: 1px solid {C["grid"]}; border-radius: 8px; }}
.mhdr {{ background: {C["navy_dk"]}; padding: 22px 28px; border-radius: 0; margin-bottom: 8px; }}
.mhdr h1 {{ color: {C["wh"]} !important; margin: 0; font-size: 1.5rem;
            font-family: Georgia, 'Times New Roman', serif !important; }}
.mhdr p {{ color: {C["cream"]}; margin: 4px 0 0 0; font-size: 0.85rem; }}
.rleg {{ background: {C["card"]}; padding: 10px 16px; border-radius: 6px;
         font-size: 0.82rem; color: {C["neut"]}; margin-bottom: 10px; }}
.ctx {{ background: {C["card"]}; padding: 10px 14px; border-radius: 6px;
        font-size: 0.82rem; margin-bottom: 8px; color: {C["neut"]}; }}
.ncard {{ padding: 8px 0; border-bottom: 1px solid {C["grid"]}; }}
.ncard:last-child {{ border-bottom: none; }}
.wsel {{ background: {C["blue"]}; color: {C["wh"]}; padding: 8px 16px; border-radius: 4px;
         font-size: 0.85rem; margin-bottom: 12px; display: inline-block; }}

/* ── Summary section ────────────────────────────────────────────────── */
.sum-section {{ margin-bottom: 28px; }}
.sum-kpi-row {{ display: flex; gap: 16px; margin-bottom: 24px; }}
.sum-kpi {{ flex: 1; background: {C["wh"]}; border: 1px solid {C["grid"]};
            border-radius: 8px; padding: 20px 22px; position: relative;
            overflow: hidden; transition: box-shadow 0.2s; }}
.sum-kpi:hover {{ box-shadow: 0 4px 16px rgba(0,0,0,0.06); }}
.sum-kpi::before {{ content: ''; position: absolute; top: 0; left: 0; right: 0;
                    height: 3px; border-radius: 8px 8px 0 0; }}
.sum-kpi.kpi-teal::before {{ background: {C["blue"]}; }}
.sum-kpi.kpi-pos::before {{ background: {C["blue"]}; }}
.sum-kpi.kpi-neg::before {{ background: {C["neg"]}; }}
.sum-kpi.kpi-neut::before {{ background: {C["neut"]}; }}
.sum-kpi .kpi-label {{ font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
                       letter-spacing: 0.08em; color: {C["neut"]}; margin-bottom: 6px; }}
.sum-kpi .kpi-value {{ font-size: 1.65rem; font-weight: 700; color: {C["navy"]};
                       font-family: Georgia, serif; line-height: 1.2; margin-bottom: 4px; }}
.sum-kpi .kpi-sub {{ font-size: 0.78rem; color: {C["neut"]}; line-height: 1.35; }}

.sum-narrative {{ background: {C["wh"]}; border: 1px solid {C["grid"]};
                  border-radius: 8px; padding: 24px 28px; margin-bottom: 24px; }}
.sum-narrative .nar-title {{ font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
                             letter-spacing: 0.08em; color: {C["neut"]}; margin-bottom: 10px; }}
.sum-narrative .nar-body {{ font-size: 0.95rem; color: {C["txt"]}; line-height: 1.7; }}

.sum-factors {{ display: flex; flex-wrap: wrap; gap: 14px; margin-bottom: 24px; }}
.sum-factor {{ flex: 1; min-width: 280px; background: {C["wh"]}; border: 1px solid {C["grid"]};
               border-radius: 8px; padding: 20px 22px; transition: box-shadow 0.2s; }}
.sum-factor:hover {{ box-shadow: 0 4px 16px rgba(0,0,0,0.06); }}
.sum-factor .fac-header {{ display: flex; align-items: center; justify-content: space-between;
                           margin-bottom: 12px; }}
.sum-factor .fac-name {{ font-size: 0.95rem; font-weight: 600; color: {C["navy"]};
                         font-family: Georgia, serif; }}
.sum-factor .fac-badge {{ font-size: 0.68rem; font-weight: 600; padding: 3px 10px;
                          border-radius: 20px; text-transform: uppercase; letter-spacing: 0.04em; }}
.fac-badge.sig {{ background: rgba(34,147,189,0.10); color: {C["blue"]}; }}
.fac-badge.nsig {{ background: {C["card"]}; color: {C["neut"]}; }}
.sum-factor .fac-direction {{ font-size: 0.85rem; color: {C["txt"]}; line-height: 1.4;
                              font-weight: 500; margin-bottom: 12px; }}
.sum-factor .fac-regimes {{ display: flex; gap: 10px; margin-bottom: 8px; }}
.sum-factor .fac-regime {{ flex: 1; border-radius: 6px; padding: 10px 12px; text-align: center; }}
.fac-regime.regime-top {{ background: rgba(34,147,189,0.07); }}
.fac-regime.regime-bot {{ background: rgba(217,83,43,0.06); }}
.fac-regime .regime-label {{ font-size: 0.65rem; font-weight: 600; text-transform: uppercase;
                             letter-spacing: 0.06em; margin-bottom: 4px; }}
.regime-top .regime-label {{ color: {C["blue"]}; }}
.regime-bot .regime-label {{ color: {C["neg"]}; }}
.fac-regime .regime-val {{ font-size: 1.2rem; font-weight: 700; line-height: 1.2;
                           font-family: Georgia, serif; }}
.regime-top .regime-val {{ color: {C["blue"]}; }}
.regime-bot .regime-val {{ color: {C["neg"]}; }}
.fac-regime .regime-sub {{ font-size: 0.7rem; color: {C["neut"]}; margin-top: 2px; }}
.sum-factor .fac-meta {{ font-size: 0.72rem; color: {C["neut"]}; display: flex;
                         justify-content: space-between; padding-top: 8px;
                         border-top: 1px solid {C["grid"]}; }}

.sum-outlook {{ display: flex; gap: 16px; margin-bottom: 24px; }}
.sum-outlook-card {{ flex: 1; border-radius: 8px; padding: 20px 22px; }}
.outlook-expect {{ background: {C["navy_dk"]}; color: {C["wh"]}; }}
.outlook-watch {{ background: {C["wh"]}; border: 1px solid {C["grid"]}; }}
.sum-outlook-card .out-label {{ font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
                                letter-spacing: 0.08em; margin-bottom: 8px; }}
.outlook-expect .out-label {{ color: {C["blue"]}; }}
.outlook-watch .out-label {{ color: {C["neut"]}; }}
.sum-outlook-card .out-body {{ font-size: 0.9rem; line-height: 1.6; }}
.outlook-expect .out-body {{ color: {C["cream"]}; }}
.outlook-watch .out-body {{ color: {C["txt"]}; }}

.sum-quarters {{ display: flex; gap: 16px; }}
.sum-qcard {{ flex: 1; border-radius: 8px; padding: 18px 20px; border: 1px solid {C["grid"]};
              background: {C["wh"]}; }}
.sum-qcard .qc-label {{ font-size: 0.68rem; font-weight: 600; text-transform: uppercase;
                         letter-spacing: 0.06em; margin-bottom: 6px; }}
.qc-best .qc-label {{ color: {C["blue"]}; }}
.qc-worst .qc-label {{ color: {C["neg"]}; }}
.sum-qcard .qc-period {{ font-size: 0.85rem; font-weight: 600; color: {C["navy"]};
                          font-family: Georgia, serif; }}
.sum-qcard .qc-return {{ font-size: 1.3rem; font-weight: 700; margin: 4px 0;
                          font-family: Georgia, serif; }}
.qc-best .qc-return {{ color: {C["blue"]}; }}
.qc-worst .qc-return {{ color: {C["neg"]}; }}
.sum-qcard .qc-drivers {{ font-size: 0.78rem; color: {C["neut"]}; line-height: 1.4; }}

.sum-databar {{ background: {C["card"]}; border-radius: 6px; padding: 10px 16px;
                font-size: 0.78rem; color: {C["neut"]}; margin-top: 16px;
                display: flex; gap: 18px; align-items: center; flex-wrap: wrap; }}
.sum-databar span {{ display: inline-flex; align-items: center; gap: 4px; }}
.sum-databar .db-dot {{ width: 6px; height: 6px; border-radius: 50%;
                        background: {C["blue"]}; display: inline-block; }}

/* ── Pairing recommendations ────────────────────────────────────────── */
.pair-section {{ margin-top: 32px; }}
.pair-section-title {{ font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
                       letter-spacing: 0.08em; color: {C["neut"]}; margin-bottom: 16px; }}
.pair-card {{ background: {C["wh"]}; border: 1px solid {C["grid"]}; border-radius: 8px;
              padding: 22px 24px; margin-bottom: 14px; transition: box-shadow 0.2s;
              position: relative; overflow: hidden; }}
.pair-card:hover {{ box-shadow: 0 4px 16px rgba(0,0,0,0.06); }}
.pair-card::before {{ content: ''; position: absolute; top: 0; left: 0; right: 0;
                      height: 3px; background: linear-gradient(90deg, {C["blue"]}, {C["gold"]}); }}
.pair-header {{ display: flex; justify-content: space-between; align-items: flex-start;
                margin-bottom: 14px; }}
.pair-names {{ flex: 1; }}
.pair-names .pair-label {{ font-size: 0.68rem; font-weight: 600; text-transform: uppercase;
                           letter-spacing: 0.06em; color: {C["neut"]}; margin-bottom: 4px; }}
.pair-names .pair-strat {{ font-size: 0.9rem; font-weight: 600; color: {C["navy"]};
                           line-height: 1.4; }}
.pair-corr {{ text-align: right; min-width: 120px; }}
.pair-corr .corr-label {{ font-size: 0.68rem; font-weight: 600; text-transform: uppercase;
                          letter-spacing: 0.06em; color: {C["neut"]}; margin-bottom: 2px; }}
.pair-corr .corr-value {{ font-size: 1.5rem; font-weight: 700; font-family: Georgia, serif; }}
.corr-neg {{ color: {C["blue"]}; }}
.corr-low {{ color: {C["warn"]}; }}
.corr-med {{ color: {C["neut"]}; }}
.pair-rationale {{ display: flex; gap: 12px; flex-wrap: wrap; }}
.pair-chip {{ background: {C["card"]}; border-radius: 6px; padding: 8px 14px;
              font-size: 0.8rem; color: {C["txt"]}; line-height: 1.4; flex: 1; min-width: 200px; }}
.pair-chip .chip-label {{ font-size: 0.68rem; font-weight: 600; text-transform: uppercase;
                          letter-spacing: 0.04em; color: {C["neut"]}; margin-bottom: 3px; }}
.pair-chip .chip-factor {{ display: flex; justify-content: space-between; align-items: center;
                           padding: 2px 0; }}
.chip-pos {{ color: {C["blue"]}; font-weight: 600; }}
.chip-neg {{ color: {C["neg"]}; font-weight: 600; }}
.pair-rank {{ display: inline-flex; align-items: center; justify-content: center;
              width: 26px; height: 26px; border-radius: 50%; background: {C["blue"]};
              color: {C["wh"]}; font-size: 0.75rem; font-weight: 700; margin-right: 10px;
              flex-shrink: 0; }}

/* ── Factor tilt rows ───────────────────────────────────────────────── */
.pair-tilts {{ margin-bottom: 14px; }}
.tilt-row {{ display: flex; align-items: center; gap: 10px; padding: 8px 0;
             border-bottom: 1px solid {C["grid"]}; }}
.tilt-row:last-child {{ border-bottom: none; }}
.tilt-factor-name {{ font-size: 0.82rem; font-weight: 600; color: {C["navy"]};
                     min-width: 130px; font-family: Georgia, serif; }}
.tilt-strats {{ display: flex; gap: 16px; flex: 1; }}
.tilt-strat {{ display: flex; flex-direction: column; flex: 1;
               background: {C["card"]}; border-radius: 6px; padding: 6px 12px; }}
.tilt-strat-name {{ font-size: 0.68rem; color: {C["neut"]}; margin-bottom: 2px; }}
.tilt-strat-label {{ font-size: 0.82rem; font-weight: 600; }}
.tilt-strat-beta {{ font-size: 0.72rem; color: {C["neut"]}; margin-top: 1px; }}
.tilt-pos {{ color: {C["blue"]}; }}
.tilt-neg {{ color: {C["neg"]}; }}

/* ── Blended performance stats ──────────────────────────────────────── */
.blend-section {{ margin-top: 16px; padding-top: 14px; border-top: 1px solid {C["grid"]}; }}
.blend-title {{ font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
                letter-spacing: 0.08em; color: {C["neut"]}; margin-bottom: 10px; }}
.blend-grid {{ display: grid; grid-template-columns: 140px 1fr 1fr 1fr; gap: 0;
               font-size: 0.82rem; }}
.blend-hdr {{ font-size: 0.68rem; font-weight: 600; text-transform: uppercase;
              letter-spacing: 0.04em; color: {C["neut"]}; padding: 4px 10px 8px;
              border-bottom: 2px solid {C["grid"]}; }}
.blend-metric {{ padding: 6px 10px; border-bottom: 1px solid {C["grid"]};
                 font-weight: 500; color: {C["txt"]}; }}
.blend-val {{ padding: 6px 10px; border-bottom: 1px solid {C["grid"]};
              text-align: right; }}
.blend-val.blend-highlight {{ background: rgba(34,147,189,0.06); font-weight: 600;
                              color: {C["navy"]}; }}
.blend-best {{ color: {C["blue"]}; font-weight: 600; }}
.blend-worst {{ color: {C["neg"]}; font-weight: 600; }}
</style>""", unsafe_allow_html=True)

for k in ["strategy_df", "factor_df", "u_s", "u_f"]:
    if k not in st.session_state: st.session_state[k] = None

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""<div class="mhdr">
    <h1>Manager Research Dashboard</h1>
    <p>Factor Regime Analysis &amp; Manager Evaluation</p>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR: Upload + Global Controls
# ══════════════════════════════════════════════════════════════════════════════
sb = st.sidebar
sb.markdown(f"""<div style="text-align:center; padding:6px 0 10px;">
    <span style="font-size:1.2rem; font-weight:700; color:{C['navy']}; font-family:Georgia,serif;">Manager Research</span><br>
    <span style="font-size:0.72rem; color:{C['neut']};">Factor Regime Analysis</span>
</div>""", unsafe_allow_html=True)
sb.markdown("---")

mpk = []
if not HAS_PLOTLY: mpk.append("plotly")
if not HAS_SM: mpk.append("statsmodels")
if not HAS_FPDF: mpk.append("fpdf2")
if mpk: sb.warning(f"Optional: `pip install {' '.join(mpk)}`")

# ── Sidebar: upload data ──────────────────────────────────────────────────────
sb.header("1 \u00b7 Data")
sf_fmt = sb.selectbox("Strategy format", ["eVestment Compare Export", "Clean Template (Excel/CSV)"], key="sb_sf_fmt")
sf_file = sb.file_uploader("Strategy file", type=["xlsx","xls","csv"], key="up_s")
ff_fmt = sb.selectbox("Factor format", ["Multiple Risk Premiums Export", "Clean Template (Excel/CSV)"], key="sb_ff_fmt")
ff_file = sb.file_uploader("Factor file", type=["xlsx","xls","csv"], key="up_f")

if sb.button("\U0001f504  Load & Parse", use_container_width=True, type="primary"):
    if sf_file is not None:
        try:
            tp = os.path.join(tempfile.gettempdir(), "mrd_s.xlsx")
            with open(tp,"wb") as f: f.write(sf_file.getbuffer())
            sdf = parse_evestment(tp) if sf_fmt=="eVestment Compare Export" else parse_clean_strategy(tp)
            try: os.unlink(tp)
            except OSError: pass
            u, sdf["excess_return"] = _detect_units(sdf["excess_return"])
            sdf = _monthly(sdf)
            sdf["asset_class"] = sdf["benchmark"].apply(_classify) if "benchmark" in sdf.columns else "Unknown"
            st.session_state["strategy_df"]=sdf; st.session_state["u_s"]=u
            sb.success(f"\u2705 {sdf['strategy'].nunique()} strategies")
        except Exception as e: sb.error(f"\u274c {e}")

    if ff_file is not None:
        try:
            tp = os.path.join(tempfile.gettempdir(), "mrd_f.xlsx")
            with open(tp,"wb") as f: f.write(ff_file.getbuffer())
            fdf = parse_risk_premiums(tp) if ff_fmt=="Multiple Risk Premiums Export" else parse_clean_factors(tp)
            try: os.unlink(tp)
            except OSError: pass
            fc=[c for c in fdf.columns if c!="date"]
            fu, _ = _detect_units(pd.concat([fdf[c] for c in fc]))
            if fu=="percent":
                for c in fc: fdf[c]=fdf[c]/100.0
            fdf=_monthly(fdf)
            st.session_state["factor_df"]=fdf; st.session_state["u_f"]=fu
            sb.success(f"\u2705 {len(fc)} factors")
        except Exception as e: sb.error(f"\u274c {e}")

strategy_df = st.session_state.get("strategy_df")
factor_df = st.session_state.get("factor_df")

if strategy_df is None and factor_df is None:
    st.info("\U0001f448  Upload strategy & factor files in the sidebar, then click **Load & Parse**.")
    st.stop()
if strategy_df is None or factor_df is None:
    st.warning("Load both strategy and factor data to run analysis."); st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CONTROLS (top of page, applies everywhere)
# ══════════════════════════════════════════════════════════════════════════════
gc1, gc2, gc3 = st.columns([2, 1, 1])
with gc1:
    window_label = st.radio("Analysis Window", ["Full Period", "Most Recent 10 Years", "Most Recent 5 Years"],
                             horizontal=True, key="global_window")
with gc2:
    latest = strategy_df["date"].max()
    st.markdown(f'<span class="wsel">As-of: {latest:%B %Y}</span>', unsafe_allow_html=True)
with gc3:
    page = st.radio("View", ["\U0001f4c4 Strategy Detail", "\U0001f50d Peer Comparison"], horizontal=True)

# Apply global window filter to factor data
fdf_w = _window_filter(factor_df.copy(), window_label)
sdf_w = _window_filter(strategy_df.copy(), window_label)

as_of_str = f"{latest:%b %Y}"
if window_label != "Full Period":
    win_start = fdf_w["date"].min()
    st.caption(f"Window: {win_start:%b %Y} \u2013 {latest:%b %Y} ({len(fdf_w)} factor months)")

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: STRATEGY DETAIL
# ══════════════════════════════════════════════════════════════════════════════
if "\U0001f4c4" in page:

    # ── Strategy selector ─────────────────────────────────────────────────────
    fc1, fc2 = st.columns([1, 2])
    with fc1:
        acs = sorted(sdf_w["asset_class"].unique())
        sel_ac = st.multiselect("Asset class", acs, default=["Equity"] if "Equity" in acs else acs[:1])
        filt = sdf_w[sdf_w["asset_class"].isin(sel_ac)]
        if "benchmark" in filt.columns:
            benches = sorted(filt["benchmark"].unique())
            sel_bm = st.multiselect("Benchmark (optional)", benches, default=[])
            if sel_bm: filt = filt[filt["benchmark"].isin(sel_bm)]
    with fc2:
        strats = sorted(filt["strategy"].unique())
        if not strats: st.warning("No strategies match filters."); st.stop()
        sel_strat = st.selectbox("Select strategy", strats)
        meta = filt[filt["strategy"]==sel_strat].iloc[0]
        mc = st.columns(3)
        mc[0].markdown(f"**Firm:** {meta.get('firm_name','N/A')}")
        mc[1].markdown(f"**Product:** {meta.get('product_name','N/A')}")
        mc[2].markdown(f"**Benchmark:** {meta.get('benchmark','N/A')}")

    # ── Align ─────────────────────────────────────────────────────────────────
    sd = sdf_w[sdf_w["strategy"]==sel_strat][["date","excess_return"]].dropna()
    try: sa, fa, ainfo = _align(sd, fdf_w)
    except ValueError as e: st.error(str(e)); st.stop()
    excess = sa["excess_return"].reset_index(drop=True)
    dates = sa["date"].reset_index(drop=True)
    factors = fa.reset_index(drop=True)
    n_mo = ainfo["common"]

    if n_mo < 36: st.warning(f"\u26a0\ufe0f Only {n_mo} months of overlap. Results may be unreliable (36+ recommended).")

    # ── Compute ───────────────────────────────────────────────────────────────
    rt = all_regimes(excess, factors, 4)
    sp = spread_summary(rt, 4) if not rt.empty else pd.DataFrame()
    sf_tbl = run_sf(excess, factors)
    mf_tbl = run_mf(excess, factors) if HAS_SM else pd.DataFrame()
    nq = notable_quarters(dates, excess, factors)
    ny = notable_years(dates, excess, factors)
    summary = make_summary(sf_tbl, rt, sp, mf_tbl, n_mo, nq, ny, window_label)

    # ══════════════════════════════════════════════════════════════════════════
    # SUMMARY (visible without scrolling)
    # ══════════════════════════════════════════════════════════════════════════

    st.markdown('<div class="sum-section">', unsafe_allow_html=True)

    # ── KPI row ──────────────────────────────────────────────────────────────
    kpi_html = '<div class="sum-kpi-row">'
    for kpi in summary["kpis"]:
        kpi_html += (
            f'<div class="sum-kpi kpi-{kpi["color"]}">'
            f'  <div class="kpi-label">{kpi["label"]}</div>'
            f'  <div class="kpi-value">{kpi["value"]}</div>'
            f'  <div class="kpi-sub">{kpi["sub"]}</div>'
            f'</div>'
        )
    kpi_html += '</div>'
    st.markdown(kpi_html, unsafe_allow_html=True)

    # ── Narrative ────────────────────────────────────────────────────────────
    st.markdown(
        f'<div class="sum-narrative">'
        f'  <div class="nar-title">Analysis Overview</div>'
        f'  <div class="nar-body">{summary["narrative"]}</div>'
        f'</div>', unsafe_allow_html=True)

    # ── Factor relationship cards ────────────────────────────────────────────
    facs = summary["factors"]
    if facs:
        fac_html = '<div class="sum-factors">'
        for f in facs:
            badge_cls = "sig" if f["significant"] else "nsig"
            badge_txt = "Significant" if f["significant"] else "Not significant"

            # Top-quartile label
            if f["raw"] == "style":
                top_label = "Growth-Led Months"
                bot_label = "Value-Led Months"
            else:
                top_label = f'{f["name"]} In Favor'
                bot_label = f'{f["name"]} Out of Favor'

            fac_html += (
                f'<div class="sum-factor">'
                f'  <div class="fac-header">'
                f'    <span class="fac-name">{f["name"]}</span>'
                f'    <span class="fac-badge {badge_cls}">{badge_txt}</span>'
                f'  </div>'
                f'  <div class="fac-direction">{f["direction"]}</div>'
            )

            # Regime boxes — the intuitive part
            if f.get("top_bps") is not None and f.get("bot_bps") is not None:
                top_v = f["top_bps"]
                bot_v = f["bot_bps"]
                hit_top = f.get("hit_top")
                hit_bot = f.get("hit_bot")
                top_hit_str = f'{hit_top:.0f}% of months were positive' if hit_top is not None else ''
                bot_hit_str = f'{hit_bot:.0f}% of months were positive' if hit_bot is not None else ''
                fac_html += (
                    f'<div style="font-size:0.72rem;color:{C["neut"]};margin-bottom:6px;">'
                    f'Avg. strategy excess return during top &amp; bottom quartile months for {f["name"]}:</div>'
                    f'<div class="fac-regimes">'
                    f'  <div class="fac-regime regime-top">'
                    f'    <div class="regime-label">Top Quartile ({top_label})</div>'
                    f'    <div class="regime-val">{top_v:+.0f} bps/mo</div>'
                    f'    <div class="regime-sub">{top_hit_str}</div>'
                    f'  </div>'
                    f'  <div class="fac-regime regime-bot">'
                    f'    <div class="regime-label">Bottom Quartile ({bot_label})</div>'
                    f'    <div class="regime-val">{bot_v:+.0f} bps/mo</div>'
                    f'    <div class="regime-sub">{bot_hit_str}</div>'
                    f'  </div>'
                    f'</div>'
                )

            # Subtle meta row with the technical stats
            fac_html += (
                f'<div class="fac-meta">'
                f'  <span>Spread: {abs(f.get("top_bps",0) - f.get("bot_bps",0)):.0f} bps</span>'
                f'  <span>R\u00b2: {f["r2_pct"]:.1f}%</span>'
                f'</div>'
            )

            fac_html += '</div>'
        fac_html += '</div>'
        st.markdown(fac_html, unsafe_allow_html=True)

    # ── Outlook cards ────────────────────────────────────────────────────────
    if summary["outlook"]:
        outlook_html = '<div class="sum-outlook">'
        outlook_html += (
            f'<div class="sum-outlook-card outlook-expect">'
            f'  <div class="out-label">What to Expect</div>'
            f'  <div class="out-body">{summary["outlook"]}</div>'
            f'</div>'
        )
        if summary["outlook_watch"]:
            outlook_html += (
                f'<div class="sum-outlook-card outlook-watch">'
                f'  <div class="out-label">What to Watch</div>'
                f'  <div class="out-body">{summary["outlook_watch"]}</div>'
                f'</div>'
            )
        outlook_html += '</div>'
        st.markdown(outlook_html, unsafe_allow_html=True)

    # ── Best / Worst quarter cards ───────────────────────────────────────────
    bq, wq = summary["best_q"], summary["worst_q"]
    if bq or wq:
        q_html = '<div class="sum-quarters">'
        if bq:
            q_html += (
                f'<div class="sum-qcard qc-best">'
                f'  <div class="qc-label">Best Quarter</div>'
                f'  <div class="qc-period">{bq["period"]}</div>'
                f'  <div class="qc-return">{bq["excess_return"]:+.1%}</div>'
                f'  <div class="qc-drivers">{bq["factor_drivers"]}</div>'
                f'</div>'
            )
        if wq:
            q_html += (
                f'<div class="sum-qcard qc-worst">'
                f'  <div class="qc-label">Worst Quarter</div>'
                f'  <div class="qc-period">{wq["period"]}</div>'
                f'  <div class="qc-return">{wq["excess_return"]:+.1%}</div>'
                f'  <div class="qc-drivers">{wq["factor_drivers"]}</div>'
                f'</div>'
            )
        q_html += '</div>'
        st.markdown(q_html, unsafe_allow_html=True)

    # ── Data context bar ─────────────────────────────────────────────────────
    dmin, dmax = dates.min(), dates.max()
    fac_names = ", ".join(fl(c) for c in [col for col in factors.columns if col!="date"])
    st.markdown(
        f'<div class="sum-databar">'
        f'  <span><span class="db-dot"></span> {n_mo} months</span>'
        f'  <span>{dmin:%b %Y} \u2013 {dmax:%b %Y}</span>'
        f'  <span>{len([f for f in facs if f["significant"]])} significant factors</span>'
        f'  <span>{window_label}</span>'
        f'</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("Factor sign conventions"):
        for fc in [c for c in factors.columns if c!="date"]:
            tip = FSIGN.get(fc, "")
            st.markdown(f"**{fl(fc)}:** {tip}" if tip else f"**{fl(fc)}**")

    # ══════════════════════════════════════════════════════════════════════════
    # CUMULATIVE
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.header("Performance Overview")
    fig_cum = ch_cum(dates, excess, sel_strat)
    if fig_cum: st.plotly_chart(fig_cum, width='stretch')

    # ══════════════════════════════════════════════════════════════════════════
    # NOTABLE QUARTERS & YEARS
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.header("When Did It Work? When Didn't It?")
    st.markdown("*Quarters and years where factor environments mattered most, with drivers underneath.*")

    tab_q, tab_y = st.tabs(["\U0001f4c5 Quarters", "\U0001f4ca Calendar Years"])
    with tab_q:
        fig_nq = ch_notable_q(nq)
        if fig_nq: st.plotly_chart(fig_nq, width='stretch')
        if not nq.empty:
            for _, row in nq.sort_values("start").iterrows():
                icon = "\U0001f7e2" if row["label"]=="Best" else "\U0001f534"
                st.markdown(f'<div class="ncard">{icon} <strong>{row["period"]}</strong> \u2014 '
                            f'<strong>{row["excess_return"]:+.1%}</strong><br>'
                            f'<span style="color:{C["neut"]};font-size:0.85rem;padding-left:24px;">'
                            f'Factor drivers: {row["factor_drivers"]}</span></div>', unsafe_allow_html=True)
    with tab_y:
        fig_ny = ch_notable_y(ny)
        if fig_ny: st.plotly_chart(fig_ny, width='stretch')
        if not ny.empty:
            for _, row in ny.sort_values("period").iterrows():
                icon = "\U0001f7e2" if row["label"]=="Best" else "\U0001f534"
                st.markdown(f'<div class="ncard">{icon} <strong>{row["period"]}</strong> \u2014 '
                            f'<strong>{row["excess_return"]:+.1%}</strong> ({row["months"]}mo)<br>'
                            f'<span style="color:{C["neut"]};font-size:0.85rem;padding-left:24px;">'
                            f'Factor drivers: {row["factor_drivers"]}</span></div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # REGIME ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.header("Factor Regime Analysis")
    st.markdown(f'<div class="rleg"><strong>How to read:</strong> Monthly factor returns ranked into quartiles. '
                f'<strong>Q1 (Bottom) = Out-of-Favor</strong> (weakest factor returns). '
                f'<strong>Q4 (Top) = In-Favor</strong> (factor most rewarded). '
                f'For Growth vs Value: Q4 = Growth-led, Q1 = Value-led. Values in <strong>bps/month</strong>.</div>',
                unsafe_allow_html=True)

    rc1, rc2 = st.columns([1, 3])
    with rc1:
        hm_metric = st.radio("Metric", ["avg_excess", "hit_rate", "median_excess"])
    with rc2:
        fig_hm = ch_heatmap(rt, hm_metric)
        if fig_hm: st.plotly_chart(fig_hm, width='stretch')

    if not sp.empty:
        st.subheader("Spread Summary (In-Favor minus Out-of-Favor)")
        spd = sp.copy()
        spd["Factor"] = spd["factor"].apply(fl)
        spd["Spread (bps/mo)"] = (spd["spread"]*10000).round(1)
        spd["t-stat"] = spd["t_stat"].round(2)
        spd["Sig?"] = spd["significant"].apply(lambda x: "\u2705" if x else "")
        spd["Interpretation"] = spd["interpretation"]
        st.dataframe(spd[["Factor","Spread (bps/mo)","t-stat","Sig?","Interpretation"]],
                     width='stretch', hide_index=True)

    with st.expander("\U0001f4ca Per-Factor Bar Charts (95% CI)"):
        fnames = rt["factor"].unique() if not rt.empty else []
        cols = st.columns(min(3, max(1, len(fnames))))
        for i, fn in enumerate(fnames):
            with cols[i % len(cols)]:
                fb = ch_bar(rt, fn)
                if fb: st.plotly_chart(fb, width='stretch', key=f"bar_{fn}")

    # ══════════════════════════════════════════════════════════════════════════
    # REGRESSIONS
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.header("Factor Exposure (Regressions)")

    st.subheader("Single-Factor: Impact per 1-SD Factor Move")
    fig_sf = ch_sf(sf_tbl)
    if fig_sf: st.plotly_chart(fig_sf, width='stretch')

    with st.expander("\U0001f4cb Single-Factor Table"):
        sfd = sf_tbl.copy()
        sfd["Factor"] = sfd["factor"].apply(fl)
        sfd["Impact (bps)"] = (sfd["impact"]*10000).round(1)
        sfd["Sig"] = sfd["p"].apply(lambda p: "\u2705" if p<0.05 else "")
        for c in ["alpha","beta","std_beta","t","p","r2"]:
            sfd[c] = sfd[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
        st.dataframe(sfd[["Factor","beta","std_beta","Impact (bps)","t","p","r2","Sig"]],
                     width='stretch', hide_index=True)

    if HAS_SM:
        st.subheader("Multi-Factor Regression")
        fig_mf = ch_mf(mf_tbl)
        if fig_mf: st.plotly_chart(fig_mf, width='stretch')
        if not mf_tbl.empty:
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("R\u00b2", f"{mf_tbl.attrs.get('r2',0):.4f}")
            mc2.metric("Adj. R\u00b2", f"{mf_tbl.attrs.get('adj_r2',0):.4f}")
            mc3.metric("F-stat", f"{mf_tbl.attrs.get('f',0):.2f}")
            mc4.metric("Obs", mf_tbl.attrs.get("n", 0))

    # ══════════════════════════════════════════════════════════════════════════
    # STABILITY
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.header("Exposure Stability")
    tab_r, tab_s = st.tabs(["\U0001f4c8 Rolling 36-Mo Betas", "\U0001f4ca Subperiod Check"])
    with tab_r:
        r36 = rolling_betas(dates, excess, factors, 36)
        fr36 = ch_rolling(r36, 36)
        if fr36: st.plotly_chart(fr36, width='stretch')
    with tab_s:
        st.markdown("*Do betas hold across regimes, or are they period-specific?*")
        spb = subperiod_betas(excess, factors, dates)
        if not spb.empty:
            spb["Factor"] = spb["factor"].apply(fl)
            pivot = spb.pivot_table(index="Factor", columns="period", values="beta", aggfunc="first")
            st.dataframe(pivot.style.format("{:+.3f}", na_rep="\u2014").background_gradient(
                cmap="RdBu_r", vmin=-1, vmax=1, axis=None), width='stretch')
            st.caption("Sign consistency across periods = stable exposure. Sign flips = period-specific, less trustworthy.")

    with st.expander("\U0001f517 Factor Correlation Matrix"):
        fc_fig = ch_corr(factors)
        if fc_fig: st.plotly_chart(fc_fig, width='stretch')

    # ══════════════════════════════════════════════════════════════════════════
    # EXPORT
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.header("Export")
    ex1, ex2 = st.columns(2)
    with ex1:
        if st.button("\U0001f4c4 Generate Excel", use_container_width=True):
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as w:
                pd.DataFrame({"Summary": _summary_to_bullets(summary)}).to_excel(w, sheet_name="Summary", index=False)
                if not rt.empty:
                    rto = rt.copy(); rto["factor"] = rto["factor"].apply(fl)
                    rto.to_excel(w, sheet_name="Regime Analysis", index=False)
                    spo = sp.copy(); spo["factor"] = spo["factor"].apply(fl)
                    spo.to_excel(w, sheet_name="Spread Summary", index=False)
                sfo = sf_tbl.copy(); sfo["factor"] = sfo["factor"].apply(fl)
                sfo.to_excel(w, sheet_name="Single Factor", index=False)
                if not mf_tbl.empty: mf_tbl.to_excel(w, sheet_name="Multi Factor", index=False)
                if not nq.empty: nq.to_excel(w, sheet_name="Notable Quarters", index=False)
                if not ny.empty: ny.to_excel(w, sheet_name="Notable Years", index=False)
            buf.seek(0)
            cn = sel_strat.replace("|","-").replace("/","-").strip()
            st.download_button("\u2b07\ufe0f Download Excel", buf, f"factor_{cn}.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               use_container_width=True)
    with ex2:
        if st.button("\U0001f4d1 Generate PDF Report", use_container_width=True):
            try:
                bm = meta.get("benchmark", "N/A")
                _pdf_hm = ch_heatmap(rt, "avg_excess") if not rt.empty else None
                _pdf_nq = ch_notable_q(nq)
                _pdf_mf = ch_mf(mf_tbl) if not mf_tbl.empty else None
                _pdf_ny = ch_notable_y(ny) if not ny.empty else None
                _pdf_r36 = ch_rolling(rolling_betas(dates, excess, factors, 36), 36) if n_mo >= 36 else None
                pdf_bytes = build_strategy_pdf(
                    sel_strat, bm, window_label, as_of_str, summary,
                    sf_tbl, mf_tbl, rt, sp, nq, ny, fig_cum, fig_sf,
                    _pdf_hm, _pdf_nq,
                    fig_mf=_pdf_mf, fig_roll=_pdf_r36, fig_ny=_pdf_ny)
                if pdf_bytes:
                    cn = sel_strat.replace("|","-").replace("/","-").strip()
                    st.download_button("\u2b07\ufe0f Download PDF", pdf_bytes,
                                       f"factor_report_{cn}.pdf", "application/pdf",
                                       use_container_width=True)
                else:
                    st.error("PDF generation requires fpdf2: `pip install fpdf2`")
            except Exception as e:
                st.error(f"PDF generation error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PEER COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif "\U0001f50d" in page:
    st.header("Benchmark Peer Comparison")
    st.markdown("*Compare factor exposures across all strategies sharing the same benchmark.*")

    # Benchmark selector
    bm_counts = sdf_w.groupby("benchmark")["strategy"].nunique().sort_values(ascending=False)
    bm_options = [f"{bm} ({n} strategies)" for bm, n in bm_counts.items() if n >= 2]
    if not bm_options:
        st.warning("No benchmarks with 2+ strategies in current window."); st.stop()
    sel_bm_display = st.selectbox("Select benchmark", bm_options)
    sel_bm = sel_bm_display.split(" (")[0]

    with st.spinner("Computing peer factor exposures..."):
        peer_df = compute_peer_table(sdf_w, fdf_w, sel_bm, window_label)

    if peer_df.empty:
        st.warning("Not enough overlapping data for peer comparison."); st.stop()

    st.markdown(f"**{len(peer_df)} strategies** benchmarked to {sel_bm}")

    # ── View toggle ───────────────────────────────────────────────────────────
    view_type = st.radio("Beta type", ["Multi-Factor Betas", "Standardized Betas (per 1-SD)"], horizontal=True)

    # ── Build display table ───────────────────────────────────────────────────
    fcols = [c for c in factor_df.columns if c != "date"]
    disp = peer_df[["strategy", "months", "r2"]].copy()
    disp["strategy"] = disp["strategy"].apply(_abbrev_strategy)
    disp["R\u00b2"] = disp["r2"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    disp["Unexplained"] = disp["r2"].map(lambda x: f"{(1-x)*100:.0f}%" if pd.notna(x) else "")

    if "Multi" in view_type:
        for fc in fcols:
            col = f"mf_{fc}"
            if col in peer_df.columns:
                disp[fl(fc)] = peer_df[col].map(lambda x: f"{x:+.3f}" if pd.notna(x) else "")
    else:
        for fc in fcols:
            col = f"std_{fc}"
            if col in peer_df.columns:
                disp[fl(fc)] = peer_df[col].map(lambda x: f"{x:+.3f}" if pd.notna(x) else "")

    if "alpha_bps" in peer_df.columns:
        disp["Alpha (bps)"] = peer_df["alpha_bps"].map(lambda x: f"{x:+.1f}" if pd.notna(x) else "")

    disp = disp.drop(columns=["r2"])
    disp = disp.rename(columns={"strategy": "Strategy", "months": "Months"})

    # ── Sort control ──────────────────────────────────────────────────────────
    sort_options = ["R\u00b2"] + [fl(fc) for fc in fcols if fl(fc) in disp.columns]
    sort_col = st.selectbox("Sort by", sort_options)
    asc = st.checkbox("Ascending", value=False)

    # Convert sort column back to numeric for sorting
    if sort_col in disp.columns:
        disp["_sort"] = pd.to_numeric(disp[sort_col].str.replace("%", ""), errors="coerce")
        disp = disp.sort_values("_sort", ascending=asc).drop(columns="_sort")

    st.dataframe(disp, width='stretch', hide_index=True, height=min(800, 40 + len(disp) * 35))

    # ── Summary insights ──────────────────────────────────────────────────────
    if not peer_df.empty and "r2" in peer_df.columns:
        st.subheader("Peer Insights")
        avg_r2 = peer_df["r2"].mean() * 100
        most_exp = peer_df.loc[peer_df["r2"].idxmax(), "strategy"] if len(peer_df) else ""
        least_exp = peer_df.loc[peer_df["r2"].idxmin(), "strategy"] if len(peer_df) else ""

        ic1, ic2, ic3 = st.columns(3)
        ic1.metric("Avg R\u00b2", f"{avg_r2:.1f}%")
        ic2.metric("Most Factor-Driven", _abbrev_strategy(most_exp))
        ic3.metric("Most Idiosyncratic", _abbrev_strategy(least_exp))

    st.markdown(f'<div class="ctx"><strong>Methodology:</strong> Multi-factor OLS regression of monthly excess returns on '
                f'{len(fcols)} factors. Window: {window_label}. Betas shown are {"raw multi-factor coefficients" if "Multi" in view_type else "standardized (per 1-SD factor move)"}. '
                f'R\u00b2 indicates proportion of excess return variation explained by factors.</div>',
                unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # EXCESS RETURN CORRELATION
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.header("Excess Return Correlations")
    st.markdown("*How similarly do these strategies behave? Lower correlation = more diversification potential.*")

    with st.spinner("Computing pairwise correlations..."):
        corr_matrix = compute_peer_corr(sdf_w, sel_bm, window_label)

    if not corr_matrix.empty:
        fig_pcorr = ch_peer_corr(corr_matrix)
        if fig_pcorr:
            st.plotly_chart(fig_pcorr, width='stretch')

        # Correlation summary stats
        mask = np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
        upper_vals = corr_matrix.values[mask]
        upper_vals = upper_vals[~np.isnan(upper_vals)]
        if len(upper_vals) > 0:
            cs1, cs2, cs3, cs4 = st.columns(4)
            cs1.metric("Avg Pairwise Correlation", f"{np.mean(upper_vals):.2f}")
            cs2.metric("Median", f"{np.median(upper_vals):.2f}")
            cs3.metric("Min (Most Diversifying)", f"{np.min(upper_vals):.2f}")
            cs4.metric("Max (Most Similar)", f"{np.max(upper_vals):.2f}")

        # ══════════════════════════════════════════════════════════════════════
        # PAIRING RECOMMENDATIONS
        # ══════════════════════════════════════════════════════════════════════
        st.markdown("---")
        st.header("Recommended Pairings")
        st.markdown("*Strategy pairs with low excess return correlation and complementary factor exposures — combining them may reduce factor-driven drawdowns.*")

        pairings = find_pairings(corr_matrix, peer_df, fcols)

        if pairings:
            pair_html = '<div class="pair-section">'
            for idx, p in enumerate(pairings, 1):
                s1_short = _abbrev_strategy(p["s1"])
                s2_short = _abbrev_strategy(p["s2"])
                rho = p["correlation"]
                corr_cls = "corr-neg" if rho < 0.2 else ("corr-low" if rho < 0.5 else "corr-med")

                pair_html += f'<div class="pair-card">'

                # ── Header: names + correlation ──────────────────────────────
                pair_html += f'<div class="pair-header">'
                pair_html += (
                    f'<div class="pair-names" style="display:flex;align-items:flex-start;">'
                    f'  <span class="pair-rank">{idx}</span>'
                    f'  <div>'
                    f'    <div class="pair-label">Recommended Pair</div>'
                    f'    <div class="pair-strat">{s1_short}</div>'
                    f'    <div class="pair-strat" style="color:{C["neut"]};">+ {s2_short}</div>'
                    f'  </div>'
                    f'</div>'
                )
                pair_html += (
                    f'<div class="pair-corr">'
                    f'  <div class="corr-label">Excess Correlation</div>'
                    f'  <div class="corr-value {corr_cls}">{rho:.2f}</div>'
                    f'</div>'
                )
                pair_html += '</div>'  # pair-header

                # ── Factor tilt rows ─────────────────────────────────────────
                all_exposures = p["opposing"] + p.get("different_magnitude", [])
                # Sort by magnitude of difference
                all_exposures.sort(key=lambda d: abs(d["s1_beta"] - d["s2_beta"]), reverse=True)
                if all_exposures:
                    pair_html += '<div class="pair-tilts">'
                    for exp in all_exposures[:4]:
                        s1_tilt = _tilt_label(exp["raw_factor"], exp["s1_beta"])
                        s2_tilt = _tilt_label(exp["raw_factor"], exp["s2_beta"])
                        s1_cls = "tilt-pos" if exp["s1_beta"] > 0 else "tilt-neg"
                        s2_cls = "tilt-pos" if exp["s2_beta"] > 0 else "tilt-neg"
                        opposing = exp["s1_beta"] * exp["s2_beta"] < 0
                        pair_html += (
                            f'<div class="tilt-row">'
                            f'  <div class="tilt-factor-name">{exp["factor"]}</div>'
                            f'  <div class="tilt-strats">'
                            f'    <div class="tilt-strat">'
                            f'      <div class="tilt-strat-name">{s1_short}</div>'
                            f'      <div class="tilt-strat-label {s1_cls}">{s1_tilt}</div>'
                            f'      <div class="tilt-strat-beta">Beta: {exp["s1_beta"]:+.3f}</div>'
                            f'    </div>'
                            f'    <div class="tilt-strat">'
                            f'      <div class="tilt-strat-name">{s2_short}</div>'
                            f'      <div class="tilt-strat-label {s2_cls}">{s2_tilt}</div>'
                            f'      <div class="tilt-strat-beta">Beta: {exp["s2_beta"]:+.3f}</div>'
                            f'    </div>'
                            f'  </div>'
                            f'</div>'
                        )
                    pair_html += '</div>'

                # ── Rationale chip ───────────────────────────────────────────
                if rho < 0:
                    rationale = "Negative correlation — these strategies tend to move in opposite directions, providing strong diversification."
                elif rho < 0.3:
                    rationale = "Very low correlation — combining these strategies should meaningfully reduce portfolio-level drawdowns."
                elif rho < 0.5:
                    rationale = "Low correlation — moderate diversification benefit when paired together."
                else:
                    rationale = "Different factor profiles provide some diversification despite moderate correlation."
                pair_html += (
                    f'<div class="pair-rationale">'
                    f'  <div class="pair-chip" style="flex:1;">'
                    f'    <div class="chip-label">Why This Pair</div>'
                    f'    <div>{rationale}</div>'
                    f'  </div>'
                    f'</div>'
                )

                # ── Blended 50/50 performance ────────────────────────────────
                bstats = compute_blend_stats(sdf_w, p["s1"], p["s2"], window_label)
                if bstats:
                    def _best(vals, higher_better=True):
                        """Return tuple of CSS classes for (s1, s2, blend) — highlight the best."""
                        classes = ["", "", ""]
                        if higher_better:
                            bi = int(np.argmax(vals))
                        else:
                            bi = int(np.argmin([abs(v) for v in vals]))
                        classes[bi] = " blend-best"
                        return classes

                    ret_cls = _best(bstats["ann_ret"], True)
                    vol_cls = _best(bstats["ann_vol"], False)
                    dd_cls = _best(bstats["max_dd"], False)
                    hit_cls = _best(bstats["hit_rate"], True)

                    pair_html += (
                        f'<div class="blend-section">'
                        f'  <div class="blend-title">Historical 50/50 Blend ({bstats["months"]} months)</div>'
                        f'  <div class="blend-grid">'
                        # Header row
                        f'    <div class="blend-hdr"></div>'
                        f'    <div class="blend-hdr" style="text-align:right;">{s1_short}</div>'
                        f'    <div class="blend-hdr" style="text-align:right;">{s2_short}</div>'
                        f'    <div class="blend-hdr blend-highlight" style="text-align:right;">50/50 Blend</div>'
                        # Ann. excess return
                        f'    <div class="blend-metric">Ann. Excess Return</div>'
                        f'    <div class="blend-val{ret_cls[0]}">{bstats["ann_ret"][0]:+.2%}</div>'
                        f'    <div class="blend-val{ret_cls[1]}">{bstats["ann_ret"][1]:+.2%}</div>'
                        f'    <div class="blend-val blend-highlight{ret_cls[2]}">{bstats["ann_ret"][2]:+.2%}</div>'
                        # Ann. volatility
                        f'    <div class="blend-metric">Ann. Volatility</div>'
                        f'    <div class="blend-val{vol_cls[0]}">{bstats["ann_vol"][0]:.2%}</div>'
                        f'    <div class="blend-val{vol_cls[1]}">{bstats["ann_vol"][1]:.2%}</div>'
                        f'    <div class="blend-val blend-highlight{vol_cls[2]}">{bstats["ann_vol"][2]:.2%}</div>'
                        # Max drawdown
                        f'    <div class="blend-metric">Max Drawdown</div>'
                        f'    <div class="blend-val{dd_cls[0]}">{bstats["max_dd"][0]:.1%}</div>'
                        f'    <div class="blend-val{dd_cls[1]}">{bstats["max_dd"][1]:.1%}</div>'
                        f'    <div class="blend-val blend-highlight{dd_cls[2]}">{bstats["max_dd"][2]:.1%}</div>'
                        # Hit rate
                        f'    <div class="blend-metric">Monthly Hit Rate</div>'
                        f'    <div class="blend-val{hit_cls[0]}">{bstats["hit_rate"][0]:.0f}%</div>'
                        f'    <div class="blend-val{hit_cls[1]}">{bstats["hit_rate"][1]:.0f}%</div>'
                        f'    <div class="blend-val blend-highlight{hit_cls[2]}">{bstats["hit_rate"][2]:.0f}%</div>'
                        f'  </div>'
                        f'</div>'
                    )

                pair_html += '</div>'  # pair-card

            pair_html += '</div>'
            st.markdown(pair_html, unsafe_allow_html=True)
        else:
            st.info("Not enough strategy overlap to generate pairing recommendations.")

    else:
        st.info("Not enough overlapping data to compute correlations (need 2+ strategies with 24+ months).")

    # ══════════════════════════════════════════════════════════════════════════
    # CUSTOM PAIR BUILDER
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.header("Custom Pair Builder")
    st.markdown("*Select any two strategies to compare factor profiles and simulated blend performance.*")

    peer_strats = sorted(peer_df["strategy"].unique()) if not peer_df.empty else []
    if len(peer_strats) >= 2:
        cb1, cb2 = st.columns(2)
        with cb1:
            cust_s1 = st.selectbox("Strategy A", peer_strats, index=0, key="cust_s1")

        # Compute top 3 recommended partners for Strategy A using pairing logic
        rec_partners = get_recommended_partners(cust_s1, corr_matrix, peer_df, fcols, n=3)
        # Build Strategy B list: recommended partners first, then remaining (excluding A)
        other_strats = [s for s in peer_strats if s not in rec_partners and s != cust_s1]
        s2_options = rec_partners + other_strats
        rec_rank = {s: i + 1 for i, s in enumerate(rec_partners)}

        def _fmt_s2(s, _rr=rec_rank):
            if s in _rr:
                return f"\u2605 Recommended Pairing #{_rr[s]}: {_abbrev_strategy(s)}"
            return _abbrev_strategy(s)

        with cb2:
            cust_s2 = st.selectbox("Strategy B", s2_options, index=0,
                                   format_func=_fmt_s2, key="cust_s2")

        if cust_s1 == cust_s2:
            st.warning("Select two different strategies to compare.")
        else:
            cs1_short = _abbrev_strategy(cust_s1)
            cs2_short = _abbrev_strategy(cust_s2)

            # Correlation
            cust_rho = np.nan
            if not corr_matrix.empty and cust_s1 in corr_matrix.columns and cust_s2 in corr_matrix.columns:
                cust_rho = corr_matrix.loc[cust_s1, cust_s2]

            # Factor exposures
            cr1 = peer_df[peer_df["strategy"]==cust_s1]
            cr2 = peer_df[peer_df["strategy"]==cust_s2]

            cust_exposures = []
            if not cr1.empty and not cr2.empty:
                cr1, cr2 = cr1.iloc[0], cr2.iloc[0]
                for fc in fcols:
                    mf_col = f"mf_{fc}"
                    v1 = cr1.get(mf_col, np.nan)
                    v2 = cr2.get(mf_col, np.nan)
                    if pd.notna(v1) and pd.notna(v2) and abs(v1 - v2) > 0.005:
                        cust_exposures.append({"factor": fl(fc), "raw_factor": fc,
                                               "s1_beta": v1, "s2_beta": v2})
                cust_exposures.sort(key=lambda d: abs(d["s1_beta"] - d["s2_beta"]), reverse=True)

            # Blended stats
            cust_bstats = compute_blend_stats(sdf_w, cust_s1, cust_s2, window_label)

            # ── Render card ──────────────────────────────────────────────────
            corr_cls = "corr-neg" if cust_rho < 0.2 else ("corr-low" if cust_rho < 0.5 else "corr-med")
            cust_html = '<div class="pair-card">'

            # Header
            cust_html += '<div class="pair-header">'
            cust_html += (
                f'<div class="pair-names">'
                f'  <div>'
                f'    <div class="pair-label">Custom Pair Analysis</div>'
                f'    <div class="pair-strat">{cs1_short}</div>'
                f'    <div class="pair-strat" style="color:{C["neut"]};">+ {cs2_short}</div>'
                f'  </div>'
                f'</div>'
            )
            if not np.isnan(cust_rho):
                cust_html += (
                    f'<div class="pair-corr">'
                    f'  <div class="corr-label">Excess Correlation</div>'
                    f'  <div class="corr-value {corr_cls}">{cust_rho:.2f}</div>'
                    f'</div>'
                )
            cust_html += '</div>'  # pair-header

            # Factor tilts
            if cust_exposures:
                cust_html += '<div class="pair-tilts">'
                for exp in cust_exposures[:6]:
                    s1_tilt = _tilt_label(exp["raw_factor"], exp["s1_beta"])
                    s2_tilt = _tilt_label(exp["raw_factor"], exp["s2_beta"])
                    s1_cls = "tilt-pos" if exp["s1_beta"] > 0 else "tilt-neg"
                    s2_cls = "tilt-pos" if exp["s2_beta"] > 0 else "tilt-neg"
                    cust_html += (
                        f'<div class="tilt-row">'
                        f'  <div class="tilt-factor-name">{exp["factor"]}</div>'
                        f'  <div class="tilt-strats">'
                        f'    <div class="tilt-strat">'
                        f'      <div class="tilt-strat-name">{cs1_short}</div>'
                        f'      <div class="tilt-strat-label {s1_cls}">{s1_tilt}</div>'
                        f'      <div class="tilt-strat-beta">Beta: {exp["s1_beta"]:+.3f}</div>'
                        f'    </div>'
                        f'    <div class="tilt-strat">'
                        f'      <div class="tilt-strat-name">{cs2_short}</div>'
                        f'      <div class="tilt-strat-label {s2_cls}">{s2_tilt}</div>'
                        f'      <div class="tilt-strat-beta">Beta: {exp["s2_beta"]:+.3f}</div>'
                        f'    </div>'
                        f'  </div>'
                        f'</div>'
                    )
                cust_html += '</div>'

            # Blended performance
            if cust_bstats:
                def _best_c(vals, higher_better=True):
                    classes = ["", "", ""]
                    if higher_better:
                        bi = int(np.argmax(vals))
                    else:
                        bi = int(np.argmin([abs(v) for v in vals]))
                    classes[bi] = " blend-best"
                    return classes

                ret_cls = _best_c(cust_bstats["ann_ret"], True)
                vol_cls = _best_c(cust_bstats["ann_vol"], False)
                dd_cls = _best_c(cust_bstats["max_dd"], False)
                hit_cls = _best_c(cust_bstats["hit_rate"], True)

                cust_html += (
                    f'<div class="blend-section">'
                    f'  <div class="blend-title">Historical 50/50 Blend ({cust_bstats["months"]} months)</div>'
                    f'  <div class="blend-grid">'
                    f'    <div class="blend-hdr"></div>'
                    f'    <div class="blend-hdr" style="text-align:right;">{cs1_short}</div>'
                    f'    <div class="blend-hdr" style="text-align:right;">{cs2_short}</div>'
                    f'    <div class="blend-hdr blend-highlight" style="text-align:right;">50/50 Blend</div>'
                    f'    <div class="blend-metric">Ann. Excess Return</div>'
                    f'    <div class="blend-val{ret_cls[0]}">{cust_bstats["ann_ret"][0]:+.2%}</div>'
                    f'    <div class="blend-val{ret_cls[1]}">{cust_bstats["ann_ret"][1]:+.2%}</div>'
                    f'    <div class="blend-val blend-highlight{ret_cls[2]}">{cust_bstats["ann_ret"][2]:+.2%}</div>'
                    f'    <div class="blend-metric">Ann. Volatility</div>'
                    f'    <div class="blend-val{vol_cls[0]}">{cust_bstats["ann_vol"][0]:.2%}</div>'
                    f'    <div class="blend-val{vol_cls[1]}">{cust_bstats["ann_vol"][1]:.2%}</div>'
                    f'    <div class="blend-val blend-highlight{vol_cls[2]}">{cust_bstats["ann_vol"][2]:.2%}</div>'
                    f'    <div class="blend-metric">Max Drawdown</div>'
                    f'    <div class="blend-val{dd_cls[0]}">{cust_bstats["max_dd"][0]:.1%}</div>'
                    f'    <div class="blend-val{dd_cls[1]}">{cust_bstats["max_dd"][1]:.1%}</div>'
                    f'    <div class="blend-val blend-highlight{dd_cls[2]}">{cust_bstats["max_dd"][2]:.1%}</div>'
                    f'    <div class="blend-metric">Monthly Hit Rate</div>'
                    f'    <div class="blend-val{hit_cls[0]}">{cust_bstats["hit_rate"][0]:.0f}%</div>'
                    f'    <div class="blend-val{hit_cls[1]}">{cust_bstats["hit_rate"][1]:.0f}%</div>'
                    f'    <div class="blend-val blend-highlight{hit_cls[2]}">{cust_bstats["hit_rate"][2]:.0f}%</div>'
                    f'  </div>'
                    f'</div>'
                )
            else:
                cust_html += f'<div class="ctx" style="margin-top:12px;">Not enough overlapping months to compute blended performance.</div>'

            cust_html += '</div>'  # pair-card
            st.markdown(cust_html, unsafe_allow_html=True)
    else:
        st.info("Need at least 2 strategies in the peer group to build custom pairs.")

st.markdown("---")
st.caption("Manager Research Dashboard")