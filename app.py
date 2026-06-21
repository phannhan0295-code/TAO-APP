import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import ttest_1samp, wilcoxon
import warnings
import os

warnings.filterwarnings("ignore")

# Configuration and Page Setup
st.set_page_config(
    page_title="Tối ưu hóa Danh mục Đầu tư - HOSE",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium CSS Styling
st.markdown("""
    <style>
        /* General Styles */
        .main {
            background-color: #0f1116;
            color: #e2e8f0;
        }
        h1, h2, h3 {
            font-family: 'Outfit', 'Inter', sans-serif;
            font-weight: 700;
        }
        
        /* Premium Header Banner */
        .header-container {
            background: linear-gradient(135deg, #1e3a8a 0%, #0f172a 100%);
            padding: 2.5rem;
            border-radius: 16px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
            margin-bottom: 2rem;
            border: 1px solid #1e40af;
        }
        .header-title {
            color: #ffffff;
            font-size: 2.2rem;
            margin: 0;
            font-weight: 800;
        }
        .header-subtitle {
            color: #93c5fd;
            font-size: 1.1rem;
            margin-top: 0.5rem;
            font-weight: 400;
        }
        
        /* Metric Card Styling */
        .metric-card {
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            text-align: center;
            transition: all 0.3s ease;
        }
        .metric-card:hover {
            transform: translateY(-5px);
            border-color: #3b82f6;
            box-shadow: 0 10px 15px -3px rgba(59, 130, 246, 0.2);
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: 800;
            color: #3b82f6;
            margin-bottom: 0.25rem;
        }
        .metric-value-green {
            color: #10b981;
        }
        .metric-value-red {
            color: #f43f5e;
        }
        .metric-label {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #94a3b8;
            font-weight: 600;
        }
        
        /* Info Callout */
        .info-callout {
            background: rgba(30, 41, 59, 0.5);
            border-left: 4px solid #3b82f6;
            padding: 1rem;
            border-radius: 0 8px 8px 0;
            margin-bottom: 1.5rem;
        }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------------------
# CONSTANTS & STRATEGY FUNCTIONS
# -----------------------------------------------------------------------------------------
VNINDEX = "vnindex"
SO_NGAY_TRONG_NAM = 252

def tinh_macd(close, nhanh, cham, tin_hieu):
    """MACD = EMA(nhanh) - EMA(cham); Signal Line = EMA(tin_hieu) of MACD."""
    ema_nhanh = close.ewm(span=nhanh, adjust=False).mean()
    ema_cham  = close.ewm(span=cham,  adjust=False).mean()
    macd      = ema_nhanh - ema_cham
    duong_tin_hieu = macd.ewm(span=tin_hieu, adjust=False).mean()
    return macd, duong_tin_hieu

def duong_trung_binh_dong(df, cua_so):
    """Simple Moving Average (SMA)."""
    return df.rolling(window=cua_so, min_periods=cua_so).mean()

def cat_len(a, b):
    """a crosses UP b: a > b today but a <= b yesterday."""
    return (a > b) & (a.shift(1) <= b.shift(1))

def cat_xuong(a, b):
    """a crosses DOWN b: a < b today but a >= b yesterday."""
    return (a < b) & (a.shift(1) >= b.shift(1))

def trang_thai_nam_giu(close, nhanh, cham, tin_hieu):
    """Generate MACD states: 1 = Hold, 0 = Cash."""
    macd, duong_tin_hieu = tinh_macd(close, nhanh, cham, tin_hieu)
    mua  = cat_len(macd, duong_tin_hieu)
    ban  = cat_xuong(macd, duong_tin_hieu)
    trang_thai = pd.Series(np.nan, index=close.index)
    trang_thai[mua.fillna(False)] = 1.0
    trang_thai[ban.fillna(False)] = 0.0
    return trang_thai.ffill().fillna(0.0)

def sharpe_chien_luoc_mot_ma(close, nhanh, cham, tin_hieu, chi_tiet=False):
    """Run MACD strategy backtest on a single stock and return Sharpe ratio or metrics."""
    st = trang_thai_nam_giu(close, nhanh, cham, tin_hieu)
    loi_nhuan_ngay = close.pct_change().fillna(0.0)
    loi_nhuan_cl   = st.shift(1).fillna(0.0) * loi_nhuan_ngay
    if loi_nhuan_cl.std() == 0 or len(loi_nhuan_cl.dropna()) < 20:
        return (-np.inf if not chi_tiet else {"sharpe": -np.inf})
    sharpe = np.sqrt(SO_NGAY_TRONG_NAM) * loi_nhuan_cl.mean() / loi_nhuan_cl.std()
    if not chi_tiet:
        return sharpe
    macd, sig = tinh_macd(close, nhanh, cham, tin_hieu)
    return {
        "so_phien": len(close),
        "so_ngay_nam_giu": int((st == 1.0).sum()),
        "ty_le_nam_giu_%": float((st == 1.0).mean() * 100),
        "so_lan_mua": int(cat_len(macd, sig).fillna(False).sum()),
        "so_lan_ban": int(cat_xuong(macd, sig).fillna(False).sum()),
        "loi_nhuan_cl_tb_ngay_%": float(loi_nhuan_cl.mean() * 100),
        "bien_dong_cl_ngay_%": float(loi_nhuan_cl.std() * 100),
        "sharpe": float(sharpe),
    }

def chon_top_ma(ngay, adj_close, volume, ma_loc, so_ma, cua_so, min_lich_su, max_ngay_kl_0):
    """Select top performing assets based on historical MACD Sharpe and filters."""
    try:
        i = adj_close.index.get_loc(ngay)
    except KeyError:
        # Fallback to closest available date index
        i = adj_close.index.get_indexer([ngay], method="pad")[0]
        if i == -1:
            return [], {}
            
    diem = []
    for ma in adj_close.columns:
        # --- Filter 1: History length ---
        chuoi = adj_close[ma].iloc[:i + 1].dropna()
        if len(chuoi) < min_lich_su:
            continue
        # --- Filter 3: Price above trend SMA ---
        if pd.isna(ma_loc.loc[ngay, ma]) or pd.isna(adj_close.loc[ngay, ma]) \
           or adj_close.loc[ngay, ma] <= ma_loc.loc[ngay, ma]:
            continue
        # --- Filter 2: Liquidity check ---
        vol_cua_so = volume[ma].iloc[max(0, i - 126 + 1):i + 1]
        if int((vol_cua_so.fillna(0) == 0).sum()) > max_ngay_kl_0:
            continue
        # --- Score: Sharpe Ratio of MACD over the lookback window ---
        cua = adj_close[ma].iloc[max(0, i - cua_so + 1):i + 1]
        s = sharpe_chien_luoc_mot_ma(cua, 12, 26, 9) # defaults match global kinh dien
        if np.isfinite(s):
            diem.append((ma, s))
            
    diem.sort(key=lambda x: (-x[1], x[0]))
    top = diem[:so_ma]
    return [m for m, _ in top], {m: s for m, s in top}

def trong_so_danh_muc(ds_ma, diem_map, ngay, adj_close, kieu, lookback=126, tran=0.40, san=0.08):
    """Determine asset weights based on the selected method, applying caps and floors."""
    if len(ds_ma) == 0:
        return {}
    try:
        i = adj_close.index.get_loc(ngay)
    except KeyError:
        i = adj_close.index.get_indexer([ngay], method="pad")[0]
        if i == -1:
            return {m: 1.0 / len(ds_ma) for m in ds_ma}
            
    if kieu == "equal":
        w = {m: 1.0 / len(ds_ma) for m in ds_ma}
    elif kieu == "score":
        sc = {m: max(diem_map.get(m, 0.0), 0.01) for m in ds_ma}
        tong = sum(sc.values())
        w = {m: sc[m] / tong if tong > 0 else 1.0 / len(ds_ma) for m in ds_ma}
    elif kieu in ("inverse_vol", "vol_target"):
        loi_nhuan_ngay = adj_close.pct_change()
        thanh_phan = {}
        for m in ds_ma:
            v = loi_nhuan_ngay[m].iloc[max(0, i - lookback + 1):i + 1].dropna().std()
            if v is not None and v > 0:
                thanh_phan[m] = (1.0 / v) if kieu == "inverse_vol" else (1.0 / (v * v))
            else:
                thanh_phan[m] = 0.0
        tong = sum(thanh_phan.values())
        w = {m: (thanh_phan[m] / tong if tong > 0 else 1.0 / len(ds_ma)) for m in ds_ma}
    else:
        w = {m: 1.0 / len(ds_ma) for m in ds_ma}

    # Apply limits and re-normalize
    w = {m: min(max(x, san), tran) for m, x in w.items()}
    tong = sum(w.values())
    return {m: x / tong for m, x in w.items()} if tong > 0 else {m: 1.0 / len(ds_ma) for m in ds_ma}

def lay_ngay_tai_can_bang(index, ngay_bat_dau, ngay_ket_thuc):
    """Get the first trading day of each quarter in the date range."""
    idx = index[(index >= pd.to_datetime(ngay_bat_dau)) & (index <= pd.to_datetime(ngay_ket_thuc))]
    s = pd.DataFrame({"date": idx})
    s["quy"] = s["date"].dt.to_period("Q")
    return sorted(s.groupby("quy")["date"].min().tolist())

# Cache the load data to prevent reloading files repeatedly
@st.cache_data
def doc_du_lieu_tu_file(uploaded_file):
    """Load data from uploaded CSV, normalize column headers and scale asset prices."""
    df = pd.read_csv(uploaded_file, encoding="utf-8-sig", low_memory=False)
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Flexible date parsing
    date_col = df["date"].copy()
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%d/%m/%Y"):
        parsed = pd.to_datetime(date_col, format=fmt, errors="coerce")
        if not parsed.isnull().all():
            df["date"] = parsed
            break
    if df["date"].isnull().all():
        df["date"] = pd.to_datetime(date_col, errors="coerce")
        
    df["ticker"] = df["ticker"].astype(str).str.lower().str.strip()
    df = df.dropna(subset=["date"]).sort_values(["ticker", "date"]).reset_index(drop=True)

    cot_gia = ["open", "high", "low", "close", "adj_open", "adj_high", "adj_low", "adj_close"]
    for c in cot_gia:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    vnindex = df[df["ticker"] == VNINDEX].set_index("date")["adj_close"].sort_index()

    co_phieu = df[df["ticker"] != VNINDEX].copy()
    for c in cot_gia:
        if c in co_phieu.columns:
            co_phieu[c] = co_phieu[c] * 1000.0          # nghìn đồng -> đồng

    adj_close = co_phieu.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    adj_open  = co_phieu.pivot(index="date", columns="ticker", values="adj_open").sort_index()
    volume    = co_phieu.pivot(index="date", columns="ticker", values="volume").sort_index()
    
    # Forward fill prices to handle missing trading days, then drop completely empty columns
    adj_close = adj_close.ffill().bfill().dropna(how="all", axis=1)
    adj_open  = adj_open.ffill().bfill().dropna(how="all", axis=1)
    volume    = volume.fillna(0.0)
    
    return adj_close, adj_open, volume, vnindex

# Cache the backtest runs
@st.cache_data
def chay_backtest_va_benchmarks(adj_close, adj_open, volume, vnindex, 
                                loc_che_do, ma_che_do, von, phi,
                                ngay_bat_dau, ngay_ket_thuc, so_ma,
                                cua_so_hoc, ma_loc_xu_huong, min_lich_su,
                                max_ngay_kl_0, kieu_trong_so):
    """Execute the core strategy and calculate comparison benchmarks."""
    
    # Calculate global indicators needed
    ma_loc = duong_trung_binh_dong(adj_close, ma_loc_xu_huong)
    che_do = (vnindex >= vnindex.rolling(ma_che_do, min_periods=ma_che_do).mean())

    lich = adj_close.index
    ngay_gd = lich[(lich >= pd.to_datetime(ngay_bat_dau)) & (lich <= pd.to_datetime(ngay_ket_thuc))]
    ngay_tcb = set(lay_ngay_tai_can_bang(lich, ngay_bat_dau, ngay_ket_thuc))

    tien_mat = float(von)
    co_phieu = {}
    top_hien = []
    diem_hien = {}
    nhat_ky, duong_von, lich_su = [], [], []
    tong_phi = 0.0
    che_do_truoc = None

    def gia_tri_mo_cua(ngay):
        go = adj_open.loc[ngay]
        v = tien_mat
        for m, s in co_phieu.items():
            if s > 0 and not pd.isna(go.get(m, np.nan)):
                v += s * go[m]
        return v

    def thuc_thi_ve_target(ngay, target_w):
        nonlocal tien_mat, tong_phi
        go = adj_open.loc[ngay]
        V  = gia_tri_mo_cua(ngay)
        cac_ma = sorted(set(list(co_phieu.keys()) + list(target_w.keys())))
        muc_tieu = {}
        for m in cac_ma:
            gp = go.get(m, np.nan)
            muc_tieu[m] = co_phieu.get(m, 0) if (pd.isna(gp) or gp <= 0) \
                          else int((V * target_w.get(m, 0.0)) / (gp * (1 + phi)))
        for m in cac_ma:  # Sell first
            gp = go.get(m, np.nan)
            if pd.isna(gp) or gp <= 0:
                continue
            cur, ts = co_phieu.get(m, 0), muc_tieu[m]
            if ts < cur:
                sl = cur - ts
                tien = sl * gp * (1 - phi)
                tien_mat += tien
                tong_phi += sl * gp * phi
                co_phieu[m] = ts
                nhat_ky.append({"ngay": ngay, "ma": m.upper(), "loai": "BÁN",
                                "so_co_phieu": sl, "gia": gp, "gia_tri": tien})
        for m in cac_ma:  # Buy after
            gp = go.get(m, np.nan)
            if pd.isna(gp) or gp <= 0:
                continue
            cur, ts = co_phieu.get(m, 0), muc_tieu[m]
            if ts > cur:
                sl = ts - cur
                chi = sl * gp * (1 + phi)
                if chi > tien_mat + 1e-6:
                    sl = int(tien_mat / (gp * (1 + phi)))
                    chi = sl * gp * (1 + phi)
                if sl > 0:
                    tien_mat -= chi
                    tong_phi += sl * gp * phi
                    co_phieu[m] = cur + sl
                    nhat_ky.append({"ngay": ngay, "ma": m.upper(), "loai": "MUA",
                                    "so_co_phieu": sl, "gia": gp, "gia_tri": chi})

    for ngay in ngay_gd:
        vt = lich.get_loc(ngay)
        ngay_tin_hieu = lich[vt - 1] if vt > 0 else ngay

        che_do_on = bool(che_do.loc[ngay_tin_hieu]) \
            if (ngay_tin_hieu in che_do.index and not pd.isna(che_do.loc[ngay_tin_hieu])) else True

        la_tai_can_bang = ngay in ngay_tcb
        dao_che_do = (che_do_truoc is not None) and (che_do_on != che_do_truoc) and loc_che_do

        if la_tai_can_bang or dao_che_do:
            if la_tai_can_bang:
                top_hien, diem_hien = chon_top_ma(ngay_tin_hieu, adj_close, volume,
                                                  ma_loc, so_ma, cua_so_hoc,
                                                  min_lich_su, max_ngay_kl_0)
            if (not loc_che_do) or che_do_on:
                target_w = trong_so_danh_muc(top_hien, diem_hien, ngay_tin_hieu, adj_close, kieu_trong_so)
            else:
                target_w = {}
            thuc_thi_ve_target(ngay, target_w)

        che_do_truoc = che_do_on

        g = adj_close.loc[ngay]
        gt = tien_mat + sum(s * g[m] for m, s in co_phieu.items() if s > 0 and not pd.isna(g.get(m, np.nan)))
        duong_von.append((ngay, gt))
        ty_trong = sum(s * g[m] for m, s in co_phieu.items() if s > 0 and not pd.isna(g.get(m, np.nan))) / gt
        lich_su.append((ngay, ty_trong, dict(co_phieu)))

    dv_cl = pd.Series({d: v for d, v in duong_von}).sort_index()

    # --- Benchmark 1: Buy & Hold Top 5 from the start ---
    ngay_cuoi_hoc = adj_close.index[adj_close.index < pd.to_datetime(ngay_bat_dau)][-1]
    top5_dau, diem_dau = chon_top_ma(ngay_cuoi_hoc, adj_close, volume, ma_loc, 
                                     so_ma, cua_so_hoc, min_lich_su, max_ngay_kl_0)
    w_dau = trong_so_danh_muc(top5_dau, diem_dau, ngay_cuoi_hoc, adj_close, kieu_trong_so)
    
    go = adj_open.loc[ngay_gd[0]]
    co_phieu_bm = {}
    for ma in top5_dau:
        gp = go[ma]
        if pd.isna(gp) or gp <= 0:
            continue
        co_phieu_bm[ma] = int((von * w_dau.get(ma, 0.0)) / (gp * (1 + phi)))
    
    dv_top5_list = []
    for ngay in ngay_gd:
        g = adj_close.loc[ngay]
        val = sum(s * g[m] for m, s in co_phieu_bm.items() if not pd.isna(g[m]))
        dv_top5_list.append((ngay, val))
    dv_top5 = pd.Series({d: v for d, v in dv_top5_list}).sort_index()

    # --- Benchmark 2: VN-Index buy and hold ---
    s_vni = vnindex[(vnindex.index >= pd.to_datetime(ngay_bat_dau)) & (vnindex.index <= pd.to_datetime(ngay_ket_thuc))]
    dv_vni = von * s_vni / s_vni.iloc[0]

    # --- Benchmark 3: Equal allocation across all market stock ---
    gia_dau = adj_close.loc[ngay_gd[0]]
    ma_hop_le = [m for m in adj_close.columns if not pd.isna(gia_dau[m]) and gia_dau[m] > 0]
    sub = adj_close.loc[ngay_gd, ma_hop_le]
    dv_deu = von * ((sub / sub.iloc[0]) * (1.0 / len(ma_hop_le))).sum(axis=1)

    return {
        "strategy": dv_cl,
        "bm_top5": dv_top5,
        "bm_vni": dv_vni,
        "bm_deu": dv_deu,
        "nhat_ky": pd.DataFrame(nhat_ky),
        "tong_phi": tong_phi,
        "so_lenh": len(nhat_ky),
        "lich_su": lich_su,
        "top_dau": top5_dau,
        "w_dau": w_dau,
        "diem_dau": diem_dau
    }

def cac_chi_so(duong_von, von_ban_dau):
    """Calculate key performance indicators (KPIs) for portfolio equity curve."""
    dv = duong_von.dropna()
    if len(dv) < 2:
        return {}
    ln = dv.pct_change().dropna()
    tong_loi = dv.iloc[-1] / von_ban_dau - 1.0
    n = len(dv)
    cagr = (dv.iloc[-1] / von_ban_dau) ** (SO_NGAY_TRONG_NAM / n) - 1.0
    vol = ln.std() * np.sqrt(SO_NGAY_TRONG_NAM)
    sharpe = (ln.mean() / ln.std() * np.sqrt(SO_NGAY_TRONG_NAM)) if ln.std() > 0 else 0.0
    am = ln[ln < 0]
    sortino = (ln.mean() / am.std() * np.sqrt(SO_NGAY_TRONG_NAM)) if (len(am) > 0 and am.std() > 0) else np.nan
    dd = dv / dv.cummax() - 1.0
    max_dd = dd.min()
    calmar = (cagr / abs(max_dd)) if max_dd < 0 else np.nan
    return {
        "Giá trị cuối (VND)": dv.iloc[-1],
        "Tổng lợi nhuận (%)": tong_loi * 100,
        "CAGR (%)": cagr * 100,
        "Biến động năm (%)": vol * 100,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max Drawdown (%)": max_dd * 100,
        "Calmar": calmar
    }

def kiem_dinh_lon_hon_0(chuoi_loi_nhuan, alpha=0.05):
    """H0: trung bình <= 0 ; H1: trung bình > 0 (một phía, t-test một mẫu)."""
    x = pd.Series(chuoi_loi_nhuan).dropna()
    if len(x) == 0:
        return {"t_stat": np.nan, "p_value": np.nan, "co_y_nghia": False, "trung_binh": np.nan}
    t, p = ttest_1samp(x, 0.0, alternative="greater")
    return {"t_stat": t, "p_value": p, "co_y_nghia": p < alpha, "trung_binh": x.mean()}

def kiem_dinh_vuot_benchmark(ln_chien_luoc, ln_benchmark, alpha=0.05):
    """Kiểm định theo cặp. H1: lợi nhuận chiến lược > benchmark. Dùng t-test cặp và Wilcoxon."""
    a = pd.Series(ln_chien_luoc).reset_index(drop=True)
    b = pd.Series(ln_benchmark).reset_index(drop=True)
    n = min(len(a), len(b))
    if n == 0:
        return {"chenh_lech_tb": np.nan, "t_p_value": np.nan, "t_co_y_nghia": False,
                "wilcoxon_p_value": np.nan, "wilcoxon_co_y_nghia": False}
    a, b = a.iloc[:n], b.iloc[:n]
    chenh = (a - b).dropna()
    if len(chenh) == 0:
        return {"chenh_lech_tb": np.nan, "t_p_value": np.nan, "t_co_y_nghia": False,
                "wilcoxon_p_value": np.nan, "wilcoxon_co_y_nghia": False}
    t, t_p = ttest_1samp(chenh, 0.0, alternative="greater")
    try:
        _, w_p = wilcoxon(a, b, alternative="greater")
    except Exception:
        w_p = np.nan
    return {"chenh_lech_tb": chenh.mean(), "t_p_value": t_p, "t_co_y_nghia": t_p < alpha,
            "wilcoxon_p_value": w_p,
            "wilcoxon_co_y_nghia": (w_p < alpha) if not (w_p is np.nan or pd.isna(w_p)) else False}

def loi_nhuan_dinh_ky(duong_von, ky="ME"):
    """Suất sinh lời theo kỳ: 'ME'/'M' = cuối tháng, 'QE'/'Q' = cuối quý."""
    try:
        return duong_von.dropna().resample(ky).last().pct_change().dropna()
    except ValueError:
        old_ky = "M" if ky.startswith("M") else "Q"
        return duong_von.dropna().resample(old_ky).last().pct_change().dropna()

# -----------------------------------------------------------------------------------------
# INTERFACE IMPLEMENTATION
# -----------------------------------------------------------------------------------------

# Title Banner
st.markdown("""
    <div class="header-container">
        <h1 class="header-title">📈 TỐI ƯU HÓA DANH MỤC ĐẦU TƯ CỔ PHIẾU HOSE</h1>
        <p class="header-subtitle">Chiến lược MACD Chọn lọc + Nắm giữ + Bộ lọc chế độ thị trường SMA (2020 - 2023)</p>
    </div>
""", unsafe_allow_html=True)

# Brief Info Callout
st.markdown("""
    <div class="info-callout">
        <strong>Giới thiệu chiến lược:</strong> Ứng dụng này giúp bạn kiểm định chiến lược tối ưu hóa danh mục đầu tư:
        1. <strong>Chọn lọc:</strong> Lọc danh mục gồm 5 cổ phiếu tối ưu nhất bằng Sharpe dựa trên chiến lược MACD trong quá khứ 1 năm.
        2. <strong>Bảo vệ:</strong> Sử dụng đường SMA 200 ngày của VN-Index làm bộ lọc xu hướng thị trường (chế độ Bull/Bear).
        3. <strong>Vận hành:</strong> Tự động rút về 100% tiền mặt nếu VN-Index thủng SMA200, hạn chế tối đa rủi ro suy thoái.
    </div>
""", unsafe_allow_html=True)

# Sidebar - Parameter Settings
st.sidebar.markdown("### ⚙️ Cấu hình Chiến lược")

# File Upload Section
uploaded_file = st.sidebar.file_uploader("1. Tải lên tệp HOSE CSV", type=["csv"])

# Parameters Group 1: General
st.sidebar.markdown("#### Tham số chung")
von_ban_dau = st.sidebar.number_input("Số vốn ban đầu (VND)", min_value=10_000_000, max_value=100_000_000_000, value=1_000_000_000, step=50_000_000)
phi_giao_dich = st.sidebar.number_input("Phí giao dịch mỗi lệnh (%)", min_value=0.0, max_value=2.0, value=0.1, step=0.05, format="%f") / 100.0
so_ma = st.sidebar.slider("Số cổ phiếu trong danh mục", min_value=3, max_value=15, value=5)
kieu_trong_so = st.sidebar.selectbox("Kiểu phân bổ trọng số", options=["score", "equal", "inverse_vol", "vol_target"], index=0, format_func=lambda x: {
    "score": "Tỷ lệ theo điểm MACD (Sharpe)",
    "equal": "Chia đều trọng số (Equal)",
    "inverse_vol": "Nghịch biến động (Inverse Vol)",
    "vol_target": "Nghịch phương sai (Vol Target)"
}[x])

# Parameters Group 2: Backtest time
st.sidebar.markdown("#### Thời gian backtest")
# Let users configure date range
date_range = st.sidebar.date_input("Khoảng thời gian đầu tư", value=(pd.to_datetime("2021-01-01"), pd.to_datetime("2023-12-31")))

# Parameters Group 3: Technical Indicators
st.sidebar.markdown("#### Tham số kỹ thuật")
macd_nhanh = st.sidebar.number_input("MACD Chu kỳ nhanh", min_value=5, max_value=30, value=12)
macd_cham = st.sidebar.number_input("MACD Chu kỳ chậm", min_value=10, max_value=60, value=26)
macd_tin_hieu = st.sidebar.number_input("MACD Chu kỳ tín hiệu", min_value=3, max_value=20, value=9)

ma_loc_che_do = st.sidebar.number_input("Độ dài SMA của VN-Index (chế độ thị trường)", min_value=50, max_value=300, value=200)
ma_loc_xu_huong = st.sidebar.number_input("Độ dài SMA lọc xu hướng cổ phiếu", min_value=50, max_value=200, value=120)
cua_so_hoc = st.sidebar.number_input("Cửa sổ học Sharpe (ngày)", min_value=50, max_value=500, value=252)
min_lich_su = st.sidebar.number_input("Lịch sử tối thiểu (ngày)", min_value=100, max_value=300, value=200)
max_ngay_kl_0 = st.sidebar.number_input("Số ngày KL = 0 tối đa", min_value=5, max_value=100, value=30)

# Process logic when CSV uploaded
if uploaded_file is not None:
    try:
        # Load and parse data
        adj_close, adj_open, volume, vnindex = doc_du_lieu_tu_file(uploaded_file)
        
        # Verify uploaded data range
        min_date = adj_close.index.min()
        max_date = adj_close.index.max()
        
        st.success(f"Tải file thành công! Số mã: {adj_close.shape[1]} | Khoảng dữ liệu: {min_date.strftime('%d/%m/%Y')} - {max_date.strftime('%d/%m/%Y')}")
        
        # Parse start and end dates from user input
        if len(date_range) == 2:
            ngay_bat_dau, ngay_ket_thuc = date_range
        else:
            ngay_bat_dau = pd.to_datetime("2021-01-01")
            ngay_ket_thuc = pd.to_datetime("2023-12-31")
            
        # Call backtest engine
        with st.spinner("Đang xử lý mô phỏng danh mục đầu tư..."):
            kq = chay_backtest_va_benchmarks(
                adj_close, adj_open, volume, vnindex,
                loc_che_do=True, ma_che_do=ma_loc_che_do, von=von_ban_dau, phi=phi_giao_dich,
                ngay_bat_dau=ngay_bat_dau, ngay_ket_thuc=ngay_ket_thuc, so_ma=so_ma,
                cua_so_hoc=cua_so_hoc, ma_loc_xu_huong=ma_loc_xu_huong, min_lich_su=min_lich_su,
                max_ngay_kl_0=max_ngay_kl_0, kieu_trong_so=kieu_trong_so
            )

        # Performance Calculations
        metrics_cl = cac_chi_so(kq["strategy"], von_ban_dau)
        metrics_vni = cac_chi_so(kq["bm_vni"], von_ban_dau)
        metrics_top5 = cac_chi_so(kq["bm_top5"], von_ban_dau)
        metrics_deu = cac_chi_so(kq["bm_deu"], von_ban_dau)
        
        # ---------------------------------------------------------------------------------
        # KPI SUMMARY CARD SECTION
        # ---------------------------------------------------------------------------------
        st.markdown("### 🏆 BÁO CÁO HIỆU QUẢ HOẠT ĐỘNG")
        
        kpi_cols = st.columns(5)
        
        # 1. Final Value
        val_cuoi = metrics_cl["Giá trị cuối (VND)"]
        kpi_cols[0].markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val_cuoi:,.0f} Đ</div>
                <div class="metric-label">GIÁ TRỊ CUỐI CÙNG</div>
            </div>
        """, unsafe_allow_html=True)
        
        # 2. Cumulative Return
        tong_loi = metrics_cl["Tổng lợi nhuận (%)"]
        cls_color = "metric-value-green" if tong_loi >= 0 else "metric-value-red"
        kpi_cols[1].markdown(f"""
            <div class="metric-card">
                <div class="metric-value {cls_color}">{tong_loi:+.2f}%</div>
                <div class="metric-label">TỔNG LỢI NHUẬN</div>
            </div>
        """, unsafe_allow_html=True)
        
        # 3. CAGR
        cagr = metrics_cl["CAGR (%)"]
        cls_cagr = "metric-value-green" if cagr >= 0 else "metric-value-red"
        kpi_cols[2].markdown(f"""
            <div class="metric-card">
                <div class="metric-value {cls_cagr}">{cagr:.2f}%</div>
                <div class="metric-label">CAGR (TỶ LỆ KÉP NĂM)</div>
            </div>
        """, unsafe_allow_html=True)
        
        # 4. Sharpe Ratio
        sharpe = metrics_cl["Sharpe"]
        cls_sharpe = "metric-value-green" if sharpe >= 1 else ""
        kpi_cols[3].markdown(f"""
            <div class="metric-card">
                <div class="metric-value {cls_sharpe}">{sharpe:.2f}</div>
                <div class="metric-label">HỆ SỐ SHARPE</div>
            </div>
        """, unsafe_allow_html=True)
        
        # 5. Max Drawdown
        max_dd = metrics_cl["Max Drawdown (%)"]
        cls_dd = "metric-value-red" if max_dd < -20 else "metric-value-green"
        kpi_cols[4].markdown(f"""
            <div class="metric-card">
                <div class="metric-value {cls_dd}">{max_dd:.2f}%</div>
                <div class="metric-label">MAX DRAWDOWN</div>
            </div>
        """, unsafe_allow_html=True)
        
        st.write("") # Spacer

        # ---------------------------------------------------------------------------------
        # TABS IMPLEMENTATION
        # ---------------------------------------------------------------------------------
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Kết quả & So sánh", 
            "📈 Biểu đồ Lợi nhuận", 
            "🧩 Phân tích Danh mục", 
            "📝 Nhật ký Giao dịch", 
            "🧮 Kiểm định Thống kê"
        ])
        
        with tab1:
            st.subheader("Bảng so sánh hiệu quả chi tiết (2021-2023)")
            
            # Combine indicators into dataframe
            df_ss = pd.DataFrame({
                "Chiến lược (Chính)": metrics_cl,
                "Mua & Giữ Top 5 đầu kì": metrics_top5,
                "Phân bổ đều Toàn thị trường": metrics_deu,
                "VN-Index (Mua & Giữ)": metrics_vni
            }).T
            
            # Style formatting
            df_ss_display = df_ss.copy()
            df_ss_display["Giá trị cuối (VND)"] = df_ss_display["Giá trị cuối (VND)"].map(lambda x: f"{x:,.0f} VND")
            df_ss_display["Tổng lợi nhuận (%)"] = df_ss_display["Tổng lợi nhuận (%)"].map(lambda x: f"{x:+.2f}%")
            df_ss_display["CAGR (%)"] = df_ss_display["CAGR (%)"].map(lambda x: f"{x:.2f}%")
            df_ss_display["Biến động năm (%)"] = df_ss_display["Biến động năm (%)"].map(lambda x: f"{x:.2f}%")
            df_ss_display["Sharpe"] = df_ss_display["Sharpe"].map(lambda x: f"{x:.3f}")
            df_ss_display["Sortino"] = df_ss_display["Sortino"].map(lambda x: f"{x:.3f}" if pd.notnull(x) else "NaN")
            df_ss_display["Max Drawdown (%)"] = df_ss_display["Max Drawdown (%)"].map(lambda x: f"{x:.2f}%")
            df_ss_display["Calmar"] = df_ss_display["Calmar"].map(lambda x: f"{x:.3f}" if pd.notnull(x) else "NaN")
            
            st.dataframe(df_ss_display, use_container_width=True)
            
            # Display additional info
            st.markdown(f"""
            - **Tổng phí giao dịch đã trả:** `{kq['tong_phi']:,.0f} VND`
            - **Tổng số lệnh thực hiện:** `{kq['so_lenh']} lệnh`
            - **Số tiền mặt cuối kỳ:** `{kq['tien_mat_cuoi']:,.0f} VND`
            """)
            
        with tab2:
            st.subheader("Biểu đồ tăng trưởng tài sản lũy kế")
            
            # Plotly Equity Curve Chart
            fig_equity = go.Figure()
            fig_equity.add_trace(go.Scatter(x=kq["strategy"].index, y=kq["strategy"].values, name="Chiến lược (MACD + Regime filter)", line=dict(color="#3b82f6", width=2.5)))
            fig_equity.add_trace(go.Scatter(x=kq["bm_top5"].index, y=kq["bm_top5"].values, name="Mua & Giữ 5 mã đầu", line=dict(color="#10b981", width=1.5, dash='dash')))
            fig_equity.add_trace(go.Scatter(x=kq["bm_deu"].index, y=kq["bm_deu"].values, name="Phân bổ đều toàn thị trường", line=dict(color="#f59e0b", width=1.5, dash='dot')))
            fig_equity.add_trace(go.Scatter(x=kq["bm_vni"].index, y=kq["bm_vni"].values, name="VN-Index (Mua & Giữ)", line=dict(color="#94a3b8", width=1.5)))
            
            fig_equity.update_layout(
                title="Đường giá trị tài sản danh mục (Vốn ban đầu: 1 tỷ VND)",
                xaxis_title="Thời gian",
                yaxis_title="Giá trị tài sản (VND)",
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                template="plotly_dark",
                hovermode="x unified",
                margin=dict(l=40, r=40, t=50, b=40)
            )
            st.plotly_chart(fig_equity, use_container_width=True)

            # Drawdown Chart
            st.subheader("Mức sụt giảm từ đỉnh (Drawdown)")
            dd_cl = (kq["strategy"] / kq["strategy"].cummax() - 1) * 100
            dd_vni = (kq["bm_vni"] / kq["bm_vni"].cummax() - 1) * 100
            
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(x=dd_cl.index, y=dd_cl.values, name="Chiến lược", fill='tozeroy', fillcolor="rgba(59, 130, 246, 0.25)", line=dict(color="#3b82f6", width=1.5)))
            fig_dd.add_trace(go.Scatter(x=dd_vni.index, y=dd_vni.values, name="VN-Index", line=dict(color="#ef4444", width=1.2)))
            
            fig_dd.update_layout(
                title="Mức sụt giảm Drawdown (%) theo thời gian",
                xaxis_title="Thời gian",
                yaxis_title="Phần trăm sụt giảm (%)",
                legend=dict(yanchor="bottom", y=0.01, xanchor="left", x=0.01),
                template="plotly_dark",
                margin=dict(l=40, r=40, t=50, b=40)
            )
            st.plotly_chart(fig_dd, use_container_width=True)

        with tab3:
            st.subheader("Danh mục khởi đầu & Trọng số mục tiêu (Đầu 2021)")
            top_dau = kq["top_dau"]
            w_dau = kq["w_dau"]
            diem_dau = kq["diem_dau"]
            
            col_weights = st.columns(len(top_dau) if len(top_dau) > 0 else 1)
            for idx, ma in enumerate(top_dau):
                col_weights[idx].metric(
                    label=f"Ticker: {ma.upper()}",
                    value=f"{w_dau[ma]*100:.2f}%",
                    delta=f"Sharpe: {diem_dau[ma]:.3f}"
                )
                
            # Cash Weight Over Time Chart
            st.subheader("Tỷ trọng giải ngân cổ phiếu vs Tiền mặt")
            ts_time = [d for d, _, _ in kq["lich_su"]]
            ts_weight = [w * 100 for _, w, _ in kq["lich_su"]]
            
            fig_cash = go.Figure()
            fig_cash.add_trace(go.Scatter(
                x=ts_time, y=ts_weight, 
                name="Tỷ trọng cổ phiếu",
                fill='tozeroy', 
                fillcolor="rgba(59, 130, 246, 0.4)", 
                line=dict(color="#3b82f6", width=2)
            ))
            fig_cash.update_layout(
                title="Tỷ trọng đầu tư vào cổ phiếu (%) - Phần còn lại (100% - x%) là Tiền mặt phòng thủ",
                xaxis_title="Thời gian",
                yaxis_title="Tỷ trọng đầu tư (%)",
                yaxis=dict(range=[0, 105]),
                template="plotly_dark",
                margin=dict(l=40, r=40, t=50, b=40)
            )
            st.plotly_chart(fig_cash, use_container_width=True)

            # Quarterly Returns Chart
            st.subheader("Lợi nhuận theo quý (QE): Chiến lược vs VN-Index")
            ln_cl_quy = loi_nhuan_dinh_ky(kq["strategy"], "QE")
            vni_quy = loi_nhuan_dinh_ky(kq["bm_vni"], "QE")
            
            idx_quy = [str(p) for p in ln_cl_quy.index.to_period("Q")]
            
            fig_quy = go.Figure(data=[
                go.Bar(name='Chiến lược', x=idx_quy, y=ln_cl_quy.values * 100, marker_color='#3b82f6'),
                go.Bar(name='VN-Index', x=idx_quy, y=vni_quy.values * 100, marker_color='#94a3b8')
            ])
            fig_quy.update_layout(
                barmode='group',
                title="So sánh lợi nhuận định kỳ theo quý (%)",
                xaxis_title="Quý",
                yaxis_title="Lợi nhuận (%)",
                template="plotly_dark",
                margin=dict(l=40, r=40, t=50, b=40)
            )
            st.plotly_chart(fig_quy, use_container_width=True)

        with tab4:
            st.subheader("Nhật ký Lịch sử Giao dịch chi tiết")
            df_nk = kq["nhat_ky"]
            
            if len(df_nk) > 0:
                # Format print layout
                df_nk_display = df_nk.copy()
                df_nk_display["ngay"] = pd.to_datetime(df_nk_display["ngay"]).dt.strftime('%d/%m/%Y')
                df_nk_display["gia"] = df_nk_display["gia"].map(lambda x: f"{x:,.1f}")
                df_nk_display["gia_tri"] = df_nk_display["gia_tri"].map(lambda x: f"{x:,.0f} VND")
                df_nk_display["so_co_phieu"] = df_nk_display["so_co_phieu"].map(lambda x: f"{x:,}")
                
                st.dataframe(df_nk_display, use_container_width=True)
                
                # Download Option
                csv = df_nk.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="📥 Tải xuống Nhật ký Giao dịch (CSV)",
                    data=csv,
                    file_name="nhat_ky_giao_dich.csv",
                    mime="text/csv"
                )
            else:
                st.info("Không có giao dịch nào được thực hiện trong giai đoạn này.")

        with tab5:
            st.subheader("Kiểm định ý nghĩa thống kê (Statistical Hypothesis Tests)")
            st.markdown("""
            Chúng ta kiểm định xem tỷ suất sinh lời tháng của **Chiến lược** có vượt trội hơn 0 và vượt trội hơn **VN-Index** một cách có ý nghĩa thống kê hay không.
            """)
            
            ln_cl_thang  = loi_nhuan_dinh_ky(kq["strategy"], "ME")
            ln_vni_thang = loi_nhuan_dinh_ky(kq["bm_vni"], "ME")
            
            # 1. Strategy > 0
            kd1 = kiem_dinh_lon_hon_0(ln_cl_thang)
            # 2. Strategy > VN-Index
            kd2 = kiem_dinh_vuot_benchmark(ln_cl_thang, ln_vni_thang)
            
            col_stat1, col_stat2 = st.columns(2)
            
            with col_stat1:
                st.markdown("#### 1. Kiểm định Lợi nhuận Chiến lược > 0")
                st.markdown(f"""
                - **Giả thuyết H0:** Tỷ suất lợi nhuận trung bình tháng của chiến lược $\le 0$.
                - **Giả thuyết H1:** Tỷ suất lợi nhuận trung bình tháng của chiến lược $> 0$.
                - **Trung bình tháng:** `{kd1['trung_binh']*100:.3f}%`
                - **Hệ số t-stat:** `{kd1['t_stat']:.4f}`
                - **p-value:** `{kd1['p_value']:.5f}`
                """)
                if kd1['co_y_nghia']:
                    st.success("✅ **Kết luận:** Bác bỏ H0. Lợi nhuận chiến lược > 0 có ý nghĩa thống kê (ở mức ý nghĩa 5%).")
                else:
                    st.warning("⚠️ **Kết luận:** Chưa đủ cơ sở bác bỏ H0. Lợi nhuận chiến lược > 0 chưa có ý nghĩa thống kê rõ rệt.")
                    
            with col_stat2:
                st.markdown("#### 2. Kiểm định Lợi nhuận Chiến lược > VN-Index")
                st.markdown(f"""
                - **Giả thuyết H0:** Lợi nhuận Chiến lược không vượt trội hơn VN-Index.
                - **Giả thuyết H1:** Lợi nhuận Chiến lược vượt trội hơn VN-Index.
                - **Chênh lệch trung bình tháng:** `{kd2['chenh_lech_tb']*100:+.3f}%`
                
                **Kiểm định T-test cặp (Paired t-test):**
                - **p-value:** `{kd2['t_p_value']:.5f}`
                - **Có ý nghĩa:** `{'Có' if kd2['t_co_y_nghia'] else 'Không'}`
                
                **Kiểm định Wilcoxon signed-rank:**
                - **p-value:** `{kd2['wilcoxon_p_value']:.5f}`
                - **Có ý nghĩa:** `{'Có' if kd2['wilcoxon_co_y_nghia'] else 'Không'}`
                """)
                if kd2['t_co_y_nghia'] or kd2['wilcoxon_co_y_nghia']:
                    st.success("✅ **Kết luận:** Chiến lược chiến thắng VN-Index một cách có ý nghĩa thống kê rõ rệt.")
                else:
                    st.warning("⚠️ **Kết luận:** Sự vượt trội của chiến lược so với VN-Index chưa có ý nghĩa thống kê rõ rệt.")
                    
    except Exception as e:
        st.error(f"Đã xảy ra lỗi khi chạy backtest: {e}")
        st.exception(e)
else:
    # Beautiful Landing Welcome Screen
    st.markdown("""
        <div style="background-color: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 3rem; text-align: center; margin-top: 2rem;">
            <span style="font-size: 5rem;">📥</span>
            <h2 style="color: #ffffff; margin-top: 1.5rem;">Vui lòng tải lên dữ liệu sàn HOSE để bắt đầu</h2>
            <p style="color: #94a3b8; max-width: 600px; margin: 0.5rem auto 2rem auto; font-size: 1.1rem; line-height: 1.6;">
                Tải lên tệp CSV chứa dữ liệu giá cổ phiếu của HOSE (ví dụ: <code>HOSE_2020_2023.csv</code>) ở menu bên trái. Ứng dụng sẽ tự động phân tích, tính toán các chỉ báo MACD, chạy thuật toán tối ưu danh mục đầu tư và kiểm định lịch sử.
            </p>
            <div style="background-color: #0f172a; padding: 1.5rem; border-radius: 8px; text-align: left; max-width: 500px; margin: 0 auto; border: 1px solid #1e293b;">
                <strong style="color: #60a5fa; display: block; margin-bottom: 0.5rem;">Cấu trúc file CSV đầu vào hợp lệ:</strong>
                <code style="color: #cbd5e1; font-family: monospace; display: block; font-size: 0.9rem;">
                    date,ticker,open,high,low,close,volume,adj_close,adj_open<br>
                    12/31/2020,vnindex,1103.87,1107.57,1101.44,1103.87,...<br>
                    12/31/2020,aaa,13.9,14.05,13.85,13.9,1031970,10.63,10.63<br>
                    12/31/2020,aav,12.2,12.3,12.0,12.0,147800,10.74,10.92
                </code>
            </div>
        </div>
    """, unsafe_allow_html=True)
