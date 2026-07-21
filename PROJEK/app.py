import os
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

warnings.filterwarnings('ignore')



# ==========================================
# KONFIGURASI SISTEM
# ==========================================
st.set_page_config(page_title="Dashboard Evaluasi Model Trading", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIRECTORY = os.path.join(BASE_DIR, "model")
DATA_BTC_RAW = os.path.join(BASE_DIR, "data", "btc_raw.csv")
ARCHITECTURES = ["LSTM", "GRU", "XGB"]
TOTAL_PERIODS = 5
INITIAL_CAPITAL = 1000.0
EXCHANGE_FEE_RATE = 0.001
MARGIN_INTEREST_RATE_DAILY = 0.0000099
OPTIMAL_ESTIMATOR_COUNT = 1 

COMBINATIONS = [
    ([], "Baseline (Hanya BTC)"),
    (['emas'], "+ Emas"),
    (['dxy'], "+ DXY"),
    (['vix'], "+ VIX"),
    (['news'], "+ News")
]

st.write("### 🔍 Cek Sub-Folder (Deep Debugging)")

# Kita ambil contoh arsitektur XGB
xgb_dir = os.path.join(MODEL_DIRECTORY, "XGB")
if os.path.exists(xgb_dir):
    st.write("✅ Folder 'XGB' ketemu! Isinya:", os.listdir(xgb_dir))
    
    # Kodinganmu memanggil f"{arch}_{variant_upper}" -> "XGB_MURNI"
    xgb_murni_dir = os.path.join(xgb_dir, "XGB_MURNI")
    if os.path.exists(xgb_murni_dir):
        st.write("✅ Folder 'XGB_MURNI' ketemu! Isinya:", os.listdir(xgb_murni_dir))
        
        # Cek lebih dalam lagi ke Arsip_Ujian_Silang
        arsip_dir = os.path.join(xgb_murni_dir, "Arsip_Ujian_Silang")
        if os.path.exists(arsip_dir):
            st.write("✅ Folder 'Arsip_Ujian_Silang' ketemu! Isinya:", os.listdir(arsip_dir))
        else:
            st.error(f"❌ Folder 'Arsip_Ujian_Silang' TIDAK ADA di dalam {xgb_murni_dir}. Cek huruf kapitalnya!")
            
    else:
        st.error(f"❌ Folder 'XGB_MURNI' TIDAK ADA di dalam {xgb_dir}. Jangan-jangan di GitHub tulisannya 'XGB_Murni' atau 'xgb_murni'?")
else:
    st.error("❌ Folder 'XGB' TIDAK DITEMUKAN!")
# ==========================================
# FUNGSI BACKEND UTAMA (Di-Cache)
# ==========================================
@st.cache_data
def load_raw_btc_data():
    if os.path.exists(DATA_BTC_RAW):
        df = pd.read_csv(DATA_BTC_RAW)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    return None

@st.cache_data
def simulate_financial_hold(predictions: np.ndarray, target_returns_pct: np.ndarray, low_returns_pct: np.ndarray, high_returns_pct: np.ndarray, sl_pct: float, tp_pct: float):
    balance = INITIAL_CAPITAL
    position_open = False
    entry_balance = 0.0
    
    total_trades = 0
    total_fees = 0.0
    daily_pnl = []
    balance_history = []

    for i, action in enumerate(predictions):
        ret = target_returns_pct[i]
        low_ret = low_returns_pct[i]
        high_ret = high_returns_pct[i]
        
        if action == 1:
            if not position_open:
                entry_fee = balance * EXCHANGE_FEE_RATE
                balance -= entry_fee
                total_fees += entry_fee
                position_open = True
                entry_balance = balance  
                total_trades += 1
            
            # Kalkulasi fluktuasi maksimal dan minimal hari ini dibandingkan titik Entry
            lowest_balance_today = balance * (1 + low_ret)
            highest_balance_today = balance * (1 + high_ret)
            
            drop_from_entry = (lowest_balance_today / entry_balance) - 1
            gain_from_entry = (highest_balance_today / entry_balance) - 1
            
            # CEK 1: Apakah menyentuh Stop Loss? (Prioritas Utama)
            if sl_pct > 0 and drop_from_entry <= -sl_pct:
                gross_balance_at_sl = entry_balance * (1 - sl_pct)
                exit_fee = gross_balance_at_sl * EXCHANGE_FEE_RATE
                total_fees += exit_fee
                
                net_pnl = (gross_balance_at_sl - exit_fee) - balance
                balance = gross_balance_at_sl - exit_fee
                position_open = False
                
            # CEK 2: Apakah menyentuh Take Profit?
            elif tp_pct > 0 and gain_from_entry >= tp_pct:
                gross_balance_at_tp = entry_balance * (1 + tp_pct)
                exit_fee = gross_balance_at_tp * EXCHANGE_FEE_RATE
                total_fees += exit_fee
                
                net_pnl = (gross_balance_at_tp - exit_fee) - balance
                balance = gross_balance_at_tp - exit_fee
                position_open = False
                
            # CEK 3: Hari normal, posisi tetap ditahan atau model menyuruh keluar
            else:
                gross_pnl = balance * ret
                
                # Jika hari terakhir atau besok disuruh jual oleh model
                if (i == len(predictions) - 1) or (predictions[i + 1] == 0):
                    exit_fee = (balance + gross_pnl) * EXCHANGE_FEE_RATE
                    total_fees += exit_fee
                    position_open = False
                else:
                    exit_fee = 0.0
                    
                net_pnl = gross_pnl - exit_fee
                balance += net_pnl
        else:
            net_pnl = 0.0
            position_open = False
            
        daily_pnl.append(net_pnl)
        balance_history.append(balance)

    final_balance = balance_history[-1] if balance_history else INITIAL_CAPITAL
    roi = ((final_balance - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    fee_ratio = (total_fees / INITIAL_CAPITAL) * 100

    daily_returns = np.array(daily_pnl) / (np.array(balance_history) - np.array(daily_pnl))
    daily_returns = np.nan_to_num(daily_returns)
    mean_ret = np.mean(daily_returns)
    std_ret = np.std(daily_returns)
    downside_std = np.std(daily_returns[daily_returns < 0]) if np.any(daily_returns < 0) else np.nan

    sharpe = (mean_ret / std_ret) * np.sqrt(365) if std_ret > 0 else 0.0
    sortino = (mean_ret / downside_std) * np.sqrt(365) if not np.isnan(downside_std) and downside_std > 0 else 0.0

    peak = np.maximum.accumulate(balance_history) if balance_history else [INITIAL_CAPITAL]
    drawdown = (balance_history - peak) / peak
    max_drawdown = np.min(drawdown) * 100 if len(drawdown) > 0 else 0.0

    long_days = np.where(predictions == 1)[0]
    if len(long_days) > 0:
        winning_long_days = np.sum(np.array(daily_pnl)[long_days] > 0)
        win_rate = (winning_long_days / len(long_days)) * 100
    else:
        win_rate = np.nan

    profit_factor = np.nan
    if np.sum(np.array(daily_pnl) < 0) != 0:
        gross_profit = np.sum(np.array(daily_pnl)[np.array(daily_pnl) > 0])
        gross_loss = abs(np.sum(np.array(daily_pnl)[np.array(daily_pnl) < 0]))
        profit_factor = gross_profit / gross_loss

    return {
        'roi': roi, 'sharpe': sharpe, 'sortino': sortino,
        'max_drawdown': max_drawdown, 'profit_factor': profit_factor, 'win_rate': win_rate,
        'total_trades': total_trades, 'total_fees': total_fees, 'fee_ratio': fee_ratio,
        'balance_history': balance_history 
    }

@st.cache_data
def evaluate_ensemble_scenario_full(architecture, variant_upper, mode, target_period, sl_pct, tp_pct):
    all_targets, all_predictions, all_returns, all_lows, all_highs, all_dates = [], [], [], [], [], []
    
    periods_to_run = range(1, TOTAL_PERIODS + 1) if target_period == 0 else [target_period]

    for eval_period in periods_to_run:
        model_id = 0 if mode == 'statis' else eval_period
        dir_path = os.path.join(MODEL_DIRECTORY, architecture, f"{architecture}_{variant_upper}",
                                "Arsip_Ujian_Silang", f"Model_{model_id:02d}_di_Masa_{eval_period:02d}")
        pred_file = os.path.join(dir_path, 'database_prediksi_mentah.csv')
        report_file = os.path.join(dir_path, 'database_juri_lengkap.csv')
        
        if not (os.path.exists(pred_file) and os.path.exists(report_file)):
            continue

        pred_df = pd.read_csv(pred_file)
        report_df = pd.read_csv(report_file)
        top_estimators = report_df.sort_values(by='f1_macro', ascending=False)['nama_juri'].head(OPTIMAL_ESTIMATOR_COUNT).tolist()
        avg_prob = pred_df[top_estimators].mean(axis=1)
        decisions = (avg_prob >= 0.50).astype(int)

        all_targets.extend(pred_df['target_asli'].values)
        all_predictions.extend(decisions)
        all_returns.extend(pred_df['wasit_target'].values)
        all_lows.extend(pred_df['wasit_low'].values)
        all_highs.extend(pred_df['wasit_high'].values)
        
        if 'tanggal' in pred_df.columns:
            all_dates.extend(pred_df['tanggal'].values)
        else:
            all_dates.extend([f"Masa {eval_period} - Day {i}" for i in range(len(pred_df))])

    if not all_targets:
        return None, None

    targets_arr = np.array(all_targets)
    preds_arr = np.array(all_predictions)
    pct_returns = np.exp(np.array(all_returns) / 100.0) - 1
    low_returns_pct = np.exp(np.array(all_lows) / 100.0) - 1
    high_returns_pct = np.exp(np.array(all_highs) / 100.0) - 1

    acc = accuracy_score(targets_arr, preds_arr) * 100
    f1_macro = f1_score(targets_arr, preds_arr, average='macro') * 100
    f1_long = f1_score(targets_arr, preds_arr, average='binary', pos_label=1) * 100
    f1_short = f1_score(targets_arr, preds_arr, average='binary', pos_label=0) * 100

    finansial = simulate_financial_hold(preds_arr, pct_returns, low_returns_pct, high_returns_pct, sl_pct, tp_pct)
    
    equity_curve = pd.DataFrame({
        'Tanggal': pd.to_datetime(all_dates, errors='coerce'),
        'Balance': finansial['balance_history'],
        'Signal': preds_arr
    })

    metrics = [acc, f1_macro, f1_long, f1_short,
               finansial['roi'], finansial['sharpe'], finansial['sortino'],
               finansial['max_drawdown'], finansial['profit_factor'], finansial['win_rate'],
               finansial['total_trades'], finansial['total_fees'], finansial['fee_ratio']]

    return metrics, equity_curve

@st.cache_data
def get_compounding_history(architecture, variant_upper, sl_pct, tp_pct):
    y_statis = [INITIAL_CAPITAL]
    y_berkala = [INITIAL_CAPITAL]
    
    for period in range(1, TOTAL_PERIODS + 1):
        s_met, _ = evaluate_ensemble_scenario_full(architecture, variant_upper, 'statis', period, sl_pct, tp_pct)
        p_met, _ = evaluate_ensemble_scenario_full(architecture, variant_upper, 'berkala', period, sl_pct, tp_pct)
        
        roi_s = s_met[4] if s_met else 0.0  
        roi_p = p_met[4] if p_met else 0.0
        
        y_statis.append(y_statis[-1] * (1 + roi_s / 100.0))
        y_berkala.append(y_berkala[-1] * (1 + roi_p / 100.0))
        
    return y_statis, y_berkala

@st.cache_data
def get_buy_and_hold_roi_dynamic(target_period):
    """Murni hold pasar (SL & TP = 0%)"""
    periods_to_run = range(1, TOTAL_PERIODS + 1) if target_period == 0 else [target_period]
    all_returns, all_lows, all_highs = [], [], []
    
    for p in periods_to_run:
        dir_path = os.path.join(MODEL_DIRECTORY, "XGB", "XGB_MURNI", "Arsip_Ujian_Silang", f"Model_{p:02d}_di_Masa_{p:02d}")
        pred_file = os.path.join(dir_path, 'database_prediksi_mentah.csv')
        if os.path.exists(pred_file):
            df = pd.read_csv(pred_file)
            all_returns.extend(df['wasit_target'].values)
            all_lows.extend(df['wasit_low'].values)
            all_highs.extend(df['wasit_high'].values)
            
    if not all_returns:
        return 0.0
        
    pct_returns = np.exp(np.array(all_returns) / 100.0) - 1
    low_returns_pct = np.exp(np.array(all_lows) / 100.0) - 1
    high_returns_pct = np.exp(np.array(all_highs) / 100.0) - 1
    preds_bnh = np.ones(len(all_returns), dtype=int)
    
    finansial = simulate_financial_hold(preds_bnh, pct_returns, low_returns_pct, high_returns_pct, sl_pct=0.0, tp_pct=0.0)
    return finansial['roi']

compact_metrics_cols = [
    'Akurasi (%)', 'F1-Macro (%)', 'F1-Long (%)', 'F1-Short (%)',
    'ROI (%)', 'Sharpe', 'Sortino', 'MDD (%)',
    'Profit Factor', 'Win Rate (%)', 'Total Trade', 'Total Fee ($)', 'Fee / Modal (%)'
]

@st.cache_data
def load_all_dashboard_data(target_period, sl_pct, tp_pct):
    static_data, periodic_data, row_names = [], [], []
    equity_dict = {} 
    
    for arch in ARCHITECTURES:
        for comb, label in COMBINATIONS:
            variant = "_".join(comb) if comb else "murni"
            vdir = variant.upper()
            model_name = f"{arch} - {label}"
            
            s_metrics, s_eq = evaluate_ensemble_scenario_full(arch, vdir, 'statis', target_period, sl_pct, tp_pct)
            p_metrics, p_eq = evaluate_ensemble_scenario_full(arch, vdir, 'berkala', target_period, sl_pct, tp_pct)
            
            if p_metrics is not None:
                row_names.append(model_name)
                static_data.append(s_metrics)
                periodic_data.append(p_metrics)
                equity_dict[f"{model_name} (Statis)"] = s_eq
                equity_dict[f"{model_name} (Berkala)"] = p_eq
                
    df_static = pd.DataFrame(static_data, index=row_names, columns=compact_metrics_cols)
    df_periodic = pd.DataFrame(periodic_data, index=row_names, columns=compact_metrics_cols)
    
    df_diff = pd.DataFrame()
    if not df_periodic.empty and not df_static.empty:
        df_diff = df_periodic - df_static
        
    return df_static, df_periodic, df_diff, equity_dict

@st.cache_data
def build_prediction_database():
    records = []
    for arch in ARCHITECTURES:
        for comb, label in COMBINATIONS:
            variant_upper = "_".join(comb).upper() if comb else "MURNI"
            for mode in ['statis', 'berkala']:
                for eval_period in range(1, TOTAL_PERIODS + 1):
                    model_id = 0 if mode == 'statis' else eval_period
                    dir_path = os.path.join(MODEL_DIRECTORY, arch, f"{arch}_{variant_upper}",
                                            "Arsip_Ujian_Silang", f"Model_{model_id:02d}_di_Masa_{eval_period:02d}")
                    pred_file = os.path.join(dir_path, 'database_prediksi_mentah.csv')
                    report_file = os.path.join(dir_path, 'database_juri_lengkap.csv')
                    
                    if not (os.path.exists(pred_file) and os.path.exists(report_file)):
                        continue
                        
                    pred_df = pd.read_csv(pred_file)
                    report_df = pd.read_csv(report_file)
                    top_estimators = report_df.sort_values(by='f1_macro', ascending=False)['nama_juri'].head(OPTIMAL_ESTIMATOR_COUNT).tolist()
                    avg_prob = pred_df[top_estimators].mean(axis=1)
                    decisions = (avg_prob >= 0.50).astype(int)
                    
                    if 'tanggal' in pred_df.columns:
                        dates = pred_df['tanggal'].values
                    else:
                        continue 
                        
                    for d, prob, sig in zip(dates, avg_prob, decisions):
                        records.append({
                            'Tanggal': d,
                            'Arsitektur': arch,
                            'Data': label,
                            'Mode': mode,
                            'Probabilitas': prob,
                            'Sinyal': sig
                        })
    return pd.DataFrame(records)

def create_candlestick_with_signals(eq_df, raw_btc_df, model_name):
    eq_df = eq_df.copy()
    eq_df['Tanggal'] = pd.to_datetime(eq_df['Tanggal'], errors='coerce')
    eq_df = eq_df.dropna(subset=['Tanggal'])
    
    merged_df = pd.merge(eq_df, raw_btc_df, left_on='Tanggal', right_on='timestamp', how='inner')
    
    if merged_df.empty:
        return None
        
    merged_df['Signal_Change'] = merged_df['Signal'].diff()
    if merged_df['Signal'].iloc[0] == 1:
        merged_df.at[merged_df.index[0], 'Signal_Change'] = 1

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(go.Candlestick(
        x=merged_df['Tanggal'],
        open=merged_df['open'], high=merged_df['high'],
        low=merged_df['low'], close=merged_df['close'],
        name='Harga BTC', increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
    ), secondary_y=False)
    
    buy_signals = merged_df[merged_df['Signal_Change'] == 1]
    fig.add_trace(go.Scatter(
        x=buy_signals['Tanggal'], y=buy_signals['low'] * 0.95, 
        mode='markers', marker=dict(symbol='triangle-up', size=14, color='#00ff00', line=dict(width=1, color='black')), 
        name='Sinyal LONG (Buy)'
    ), secondary_y=False)

    sell_signals = merged_df[merged_df['Signal_Change'] == -1]
    fig.add_trace(go.Scatter(
        x=sell_signals['Tanggal'], y=sell_signals['high'] * 1.05, 
        mode='markers', marker=dict(symbol='triangle-down', size=14, color='#ff0000', line=dict(width=1, color='black')), 
        name='Sinyal EXIT (Sell/Hold)'
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=merged_df['Tanggal'], y=merged_df['Balance'], 
        mode='lines', line=dict(color='#2962ff', width=2), 
        name='Saldo Uang (USD)'
    ), secondary_y=True)

    fig.update_layout(
        title=f"Analisis Teknikal & Performa Modal: {model_name}",
        xaxis_title="Waktu", xaxis_rangeslider_visible=False,
        template="plotly_dark", hovermode="x unified", height=650,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="Harga Bitcoin (USD)", secondary_y=False)
    fig.update_yaxes(title_text="Total Saldo (USD)", secondary_y=True, showgrid=False)
    
    return fig

# ==========================================
# UI STREAMLIT
# ==========================================
st.title("📈 Dashboard Evaluasi Model Kuantitatif (Bitcoin)")

# --- SIDEBAR PENGATURAN ---
st.sidebar.header("⚙️ Pengaturan Evaluasi")

period_options = {"Gabungan Seluruh Masa": 0}
for p in range(1, TOTAL_PERIODS + 1):
    period_options[f"Isolasi Masa {p}"] = p

selected_period_label = st.sidebar.selectbox("📅 Pilih Area Masa:", list(period_options.keys()))
current_target_period = period_options[selected_period_label]

st.sidebar.markdown("---")
st.sidebar.markdown("**Manajemen Risiko**")
sl_input = st.sidebar.slider("Batas Stop Loss (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.5, 
                             help="Geser ke 0.0% jika tidak ingin menggunakan Stop Loss.")
tp_input = st.sidebar.slider("Batas Take Profit (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.5, 
                             help="Geser ke 0.0% jika tidak ingin menggunakan Take Profit.")

sl_pct = sl_input / 100.0
tp_pct = tp_input / 100.0

# Muat data
df_static, df_periodic, df_diff, equity_dict = load_all_dashboard_data(current_target_period, sl_pct, tp_pct)
raw_btc_df = load_raw_btc_data()

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Metrik Utama (Heatmap)", 
    "📈 Kurva Ekuitas", 
    "⚔️ Statis vs Berkala", 
    "🥊 Adu Model A vs B",
    "📊 Komparasi ROI",
    "🕯️ Chart Trading",
    "📅 Scanner Harian"
])

col_cmap = 'RdYlGn'

# --- TAB 1: METRIK UTAMA SIDE-BY-SIDE ---
with tab1:
    st.header(f"Dashboard Metrik - {selected_period_label}")
    if sl_pct > 0 or tp_pct > 0:
        st.info(f"🛡️ **Proteksi Aktif:** Simulasi dijalankan dengan batas Stop Loss **{sl_input}%** dan Take Profit **{tp_input}%** dari titik masuk (Entry).")
        
    if not df_periodic.empty:
        sort_metric = st.selectbox("Urutkan (Ranking) Berdasarkan:", compact_metrics_cols, index=1)
        
        df_periodic_sorted = df_periodic.sort_values(by=sort_metric, ascending=False)
        df_static_sorted = df_static.sort_values(by=sort_metric, ascending=False)
        df_diff_sorted = df_diff.reindex(df_periodic_sorted.index)

        st.subheader("1. Varian Berkala (Diperbarui setiap masa)")
        st.dataframe(df_periodic_sorted.style.background_gradient(cmap=col_cmap, axis=0).format(precision=2), use_container_width=True)
        st.subheader("2. Varian Statis (Hanya dilatih di Masa 00)")
        st.dataframe(df_static_sorted.style.background_gradient(cmap=col_cmap, axis=0).format(precision=2), use_container_width=True)
        st.subheader("3. Selisih Performa (Berkala - Statis)")
        st.dataframe(df_diff_sorted.style.background_gradient(cmap=col_cmap, axis=0).format(precision=2), use_container_width=True)
    else:
        st.warning("Data tidak tersedia.")

# --- TAB 2: GRAFIK KURVA EKUITAS ---
with tab2:
    st.header(f"Grafik Pergerakan Ekuitas Harian ({selected_period_label})")
    if equity_dict:
        selected_models_eq = st.multiselect("Pilih Model untuk Ditampilkan di Kurva:", 
                                            options=list(equity_dict.keys()), 
                                            default=[list(equity_dict.keys())[1]]) 
        if selected_models_eq:
            fig = go.Figure()
            for model_sel in selected_models_eq:
                eq_df = equity_dict[model_sel]
                fig.add_trace(go.Scatter(x=eq_df['Tanggal'], y=eq_df['Balance'], mode='lines', name=model_sel))
                
            fig.update_layout(
                title=f"Pertumbuhan Ekuitas - {selected_period_label}",
                xaxis_title="Garis Waktu Harian", yaxis_title="Total Saldo (USD)",
                hovermode="x unified", template="plotly_white"
            )
            fig.add_hline(y=INITIAL_CAPITAL, line_dash="dash", line_color="gray", annotation_text="Modal Awal")
            st.plotly_chart(fig, use_container_width=True)

# --- TAB 3: STATIS VS BERKALA ---
with tab3:
    st.header("Adu Performa: Statis vs Berkala (Kombinasi yang Sama)")
    if not df_periodic.empty:
        model_choice = st.selectbox("Pilih Kombinasi Arsitektur & Data:", df_periodic.index, key="sb_statis_berkala")
        st.markdown(f"### Komparasi Langsung: **{model_choice}**")
        
        comp_df = pd.DataFrame({
            'Statis': df_static.loc[model_choice],
            'Berkala': df_periodic.loc[model_choice],
            'Selisih': df_diff.loc[model_choice]
        })
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.dataframe(comp_df.style.format(precision=2), use_container_width=True)
        with col2:
            key_metrics = ['F1-Macro (%)', 'ROI (%)', 'Win Rate (%)']
            bar_df = comp_df.loc[key_metrics, ['Statis', 'Berkala']].reset_index().melt(id_vars='index')
            fig2 = px.bar(bar_df, x='index', y='value', color='variable', barmode='group', 
                          title=f"Metrik Kunci ({selected_period_label})", labels={'index': 'Metrik', 'value': 'Nilai', 'variable': 'Varian'})
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")
        st.markdown("### Grafik Garis Perjalanan Modal (Compounding) Lintas 5 Masa")
        
        arch_part, data_part = model_choice.split(" - ")
        arch_choice = arch_part.strip()
        comb_str = data_part.replace("+ ", "").lower()
        variant_upper = "MURNI" if "baseline" in comb_str else comb_str.upper()
        
        y_stat, y_berk = get_compounding_history(arch_choice, variant_upper, sl_pct, tp_pct)
        x_labels = ["Start"] + [f"Masa {p}" for p in range(1, TOTAL_PERIODS + 1)]
        
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(x=x_labels, y=y_stat, mode='lines+markers', name='Statis (Model 00)', marker=dict(size=10), line=dict(color='#d62728', width=3)))
        fig_line.add_trace(go.Scatter(x=x_labels, y=y_berk, mode='lines+markers', name='Berkala', marker=dict(symbol='square', size=10), line=dict(color='#2ca02c', width=3)))
        
        fig_line.update_layout(title=f"Adu Compounding Modal: {model_choice}", yaxis_title="Total Uang (USD)", template="plotly_white", hovermode="x unified")
        fig_line.add_hline(y=INITIAL_CAPITAL, line_dash="dash", line_color="gray", annotation_text="Modal Awal")
        st.plotly_chart(fig_line, use_container_width=True)

# --- TAB 4: ADU MODEL BERKALA A VS B ---
with tab4:
    st.header(f"Adu Performa: Kubu A vs Kubu B ({selected_period_label})")
    if not df_periodic.empty:
        colA, colB = st.columns(2)
        with colA:
            model_A = st.selectbox("Pilih Model A (Kubu Merah):", df_periodic.index, index=0)
        with colB:
            model_B = st.selectbox("Pilih Model B (Kubu Biru):", df_periodic.index, index=1 if len(df_periodic.index)>1 else 0)
            
        st.markdown("---")
        radar_metrics = ['Akurasi (%)', 'F1-Macro (%)', 'Win Rate (%)', 'ROI (%)']
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=df_periodic.loc[model_A, radar_metrics].values, theta=radar_metrics, fill='toself', name=f'{model_A}'))
        fig_radar.add_trace(go.Scatterpolar(r=df_periodic.loc[model_B, radar_metrics].values, theta=radar_metrics, fill='toself', name=f'{model_B}'))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True, title="Komparasi Keseimbangan Metrik Kunci")
        
        col_rad, col_tab = st.columns([1, 1])
        with col_rad:
            st.plotly_chart(fig_radar, use_container_width=True)
        with col_tab:
            st.markdown(f"**Tabel Adu Mekanik Detail ({selected_period_label})**")
            battle_df = pd.DataFrame({
                model_A: df_periodic.loc[model_A], model_B: df_periodic.loc[model_B],
                'Selisih (A - B)': df_periodic.loc[model_A] - df_periodic.loc[model_B]
            })
            st.dataframe(battle_df.style.format(precision=2), use_container_width=True)

# --- TAB 5: KOMPARASI ROI (BAR CHART) ---
with tab5:
    st.header(f"Tabel dan Grafik Perjalanan ROI ({selected_period_label})")
    if not df_periodic.empty:
        bnh_roi = get_buy_and_hold_roi_dynamic(current_target_period)
        model_names = ["Buy & Hold (Murni)"] + list(df_periodic.index)
        roi_values = [bnh_roi] + list(df_periodic['ROI (%)'].values)
        
        df_roi_compare = pd.DataFrame({'Model': model_names, 'ROI (%)': roi_values})
        bar_colors = ['#808080'] + ['#2ca02c' if (pd.notna(r) and r > 0) else '#d62728' for r in roi_values[1:]]
        
        fig_bar = go.Figure(data=[go.Bar(
            x=df_roi_compare['Model'], y=df_roi_compare['ROI (%)'],
            marker_color=bar_colors, text=[f"{val:.2f}%" for val in df_roi_compare['ROI (%)']], textposition='outside'
        )])
        fig_bar.update_layout(title=f"Perbandingan ROI Model Berkala vs Buy & Hold", xaxis_title="Jenis Model", yaxis_title="ROI (%)", xaxis_tickangle=-45, template="plotly_white", height=600)
        
        st.plotly_chart(fig_bar, use_container_width=True)
        st.dataframe(df_roi_compare.style.format({'ROI (%)': '{:.2f}'}), use_container_width=True)

# --- TAB 6: CHART TRADING & SINYAL (CANDLESTICK) ---
with tab6:
    st.header(f"Visualisasi Sinyal Trading & Harga Pasar ({selected_period_label})")
    if raw_btc_df is None:
        st.error(f"File data harga BTC mentah tidak ditemukan di direktori: `{DATA_BTC_RAW}`.")
    elif equity_dict:
        selected_trading_models = st.multiselect("Pilih Model (Grafik akan disusun memanjang ke bawah):", options=list(equity_dict.keys()), default=[list(equity_dict.keys())[1]])
        if selected_trading_models:
            for model_name in selected_trading_models:
                eq_df = equity_dict[model_name]
                fig_trade = create_candlestick_with_signals(eq_df, raw_btc_df, model_name)
                if fig_trade:
                    st.plotly_chart(fig_trade, use_container_width=True)
                    st.markdown("---") 

# --- TAB 7: SCANNER PREDIKSI HARIAN (3x5 MATRIX) ---
with tab7:
    st.header("🔮 Scanner Suara Mayoritas Model (Harian)")
    df_preds = build_prediction_database()
    if not df_preds.empty and raw_btc_df is not None:
        df_preds['Tanggal_DT'] = pd.to_datetime(df_preds['Tanggal'], errors='coerce')
        raw_btc_df['timestamp_DT'] = pd.to_datetime(raw_btc_df['timestamp'], errors='coerce')
        df_preds = df_preds.dropna(subset=['Tanggal_DT'])
        
        min_date = df_preds['Tanggal_DT'].min().date()
        max_date = df_preds['Tanggal_DT'].max().date()
        
        selected_date = st.date_input("📅 Pilih Tanggal Observasi:", value=max_date, min_value=min_date, max_value=max_date)
        selected_date_dt = pd.to_datetime(selected_date)
        
        if selected_date_dt not in df_preds['Tanggal_DT'].values:
            nearest_idx = (df_preds['Tanggal_DT'] - selected_date_dt).abs().argmin()
            selected_date_dt = df_preds['Tanggal_DT'].iloc[nearest_idx]
            st.info(f"Data prediksi untuk {selected_date} kosong. Otomatis dialihkan ke tanggal terdekat: **{selected_date_dt.date()}**")
            selected_date = selected_date_dt.date()
            
        st.markdown(f"### 📊 Konteks Pasar: Posisi Harga pada {selected_date}")
        zoom_start = selected_date_dt - pd.Timedelta(days=30)
        zoom_end = selected_date_dt + pd.Timedelta(days=30)
        df_zoom = raw_btc_df[(raw_btc_df['timestamp_DT'] >= zoom_start) & (raw_btc_df['timestamp_DT'] <= zoom_end)]
        
        fig_context = go.Figure(data=[go.Candlestick(
            x=df_zoom['timestamp_DT'], open=df_zoom['open'], high=df_zoom['high'],
            low=df_zoom['low'], close=df_zoom['close'], name='Harga BTC', increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
        )])
        fig_context.add_vline(x=selected_date_dt.strftime('%Y-%m-%d'), line_width=2, line_dash="dash", line_color="yellow", annotation_text="Titik Observasi", annotation_position="top right")
        fig_context.update_layout(template="plotly_dark", height=400, margin=dict(l=0, r=0, t=30, b=0), yaxis_title="Harga Bitcoin (USD)", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_context, use_container_width=True)
        
        st.markdown("---")
        df_day = df_preds[df_preds['Tanggal_DT'] == selected_date_dt]
        
        def format_prediction_grid(df_subset):
            grid = pd.DataFrame(index=[c[1] for c in COMBINATIONS], columns=ARCHITECTURES)
            for _, row in df_subset.iterrows():
                arch = row['Arsitektur']
                data = row['Data']
                prob = row['Probabilitas'] * 100
                sig = "📈 NAIK" if row['Sinyal'] == 1 else "📉 TURUN"
                grid.at[data, arch] = f"{sig} ({prob:.1f}%)"
            return grid

        def highlight_cells(val):
            if pd.isna(val):
                return ''
            if 'NAIK' in str(val):
                return 'background-color: rgba(38, 166, 154, 0.2); color: #137333; font-weight: bold'
            elif 'TURUN' in str(val):
                return 'background-color: rgba(239, 83, 80, 0.2); color: #c5221f; font-weight: bold'
            return ''

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Kubu Berkala (Dinamis)")
            df_day_berkala = df_day[df_day['Mode'] == 'berkala']
            grid_berkala = format_prediction_grid(df_day_berkala)
            st.dataframe(grid_berkala.style.map(highlight_cells), use_container_width=True)
            
        with col2:
            st.markdown("### Kubu Statis (Dilatih di Awal)")
            df_day_statis = df_day[df_day['Mode'] == 'statis']
            grid_statis = format_prediction_grid(df_day_statis)
            st.dataframe(grid_statis.style.map(highlight_cells), use_container_width=True)
