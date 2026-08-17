import os
import json
import math
import joblib
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from scipy import stats

# ============================================================
# 1. Pure PyTorch Mamba (CPU & Cloud Compatible)
# ============================================================
class Mamba(nn.Module):
    """Pure PyTorch Mamba implementation for non-CUDA / Streamlit Cloud hosting."""
    def __init__(self, d_model, d_state=16, d_conv=2, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = torch.repeat_interleave(
            torch.arange(1, self.d_state + 1, dtype=torch.float32), self.d_inner
        ).view(self.d_inner, self.d_state)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, x):
        b, l, d = x.shape
        x_and_res = self.in_proj(x)
        x_proj, res = x_and_res.chunk(2, dim=-1)

        x_conv = self.conv1d(x_proj.transpose(1, 2))[:, :, :l].transpose(1, 2)
        x_conv = F.silu(x_conv)

        y = self.ssm(x_conv)
        y = y * F.silu(res)
        return self.out_proj(y)

    def ssm(self, x):
        b, l, d = x.shape
        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        x_dbl = self.x_proj(x)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))

        y = torch.zeros_like(x)
        h = torch.zeros(b, self.d_inner, self.d_state, device=x.device)

        for t in range(l):
            dt_t = dt[:, t, :].unsqueeze(-1)
            A_dt = torch.exp(A.unsqueeze(0) * dt_t)
            B_t = B[:, t, :].unsqueeze(1)
            x_t = x[:, t, :].unsqueeze(-1)

            h = h * A_dt + (x_t * B_t) * dt_t
            C_t = C[:, t, :].unsqueeze(-1)
            y[:, t, :] = torch.matmul(h, C_t).squeeze(-1) + x[:, t, :] * D

        return y


# ============================================================
# 2. Model Architecture
# ============================================================
class Add_Norm(nn.Module):
    def __init__(self, d_model, dropout, residual=True):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.residual = residual

    def forward(self, new, old):
        if self.residual:
            return self.norm(old + self.dropout(new))
        return self.norm(self.dropout(new))


class BimambaEncoderLayer(nn.Module):
    def __init__(self, d_model, d_conv, d_state, expand,
                 b_d_conv, b_d_state, b_expand, dropout, d_ff, residual=True):
        super().__init__()
        self.mamba_forward = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_backward = Mamba(d_model=d_model, d_state=b_d_state, d_conv=b_d_conv, expand=b_expand)
        self.norm = nn.LayerNorm(d_model)
        self.addnorm1 = Add_Norm(d_model, dropout, residual=False)
        self.addnorm2 = Add_Norm(d_model, dropout, residual=False)
        self.addnorm3 = Add_Norm(d_model, dropout, residual=residual)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.addnorm4 = Add_Norm(d_model, dropout, residual=residual)

    def forward(self, x):
        x_norm = self.norm(x)
        out_fwd = self.mamba_forward(x_norm)
        out_fwd = self.addnorm1(out_fwd, x)

        rev_input = x_norm.flip(dims=[1])
        out_bwd = self.mamba_backward(rev_input).flip(dims=[1])
        out_bwd = self.addnorm2(out_bwd, x)

        out = self.addnorm3(out_fwd + out_bwd, x)
        ffn_out = self.ffn(out)
        return self.addnorm4(ffn_out, out)


class StockMambaEncoder(nn.Module):
    def __init__(self, input_features=5, d_model=64, n_layer=2,
                 d_conv=2, d_state=16, expand=2,
                 b_d_conv=2, b_d_state=16, b_expand=2,
                 dropout=0.1, d_ff=256):
        super().__init__()
        self.input_proj = nn.Linear(input_features, d_model)
        self.layers = nn.ModuleList([
            BimambaEncoderLayer(d_model, d_conv, d_state, expand, b_d_conv, b_d_state, b_expand, dropout, d_ff)
            for _ in range(n_layer)
        ])

    def forward(self, x):
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        return x


class StockMambaDecoderLayer(nn.Module):
    def __init__(self, d_model, d_conv, d_state, expand, n_heads, dropout, d_ff):
        super().__init__()
        self.causal_mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, target, encoder_output):
        residual = target
        target = self.norm1(target)
        target = residual + self.dropout1(self.causal_mamba(target))

        residual = target
        target = self.norm2(target)
        attn_out, attn_weights = self.cross_attn(query=target, key=encoder_output, value=encoder_output)
        target = residual + self.dropout2(attn_out)

        residual = target
        target = self.norm3(target)
        target = residual + self.dropout3(self.ffn(target))
        return target, attn_weights


class StockMambaDecoder(nn.Module):
    def __init__(self, d_model=64, d_conv=2, d_state=16, expand=2, n_heads=4, n_layer=2,
                 dropout=0.1, d_ff=256, input_dim=1, output_dim=1, pred_length=1):
        super().__init__()
        self.pred_length = pred_length
        self.input_proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([
            StockMambaDecoderLayer(d_model, d_conv, d_state, expand, n_heads, dropout, d_ff)
            for _ in range(n_layer)
        ])
        self.output_proj = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, output_dim))

    def forward(self, target, encoder_output):
        x = self.input_proj(target)
        attn_weights = []
        for layer in self.layers:
            x, attn = layer(x, encoder_output)
            attn_weights.append(attn)
        predictions = self.output_proj(x)
        return predictions.squeeze(-1), attn_weights


class StockMambaModel(nn.Module):
    def __init__(self, input_features=5, d_model=64, n_layers_enc=2, n_layers_dec=2,
                 n_heads=4, dropout=0.1, d_ff=256, pred_length=1):
        super().__init__()
        self.encoder = StockMambaEncoder(
            input_features=input_features, d_model=d_model, n_layer=n_layers_enc, dropout=dropout, d_ff=d_ff
        )
        self.decoder = StockMambaDecoder(
            d_model=d_model, n_heads=n_heads, n_layer=n_layers_dec, dropout=dropout, d_ff=d_ff, pred_length=pred_length
        )
        self.pred_length = pred_length

    def forward(self, x_batch, y_teacher=None):
        enc_out = self.encoder(x_batch)
        last_close = x_batch[:, -1:, 3:4]

        if y_teacher is not None and self.pred_length > 1:
            decoder_input = torch.cat([last_close, y_teacher[:, :-1].unsqueeze(-1)], dim=1)
        else:
            decoder_input = last_close

        preds, _ = self.decoder(decoder_input, enc_out)
        return preds


# ============================================================
# 3. Peer Universe (used for t-test similarity analysis)
# ============================================================
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "AMD", "ADBE",
    "CRM", "ORCL", "INTC", "CSCO", "QCOM", "TXN", "IBM", "NOW", "INTU", "AMAT",
    "JPM", "BAC", "WFC", "GS", "MS", "V", "MA", "AXP", "SPGI", "BLK",
    "JNJ", "PFE", "UNH", "MRK", "ABBV", "LLY", "TMO", "ABT", "BMY", "GILD",
    "XOM", "CVX", "PG", "KO", "PEP", "WMT", "HD", "MCD", "DIS", "NKE",
]


@st.cache_data(ttl=3600)
def fetch_peer_returns(tickers, period="1y"):
    """Fetch daily % returns for a list of tickers over a fixed lookback period."""
    data = yf.download(tickers, period=period, progress=False, auto_adjust=True)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers[0])
    returns = data.pct_change().dropna(how="all")
    return returns


# ============================================================
# 4. Streamlit Interface
# ============================================================
@st.cache_resource
def load_artifacts(artifact_dir="artifacts"):
    device = torch.device("cpu")
    
    with open(os.path.join(artifact_dir, "config.json"), "r") as f:
        config = json.load(f)
        
    model = StockMambaModel(
        input_features=len(config["feature_cols"]),
        d_model=config["d_model"],
        n_layers_enc=config["n_layers_enc"],
        n_layers_dec=config["n_layers_dec"],
        n_heads=config["n_heads"],
        dropout=config["dropout"],
        d_ff=config["d_ff"],
        pred_length=config["pred_length"]
    ).to(device)
    
    weights_path = os.path.join(artifact_dir, "bimamba_sp500_inference.pth")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    feature_scalers = joblib.load(os.path.join(artifact_dir, "feature_scalers.pkl"))
    target_scalers = joblib.load(os.path.join(artifact_dir, "target_scalers.pkl"))
    
    return model, feature_scalers, target_scalers, config, device


@st.cache_data(ttl=3600)
def fetch_historical_data(ticker):
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=4 * 365)
    
    df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
    if df.empty:
        return None
        
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    return df


st.set_page_config(page_title="BiMamba Stock Forecaster", layout="wide")

st.title("📈 BiMamba Stock Price Forecaster")
st.markdown("Generates a **1-Year (252 Trading Days)** forecast using the trained Hybrid BiMamba Encoder-Decoder model[cite: 1].")

st.sidebar.header("Inputs")
ticker_input = st.sidebar.text_input("Enter Stock Ticker:", value="AAPL").upper().strip()
artifact_dir = st.sidebar.text_input("Artifacts Directory:", value="artifacts")
forecast_days = st.sidebar.slider("Forecast Horizon (Trading Days):", min_value=30, max_value=252, value=252)

run_btn = st.sidebar.button("Predict Stock Price", type="primary")

try:
    model, feature_scalers, target_scalers, config, device = load_artifacts(artifact_dir)
    st.sidebar.success("Artifacts loaded successfully.")
except Exception as e:
    st.error(f"Error loading artifacts from `{artifact_dir}`: {e}")
    st.stop()

if run_btn or ticker_input:
    with st.spinner(f"Fetching 4-year historical data for '{ticker_input}'..."):
        df_hist = fetch_historical_data(ticker_input)

    if df_hist is None or len(df_hist) < config["seq_length"]:
        st.error(f"Could not retrieve sufficient trading data for ticker '{ticker_input}'.")
    else:
        st.success(f"Retrieved {len(df_hist)} trading days up to {df_hist.index[-1].strftime('%Y-%m-%d')}.")
        
        if ticker_input in feature_scalers and ticker_input in target_scalers:
            f_scaler = feature_scalers[ticker_input]
            t_scaler = target_scalers[ticker_input]
            st.info(f"Using pre-trained scalers for **{ticker_input}**.")
        else:
            st.warning(f"Ticker **'{ticker_input}'** not found in training scalers[cite: 1]. Dynamically fitting scalers on historical data.")
            f_scaler = StandardScaler().fit(df_hist[config["feature_cols"]].values)
            t_scaler = StandardScaler().fit(df_hist[[config["target_col"]]].values)

        seq_len = config["seq_length"]
        current_window = df_hist[config["feature_cols"]].iloc[-seq_len:].values.copy()
        future_predictions = []
        
        with st.spinner(f"Generating forecast for {forecast_days} trading days..."):
            for _ in range(forecast_days):
                scaled_window = f_scaler.transform(current_window)
                x_tensor = torch.tensor(scaled_window, dtype=torch.float32).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    pred_scaled = model(x_tensor).cpu().numpy().ravel()[0]
                
                pred_price = t_scaler.inverse_transform([[pred_scaled]])[0, 0]
                future_predictions.append(pred_price)
                
                last_volume = current_window[-1, 4]
                next_row = np.array([pred_price, pred_price, pred_price, pred_price, last_volume])
                current_window = np.vstack([current_window[1:], next_row])

        last_date = df_hist.index[-1]
        future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
        df_forecast = pd.DataFrame({"Predicted Close": future_predictions}, index=future_dates)
        
        st.subheader(f"Historical & {forecast_days}-Day Forecast Plot for {ticker_input}")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_hist.index, y=df_hist["close"], mode="lines", name="Historical Close", line=dict(color="#1E88E5", width=2)
        ))
        fig.add_trace(go.Scatter(
            x=df_forecast.index, y=df_forecast["Predicted Close"], mode="lines", name="BiMamba Forecast", line=dict(color="#FF6D00", width=2.5, dash="dash")
        ))
        fig.update_layout(xaxis_title="Date", yaxis_title="Price (USD)", hovermode="x unified", template="plotly_dark", height=550)
        st.plotly_chart(fig, use_container_width=True)

        last_known_price = df_hist["close"].iloc[-1]
        final_forecast_price = future_predictions[-1]
        projected_return = ((final_forecast_price - last_known_price) / last_known_price) * 100
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Last Known Close", f"${last_known_price:.2f}")
        c2.metric("Predicted Close", f"${final_forecast_price:.2f}")
        c3.metric("Projected Return", f"{projected_return:+.2f}%")

# ============================================================
# Peer Similarity Analysis (t-test on daily returns)
# ============================================================
st.markdown("---")
st.subheader("🔬 Peer Similarity Analysis (t-test on Daily Returns)")
st.markdown(
    f"Runs an independent two-sample **t-test (Welch's)** comparing **{ticker_input}**'s daily "
    "% returns over the last 1 year against each of the 50 large-cap tickers below. "
    "The **p-value** is used as the similarity number: a **higher p-value** means we cannot "
    "statistically distinguish the two stocks' average daily returns (more *similar*), while a "
    "**low p-value** (e.g. < 0.05) means their average returns are significantly different."
)

run_similarity = st.button("Run Similarity Analysis")

if run_similarity:
    peer_list = [t for t in TICKERS if t != ticker_input]
    all_tickers_for_fetch = list(dict.fromkeys(peer_list + [ticker_input]))

    with st.spinner(f"Fetching 1-year daily return data for {ticker_input} and {len(peer_list)} peers..."):
        try:
            peer_returns_df = fetch_peer_returns(all_tickers_for_fetch, period="1y")
        except Exception as e:
            peer_returns_df = None
            st.error(f"Error fetching peer return data: {e}")

    if peer_returns_df is None or ticker_input not in peer_returns_df.columns:
        st.error(f"Could not retrieve 1-year daily return data for '{ticker_input}'.")
    else:
        target_returns = peer_returns_df[ticker_input].dropna()

        results = []
        for peer in peer_list:
            if peer not in peer_returns_df.columns:
                continue
            peer_returns = peer_returns_df[peer].dropna()
            if len(peer_returns) < 10 or len(target_returns) < 10:
                continue
            t_stat, p_value = stats.ttest_ind(
                target_returns, peer_returns, equal_var=False, nan_policy="omit"
            )
            results.append({
                "Ticker": peer,
                "t-statistic": t_stat,
                "Similarity (p-value)": p_value,
            })

        if not results:
            st.warning("No comparable peer return data could be retrieved.")
        else:
            df_results = (
                pd.DataFrame(results)
                .sort_values("Similarity (p-value)", ascending=False)
                .reset_index(drop=True)
            )
            df_results.index += 1

            st.dataframe(
                df_results.style.format({
                    "t-statistic": "{:.4f}",
                    "Similarity (p-value)": "{:.4f}",
                }),
                use_container_width=True,
            )

            most_similar = df_results.iloc[0]
            least_similar = df_results.iloc[-1]
            m1, m2 = st.columns(2)
            m1.success(
                f"Most similar peer: **{most_similar['Ticker']}** "
                f"(p-value = {most_similar['Similarity (p-value)']:.4f})"
            )
            m2.info(
                f"Least similar peer: **{least_similar['Ticker']}** "
                f"(p-value = {least_similar['Similarity (p-value)']:.4f})"
            )