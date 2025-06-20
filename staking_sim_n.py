import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

MAX_SUPPLY      = 100_000_000
EMIT_PER_MONTH  = 3_330_000
DAO_SPLIT       = 0.20
SCENARIOS       = {"Bear": 0.05, "Base": 0.125, "Bull": 0.30}
ADTV_TOKENS     = 1_000_000      # assumed average daily trading volume (tokens)

# ───────────────────────── helper functions ──────────────────────────────
gbm_step = lambda p, μ, σ: p * np.exp((μ - 0.5*σ**2)/12 + σ*np.sqrt(1/12)*np.random.randn())
revenue_needed = lambda apy, stake_tok, price: ((1+apy)**(1/12) - 1) * stake_tok * price

def guardian_price_factor(sold):
    if sold == 0:
        return 1.0
    pct = sold / ADTV_TOKENS * 100
    if   pct <= 1:   b = 0.995 - pct*0.005
    elif pct <= 2:   b = 0.990 - (pct-1)*0.015
    elif pct <= 3:   b = 0.980 - (pct-2)*0.03
    elif pct <= 5:   b = 0.950 - (pct-3)*0.05
    elif pct <=10:   b = 0.850 - (pct-5)*0.04
    elif pct <=20:   b = 0.650 - (pct-10)*0.025
    else:            b = max(0.1, 0.400 - (pct-20)*0.015)
    return max(0.1, b)

# ───────────────────────────── UI sidebar ────────────────────────────────
st.title("Validator – Guardian Staking Simulator")
sb = st.sidebar

sb.header("Timeline & price path")
months   = sb.slider("Months to simulate", 6, 120, 36, 6)
price0   = sb.number_input("Initial token price $", 0.01, 10.0, 0.05, 0.005, format="%.3f")
mu       = sb.number_input("GBM drift μ (annual)", -1.0, 2.0, 0.30, 0.05)
sigma    = sb.number_input("GBM volatility σ (annual)", 0.01, 3.0, 1.0, 0.05)
rng_seed = sb.number_input("Random seed", 0, 2**31-1, 1)

sb.header("DAO buy-back & burn")
burn_frac = sb.slider("Burn: fraction of DAO cash each month", 0.0, 1.0, 0.25, 0.05)

sb.header("Misbehaviour / slashing")
p_mis     = sb.slider("Prob. a validator misbehaves (per month)", 0.0, 0.5, 0.05, 0.01)
slash_amt = sb.number_input("Slash amount (tokens)", 1.0, 1_000_000.0, 500.0, 10.0)

sb.header("Guardian behaviour")
g_sell_frac = sb.slider("Fraction guardians SELL of any tokens received", 0.0, 1.0, 0.85, 0.05)
impact_on   = sb.checkbox("Apply price impact from guardian sales", value=True)

sb.header("Slot economics")
default_rent = sb.selectbox(
    "Default slot rent $/mo",
    [1_000, 1_500, 2_000, 2_500, 3_000, 3_500, 4_000, 4_500, 5_000],
    4
)
n_slots   = sb.number_input("Number of slots", 1, 50, 24)
slot_opts = np.arange(1_000, 5_001, 500)
slot_rents = [
    sb.selectbox(f"Slot {i+1} rent $/mo", slot_opts,
                 index=list(slot_opts).index(default_rent),
                 key=f"rent{i}")
    for i in range(n_slots)
]

sb.header("Validator set-up")
n_val = sb.number_input("Number of validators", 1, 100, 8)
val_cfg = []
for i in range(n_val):
    c1, c2 = sb.columns(2)
    stake_i = c1.number_input(f"Validator {i+1} initial stake (tok)",
                              1_000.0, 1_000_000.0, 100_000.0, 1_000.0,
                              key=f"s{i}")
    slots_i = c2.slider(f"Slots indexed", 0, n_slots, 3, key=f"l{i}")
    val_cfg.append({"stake": stake_i, "slots": slots_i})

sb.header("Validator costs")
var_pct = sb.slider("Variable cost % of $500", 0.20, 0.40, 0.25, 0.01)

run_sim = sb.button("Run simulation")

# ───────────────────────── simulation function ───────────────────────────
def run_simulation(apy_target: float, seed: int):
    np.random.seed(seed)

    vals = pd.DataFrame(val_cfg).assign(
        slot_usd=0.0, issuance=0.0, cost=0.0, net_usd=0.0,
        apy=0.0, token_cum=0.0, usd_cum=0.0
    )

    if vals.stake.sum() > EMIT_PER_MONTH:
        vals.stake *= EMIT_PER_MONTH / vals.stake.sum()

    # Round-robin slot masks
    mask = []
    for i, v in enumerate(vals.itertuples()):
        m = np.zeros(n_slots, bool)
        for j in range(v.slots):
            m[(i + j*n_val) % n_slots] = True
        mask.append(m)

    # Guardian liquid pool
    g_stake = 0.0

    supply_liq = vals.stake.sum()              # only validator stake is locked
    price, dao_cash = price0, 0.0

    sold_tot_tokens = sold_tot_value = 0.0
    hist = []

    for m in range(1, months+1):

        # 1. Stake per slot (validators)
        stake_ps = np.zeros((n_val, n_slots))
        for i, v in enumerate(vals.itertuples()):
            if v.slots:
                stake_ps[i, mask[i]] = v.stake / v.slots

        # 2. Slot rent split
        vals.slot_usd[:] = 0.0
        rent_to_val = 0.0
        for s in range(n_slots):
            rv = slot_rents[s] * (1 - DAO_SPLIT)
            staked = stake_ps[:, s].sum()
            if staked == 0:
                continue
            for i in range(n_val):
                if stake_ps[i, s]:
                    add = rv * stake_ps[i, s] / staked
                    vals.slot_usd.iat[i] += add
                    rent_to_val          += add
        dao_cash_month = rent_to_val * DAO_SPLIT / (1 - DAO_SPLIT)

        # 3. Monthly emission 80/20
        minted = min(EMIT_PER_MONTH, MAX_SUPPLY - supply_liq)
        supply_liq += minted
        emis_val = minted * 0.80
        emis_g   = minted * 0.20

        # Validators get locked emission
        if emis_val > 0 and vals.stake.sum() > 0:
            share = vals.stake / vals.stake.sum()
            vals.issuance = emis_val * share
            vals.stake   += vals.issuance
        else:
            vals.issuance = 0.0

        # Guardians split emission
        sold_emis = emis_g * g_sell_frac
        g_stake  += emis_g * (1 - g_sell_frac)

        sold_tokens_month = sold_emis
        sold_value_month  = sold_emis * price

        # 4. Misbehaviour / slashing
        mis = np.random.rand(n_val) < p_mis
        cut = np.minimum(slash_amt, vals.loc[mis, "stake"])
        vals.loc[mis, "stake"] -= cut
        pool = cut.sum()

        if pool > 0:
            sold_cut = pool * g_sell_frac
            g_stake += pool * (1 - g_sell_frac)
            sold_tokens_month += sold_cut
            sold_value_month  += sold_cut * price

        # Price impact from total sales
        if impact_on and sold_tokens_month > 0:
            price *= guardian_price_factor(sold_tokens_month)

        # Update sale stats
        sold_tot_tokens += sold_tokens_month
        sold_tot_value  += sold_value_month

        validator_staked = vals.stake.sum()

        # 5. DAO buy-back / burn (guardian tokens can be burned)
        dao_cash += dao_cash_month
        burn_cash = burn_frac * dao_cash
        burnable  = max(0.0, supply_liq - validator_staked)
        burn_tok  = min(burn_cash / price, burnable)
        supply_liq -= burn_tok
        dao_cash   -= burn_cash
        if supply_liq:
            price += price * (burn_cash / (supply_liq * price))

        # 6. Price GBM
        price = gbm_step(price, mu, sigma)

        # 7. Validator economics
        vals.cost    = (500 + 500*var_pct) * vals.slots
        vals.net_usd = vals.issuance*price + vals.slot_usd - vals.cost
        vals.apy     = vals.net_usd / (vals.stake * price)
        vals.token_cum += vals.issuance
        vals.usd_cum   += vals.slot_usd

        need   = revenue_needed(apy_target, validator_staked, price)
        actual = rent_to_val + vals.issuance.sum()*price
        shortfall = max(0.0, need - actual)

        hist.append({
            "month": m,
            "price": price,
            "supply": supply_liq,
            "staking_ratio": validator_staked / supply_liq,
            "avg_apy": vals.apy.mean(),
            "shortfall": shortfall
        })

    df_hist = pd.DataFrame(hist).set_index("month")
    summary = {
        "Avg APY": vals.apy.mean(),
        "Avg shortfall $": df_hist.shortfall.mean(),
        "Avg token-earn (tok/val/mo)": (vals.token_cum / months).mean(),
        "Avg USD-earn ($/val/mo)":     (vals.usd_cum  / months).mean(),
        "Sold tokens": sold_tot_tokens,
        "Avg sale price $": sold_tot_value / sold_tot_tokens if sold_tot_tokens else 0.0
    }
    return df_hist, vals.copy(), price, summary

# ───────────────────────── run & cache once ──────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = None

if run_sim:
    st.session_state.results = {
        name: run_simulation(apy, rng_seed) for name, apy in SCENARIOS.items()
    }

if st.session_state.results is None:
    st.info("Click the \"Run simulation\" button.")
    st.stop()

# ───────────────────────── visualisations ────────────────────────────────
case = st.selectbox("Scenario", ["Base", "Bear", "Bull"])
df, vals, price_last, summary = st.session_state.results[case]

fig_apy = px.line(df, x=df.index, y=df.avg_apy*100,
                  title=f"{case} – Avg APY vs Target",
                  labels={"x": "Month", "y": "APY %"})
fig_apy.add_hline(y=SCENARIOS[case]*100, line_dash="dash")
st.plotly_chart(fig_apy, use_container_width=True)

st.plotly_chart(px.line(df, x=df.index, y="price",
                        title=f"{case} – Token price $"),
                use_container_width=True)

st.plotly_chart(px.bar(df, x=df.index, y="shortfall",
                       title=f"{case} – Revenue shortfall"),
                use_container_width=True)

fig_sr = make_subplots(specs=[[{"secondary_y": True}]])
fig_sr.add_trace(go.Scatter(x=df.index, y=df.supply/1e6,
                            name="Liquid supply (M)"), secondary_y=False)
fig_sr.add_trace(go.Scatter(x=df.index, y=df.staking_ratio,
                            name="Staking ratio"), secondary_y=True)
fig_sr.update_yaxes(title_text="Tokens (M)", secondary_y=False)
fig_sr.update_yaxes(title_text="Staked / Supply", range=[0, 1.2], secondary_y=True)
fig_sr.update_layout(title=f"{case} – Supply vs Staking ratio")
st.plotly_chart(fig_sr, use_container_width=True)

avg_tok_usd = (vals.token_cum / months) * price_last
avg_usd     = vals.usd_cum / months
fig_bar = go.Figure()
fig_bar.add_bar(x=list(range(n_val)), y=avg_tok_usd, name="Avg token rewards (USD)")
fig_bar.add_bar(x=list(range(n_val)), y=avg_usd,     name="Avg stable-coin rewards")
fig_bar.update_layout(barmode="group",
                      title=f"{case} – Avg earnings / validator / month",
                      xaxis_title="Validator", yaxis_title="USD")
st.plotly_chart(fig_bar, use_container_width=True)

# Comparison table
comp_df = pd.DataFrame({k: v[3] for k, v in st.session_state.results.items()}).T
st.header("Scenario comparison – averages")
st.dataframe(
    comp_df.drop(columns=["Sold tokens", "Avg sale price $"]).style.format({
        "Avg APY": "{:.2%}",
        "Avg shortfall $": "${:,.0f}",
        "Avg token-earn (tok/val/mo)": "{:,.0f}",
        "Avg USD-earn ($/val/mo)": "${:,.0f}"
    }),
    use_container_width=True
)

# Guardian sales stats
st.subheader("Guardian sales during simulation")
st.write(f"Total tokens sold: {summary['Sold tokens']:,.0f}")
st.write(f"Average sale price: ${summary['Avg sale price $']:.4f}")
