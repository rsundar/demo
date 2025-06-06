# staking_sim.py  ──────────────────────────────────────────────────────────
# Streamlit staking-economics simulator
#   • 100 M cap, 3 330 000 tokens/month emission
#   • Bear / Base / Bull APY targets
#   • Per-slot rent split (80 % validators, 20 % DAO)
#   • All accounting fixes:
#       – tokens restaked exactly once
#       – no supply drop on slashing / guardian sells
#       – DAO burn limited to liquid – staked
#       – dynamic stake-per-slot every month
# -------------------------------------------------------------------------

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ───────────────────────── constants ─────────────────────────────────────
MAX_SUPPLY      = 100_000_000
EMIT_PER_MONTH  = 3_330_000
DAO_SPLIT       = 0.20
SCENARIOS       = {"Bear": 0.05, "Base": 0.125, "Bull": 0.30}

# ───────────────────────── helpers ───────────────────────────────────────
def gbm_step(p, mu, sigma):
    dt, z = 1/12, np.random.randn()
    return p * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)

def revenue_needed(apy, stake_tok, price):
    r_m = (1 + apy)**(1/12) - 1
    return r_m * stake_tok * price

# ───────────────────────── sidebar UI ────────────────────────────────────
st.title("Validator – Guardian Staking Simulator")

sb = st.sidebar
sb.header("Timeline & price")
months   = sb.slider("Months", 6, 120, 36, 6)
price0   = sb.number_input("Start price $", 0.01, 10.0, 0.05, 0.005, format="%.3f")
mu       = sb.number_input("GBM drift μ",  -1.0, 2.0, 0.30, 0.05)
sigma    = sb.number_input("GBM vol σ",     0.01, 3.0, 1.0, 0.05)
rng_seed = sb.number_input("Random seed",   0, 2**31-1, 1)

sb.header("DAO buy-back / burn")
burn_frac = sb.slider("Burn fraction b", 0.0, 1.0, 0.25, 0.05)

sb.header("Misbehaviour / slashing")
p_mis   = sb.slider("Misbehave prob", 0.0, 0.5, 0.05, 0.01)
slash_amt = sb.number_input("Slash amount (tok)", 1.0, 1e6, 500.0, 10.0)

sb.header("Guardian behaviour")
g_sell_frac = sb.slider("Guardian sell %", 0.0, 1.0, 0.5, 0.05)

sb.header("Slot economics")
# n_slots   = sb.number_input("Slots", 1, 50, 24)
# slot_opts = np.arange(1_000, 5_001, 500)
# slot_rents = [sb.selectbox(f"Slot {i+1} rent $/mo", slot_opts, 2)
#               for i in range(n_slots)]


default_rent = sb.selectbox("Default slot rent $/mo",
                            [1_000, 1_500, 2_000, 2_500, 3_000, 3_500, 4_000, 4_500, 5_000],
                            index=4)                                   # ← NEW

n_slots   = sb.number_input("Number of slots", 1, 50, 24)
slot_opts = np.arange(1_000, 5_001, 500)

slot_rents = []
for i in range(n_slots):
    slot_rents.append(
        sb.selectbox(f"Slot {i+1} rent $/mo",
                     slot_opts,
                     index=list(slot_opts).index(default_rent),
                     key=f"rent{i}")
    )

sb.header("Validator set-up")
n_val = sb.number_input("Validators", 1, 100, 8)
val_cfg = []
for i in range(n_val):
    c1, c2 = sb.columns(2)
    stake_tok = c1.number_input(f"Val {i+1} stake (tok)", 1_000., 1_000_000.,
                                100_000., 1_000., key=f"s{i}")
    slots_idx = c2.slider(f"Slots indexed", 0, n_slots, 3, key=f"l{i}")
    val_cfg.append({"stake": stake_tok, "slots": slots_idx})

sb.header("Validator costs")
var_pct = sb.slider("Variable cost % of $500", 0.20, 0.40, 0.25, 0.01)

run_sim = sb.button("Run simulation")

# ───────────────────────── simulation ────────────────────────────────────
def run_simulation(apy_target: float, seed: int):
    np.random.seed(seed)

    vals = pd.DataFrame(val_cfg).assign(
        slot_usd=0.0, issuance=0.0, cost=0.0, net_usd=0.0,
        apy=0.0, token_cum=0.0, usd_cum=0.0
    )

    # scale initial stakes to ≤ first-month emission (optional safety)
    tot0 = vals.stake.sum()
    if tot0 > EMIT_PER_MONTH:
        vals.stake *= EMIT_PER_MONTH / tot0
        st.info("Initial stakes scaled to first-month emission.")

    # ── round-robin slot assignment masks (True where validator indexes) ──
    slot_mask = []
    for i, v in enumerate(vals.itertuples()):
        mask = np.zeros(n_slots, bool)
        step = n_val
        for j in range(v.slots):
            mask[(i + j*step) % n_slots] = True
        slot_mask.append(mask)

    supply_liq = vals.stake.sum()      # circulating = stake only
    price      = price0
    dao_cash   = 0.0
    history    = []

    for m in range(1, months+1):

        # build stake-per-slot matrix for current epoch
        stake_per_slot = np.zeros((n_val, n_slots))
        for i, v in enumerate(vals.itertuples()):
            if v.slots == 0:
                continue
            stake_here = v.stake / v.slots
            stake_per_slot[i, slot_mask[i]] = stake_here

        # rent split
        vals.slot_usd = 0.0
        total_rent_to_validators = 0.0

        for s in range(n_slots):
            rent_s = slot_rents[s]
            rent_v = rent_s * (1 - DAO_SPLIT)
            staked_in_s = stake_per_slot[:, s].sum()
            if staked_in_s == 0:
                continue
            for i in range(n_val):
                if stake_per_slot[i, s] > 0:
                    share = stake_per_slot[i, s] / staked_in_s
                    add   = rent_v * share
                    vals.slot_usd.iat[i] += add
                    total_rent_to_validators += add

        dao_cash_month = total_rent_to_validators * DAO_SPLIT / (1 - DAO_SPLIT)

        # emission → restake once
        minted = min(EMIT_PER_MONTH, MAX_SUPPLY - supply_liq)
        supply_liq += minted
        share = vals.stake / vals.stake.sum()
        vals.issuance = minted * share
        vals.stake   += vals.issuance

        # 3️⃣ misbehaviour / guardians (no supply change)
        mis = np.random.rand(n_val) < p_mis
        slashed = np.minimum(slash_amt, vals.loc[mis, "stake"])
        vals.loc[mis, "stake"] -= slashed
        guardian_pool = slashed.sum()
        if guardian_pool:
            sold  = guardian_pool * g_sell_frac
            deleg = guardian_pool - sold
            best  = vals.apy.idxmax() if (vals.apy != 0).any() else vals.stake.idxmax()
            vals.stake.iat[best] += deleg  # tokens move, supply unchanged

        total_stake = vals.stake.sum()

        # DAO buy-back / burn (cannot burn staked tokens)
        dao_cash += dao_cash_month
        burn_cash = burn_frac * dao_cash
        burnable  = max(0.0, supply_liq - total_stake)
        burn_tok  = min(burn_cash / price, burnable)
        supply_liq -= burn_tok
        dao_cash   -= burn_cash
        if supply_liq:
            price += price * (burn_cash / (supply_liq * price))

        # price step
        price = gbm_step(price, mu, sigma)

        #P&L, APY
        vals.cost    = (500 + 500*var_pct) * vals.slots
        vals.net_usd = vals.issuance*price + vals.slot_usd - vals.cost
        vals.apy     = vals.net_usd / (vals.stake * price)
        vals.token_cum += vals.issuance
        vals.usd_cum   += vals.slot_usd
        avg_apy = vals.apy.mean()

        need_usd   = revenue_needed(apy_target, total_stake, price)
        actual_usd = total_rent_to_validators + vals.issuance.sum()*price
        shortfall  = max(0.0, need_usd - actual_usd)

        history.append({
            "month": m,
            "price": price,
            "supply": supply_liq,
            "staking_ratio": total_stake / supply_liq,
            "avg_apy": avg_apy,
            "shortfall": shortfall
        })

    df = pd.DataFrame(history).set_index("month")
    summary = {
        "Avg APY": vals.apy.mean(),
        "Avg shortfall $": df.shortfall.mean(),
        "Avg token-earn (tok/val/mo)": (vals.token_cum/ months).mean(),
        "Avg USD-earn ($/val/mo)":     (vals.usd_cum  / months).mean()
    }
    return df, vals, price, summary

# ───────────────────────── one-time run & cache ──────────────────────────
def build_all():
    return {n: run_simulation(a, rng_seed) for n, a in SCENARIOS.items()}

if "results" not in st.session_state:
    st.session_state.results = None
if run_sim:
    st.session_state.results = build_all()

results = st.session_state.results
if results is None:
    st.info("Click to run simulation.")
    st.stop()

# ───────────────────────── visualisation ─────────────────────────────────
case = st.selectbox("Scenario", ["Base", "Bear", "Bull"])
df_show, vals_show, price_last, _ = results[case]

fig_apy = px.line(df_show, x=df_show.index, y=df_show.avg_apy*100,
                  labels={"x":"Month", "y":"APY %"},
                  title=f"{case} – Avg APY vs Target")
fig_apy.add_hline(y=SCENARIOS[case]*100, line_dash="dash")
st.plotly_chart(fig_apy, use_container_width=True)

st.plotly_chart(px.line(df_show, x=df_show.index, y="price",
                        title=f"{case} – Token price $"),
                use_container_width=True)

st.plotly_chart(px.bar(df_show, x=df_show.index, y="shortfall",
                       title=f"{case} – Revenue shortfall"),
                use_container_width=True)

fig_sr = make_subplots(specs=[[{"secondary_y": True}]])
fig_sr.add_trace(go.Scatter(x=df_show.index, y=df_show.supply/1e6,
                            name="Liquid supply (M)"), secondary_y=False)
fig_sr.add_trace(go.Scatter(x=df_show.index, y=df_show.staking_ratio,
                            name="Staking ratio"), secondary_y=True)
fig_sr.update_yaxes(title_text="Staked / Supply",
                    range=[0, 1.2], secondary_y=True)
fig_sr.update_layout(title=f"{case} – Supply vs Staking ratio")
st.plotly_chart(fig_sr, use_container_width=True)

avg_tok_usd = (vals_show.token_cum / months) * price_last
avg_usd     =  vals_show.usd_cum   / months
fig_bar = go.Figure()
fig_bar.add_bar(x=list(range(n_val)), y=avg_tok_usd,
                name="Avg token rewards (USD)")
fig_bar.add_bar(x=list(range(n_val)), y=avg_usd,
                name="Avg stable-coin rewards")
fig_bar.update_layout(barmode="group",
                      title=f"{case} – Avg earnings/validator/month",
                      xaxis_title="Validator", yaxis_title="USD")
st.plotly_chart(fig_bar, use_container_width=True)

comp_df = pd.DataFrame({k: v[3] for k, v in results.items()}).T
st.header("Scenario comparison – averages")
st.dataframe(comp_df.style.format({
    "Avg APY":"{:.2%}",
    "Avg shortfall $":"${:,.0f}",
    "Avg token-earn (tok/val/mo)":"{:,.0f}",
    "Avg USD-earn ($/val/mo)":"${:,.0f}"
}), use_container_width=True)
