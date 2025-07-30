
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
ADTV_TOKENS     = 1_000_000          # average daily trading volume (tokens)

# ───────────────────────── helper functions ──────────────────────────────
def gbm_step(price, mu, sigma):
    return price * np.exp((mu - 0.5*sigma**2)/12 + sigma*np.sqrt(1/12)*np.random.randn())


def revenue_needed(apy, stake_tokens, price):
    r_m = (1 + apy)**(1/12) - 1
    return r_m * stake_tokens * price


# Guardian-sell impact (< 1)  --------------------------------------------
def guardian_price_factor(tokens_sold):
    if tokens_sold == 0:
        return 1.0
    pct = tokens_sold / ADTV_TOKENS * 100
    if   pct <= 1:   base = 0.995 - pct*0.005
    elif pct <= 2:   base = 0.990 - (pct-1)*0.015
    elif pct <= 3:   base = 0.980 - (pct-2)*0.03
    elif pct <= 5:   base = 0.950 - (pct-3)*0.05
    elif pct <= 10:  base = 0.850 - (pct-5)*0.04
    elif pct <= 20:  base = 0.650 - (pct-10)*0.025
    else:            base = max(0.1, 0.400 - (pct-20)*0.015)
    return max(0.1, base)

# Mirror curve (> 1) for organic buy --------------------------------------
def organic_buy_factor(tokens_bought):
    return 2.0 - guardian_price_factor(tokens_bought)   # symmetric mirror

# ───────────────────────── sidebar UI ────────────────────────────────────
st.title("Validator – Guardian Staking Simulator")
sb = st.sidebar

months   = sb.slider("Months to simulate", 6, 120, 36, 6)
price0   = sb.number_input("Initial token price $", 0.01, 10.0, 0.05, 0.005)
mu       = sb.number_input("GBM drift μ (annual)", -1.0, 2.0, 0.30, 0.05)
sigma    = sb.number_input("GBM volatility σ (annual)", 0.01, 3.0, 1.0, 0.05)
rng_seed = sb.number_input("Random seed", 0, 2**31-1, 1)

burn_frac   = sb.slider("DAO burn fraction of treasury", 0.0, 1.0, 0.25, 0.05)
organic_buy = sb.number_input("Organic buy pressure (USD / month)",
                              0.0, 10_000_000.0, 0.0, 1_000.0, format="%.0f")

p_mis     = sb.slider("Misbehaviour probability", 0.0, 0.5, 0.05, 0.01)
slash_amt = sb.number_input("Slash amount (tokens)", 1.0, 1_000_000.0, 500.0, 10.0)

g_sell_frac = sb.slider("Guardians sell fraction", 0.0, 1.0, 0.85, 0.05)
impact_on   = sb.checkbox("Apply price impact from guardian sales / buys", value=True)

# Slot economics
sb.header("Slot economics")
default_rent = sb.selectbox("Default slot rent $/mo",
                            [1_000,1_500,2_000,2_500,3_000,3_500,4_000,4_500,5_000],4)
n_slots   = sb.number_input("Number of slots", 1, 50, 24)
slot_opts = np.arange(1_000, 5_001, 500)
slot_rents = [sb.selectbox(f"Slot {i+1} rent $/mo", slot_opts,
                           index=list(slot_opts).index(default_rent),
                           key=f"rent{i}") for i in range(n_slots)]
rent_growth = sb.slider("Monthly slot-rent growth %", -50.0, 100.0, 0.0, 1.0)/100

# Validators
sb.header("Validator set-up")
n_val = sb.slider("Number of validators", 1, 100, 8)
stake_per_val = sb.number_input("Initial stake per validator (tok)",
                                1_000.0, 1_000_000.0, 100_000.0, 1_000.0)
slots_per_val = sb.slider("Slots indexed by each validator", 0, n_slots,
                          min(3, n_slots))

var_pct = sb.slider("Variable cost % of $500", 0.20, 0.40, 0.25, 0.01)
run_sim = sb.button("Run simulation")

# ───────────────────────── simulation core ───────────────────────────────
def run_simulation(target_apy: float, seed: int):
    np.random.seed(seed)

    vals = pd.DataFrame({
        "stake": [stake_per_val]*n_val,
        "slots": [slots_per_val]*n_val
    }).assign(
        slot_usd=0.0, issuance=0.0, cost=0.0, net_usd=0.0,
        apy=0.0, token_cum=0.0, usd_cum=0.0
    )

    # Slot mask (round-robin)
    mask = []
    for i in range(n_val):
        m = np.zeros(n_slots, bool)
        for j in range(slots_per_val):
            m[(i + j*n_val) % n_slots] = True
        mask.append(m)

    guardian_pool = 0.0
    supply_liq = vals.stake.sum()
    price, dao_cash = price0, 0.0

    sold_tot_tok = sold_tot_val = 0.0
    rents = np.array(slot_rents, dtype=float)
    hist = []

    for month in range(1, months+1):

        # 1 stake per slot
        stake_ps = np.zeros((n_val, n_slots))
        if slots_per_val:
            for i in range(n_val):
                stake_ps[i, mask[i]] = vals.stake.iat[i] / slots_per_val

        # 2 rent split
        vals.slot_usd[:] = 0.0; rent_to_val = 0.0
        for s in range(n_slots):
            rv = rents[s]*(1-DAO_SPLIT)
            staked = stake_ps[:,s].sum()
            if staked==0: continue
            vals.slot_usd += rv * stake_ps[:,s] / staked
            rent_to_val   += rv
        dao_cash_month = rent_to_val*DAO_SPLIT/(1-DAO_SPLIT)

        # 3 emission 80/20
        minted = min(EMIT_PER_MONTH, MAX_SUPPLY-supply_liq)
        supply_liq += minted
        emis_val = minted*0.80; emis_g = minted*0.20

        if emis_val and vals.stake.sum():
            share = vals.stake/vals.stake.sum()
            vals.issuance = emis_val*share
            vals.stake   += vals.issuance
        else:
            vals.issuance = 0.0

        sold_emis = emis_g*g_sell_frac
        guardian_pool += emis_g*(1-g_sell_frac)

        tokens_sold = sold_emis
        value_sold  = sold_emis*price

        # 4 slashing
        mis = np.random.rand(n_val)<p_mis
        cut = np.minimum(slash_amt, vals.loc[mis,"stake"])
        vals.loc[mis,"stake"] -= cut
        pool_cut = cut.sum()
        if pool_cut:
            sold_cut = pool_cut*g_sell_frac
            guardian_pool += pool_cut*(1-g_sell_frac)
            tokens_sold += sold_cut
            value_sold  += sold_cut*price

        # 5 price impact: guardian sales
        if impact_on and tokens_sold:
            price *= guardian_price_factor(tokens_sold)

        sold_tot_tok += tokens_sold
        sold_tot_val += value_sold

        validator_staked = vals.stake.sum()

        # 6 organic buy pressure
        buy_tokens = organic_buy / price
        buy_tokens = min(buy_tokens, supply_liq - validator_staked)  # can't buy beyond liquid
        if buy_tokens > 0:
           supply_liq -= buy_tokens 
           if impact_on:
                price *= organic_buy_factor(buy_tokens)

        # 7 DAO burn
        dao_cash += dao_cash_month
        burn_cash = burn_frac*dao_cash
        burnable  = max(0.0, supply_liq - validator_staked)
        burn_tok  = min(burn_cash/price, burnable)
        supply_liq -= burn_tok
        dao_cash   -= burn_cash
        if supply_liq:
            price += price*(burn_cash/(supply_liq*price))

        # 8 price GBM noise
        price = gbm_step(price, mu, sigma)

        # 9 economics
        vals.cost  = (500+500*var_pct)*slots_per_val
        vals.net_usd = vals.issuance*price + vals.slot_usd - vals.cost
        vals.apy  = vals.net_usd / (vals.stake*price)
        vals.token_cum += vals.issuance
        vals.usd_cum   += vals.slot_usd

        need   = revenue_needed(target_apy, validator_staked, price)
        actual = rent_to_val + vals.issuance.sum()*price
        shortfall = max(0.0, need-actual)

        hist.append({
            "month": month,
            "price": price,
            "supply": supply_liq,
            "staking_ratio": validator_staked / supply_liq,
            "avg_apy": vals.apy.mean(),
            "shortfall": shortfall
        })

        # grow rents
        rents *= (1 + rent_growth)

    df = pd.DataFrame(hist).set_index("month")
    summary = {
        "Avg APY": vals.apy.mean(),
        "Avg shortfall $": df.shortfall.mean(),
        "Avg token-earn (tok/val/mo)": (vals.token_cum / months).mean(),
        "Avg USD-earn ($/val/mo)":     (vals.usd_cum  / months).mean(),
        "Sold tokens": sold_tot_tok,
        "Avg sale price $": sold_tot_val/sold_tot_tok if sold_tot_tok else 0.
    }
    return df, vals.copy(), price, summary

# ── run & cache once ─────────────────────────────────────────────────────
if "results" not in st.session_state: st.session_state.results=None
if run_sim:
    st.session_state.results = {n: run_simulation(a, rng_seed) for n,a in SCENARIOS.items()}
if st.session_state.results is None:
    st.info("Click the \"Run simulation\" button."); st.stop()

case = st.selectbox("Scenario", ["Base","Bear","Bull"])
df, vals, price_last, summary = st.session_state.results[case]

fig = px.line(df, x=df.index, y=df.avg_apy*100, title=f"{case} – Avg APY vs Target",
              labels={"x":"Month","y":"APY %"}); fig.add_hline(y=SCENARIOS[case]*100,line_dash="dash")
st.plotly_chart(fig,use_container_width=True)

st.plotly_chart(px.line(df, x=df.index, y="price",
                        title=f"{case} – Token price $"),use_container_width=True)

st.plotly_chart(px.bar(df, x=df.index, y="shortfall",
                       title=f"{case} – Revenue shortfall"),use_container_width=True)

fig_sr = make_subplots(specs=[[{"secondary_y":True}]])
fig_sr.add_trace(go.Scatter(x=df.index, y=df.supply/1e6,
                            name="Liquid supply (M)"),secondary_y=False)
fig_sr.add_trace(go.Scatter(x=df.index, y=df.staking_ratio,
                            name="Staking ratio"),secondary_y=True)
fig_sr.update_yaxes(title_text="Tokens (M)",secondary_y=False)
fig_sr.update_yaxes(title_text="Staked / Supply",range=[0,1.2],secondary_y=True)
fig_sr.update_layout(title=f"{case} – Supply vs Staking ratio")
st.plotly_chart(fig_sr,use_container_width=True)

avg_tok_usd = (vals.token_cum/months)*price_last
avg_usd     = vals.usd_cum/months
fig_bar = go.Figure()
fig_bar.add_bar(x=list(range(n_val)), y=avg_tok_usd,
                name="Avg token rewards (USD)")
fig_bar.add_bar(x=list(range(n_val)), y=avg_usd,
                name="Avg stable-coin rewards")
fig_bar.update_layout(barmode="group",
                      title=f"{case} – Avg earnings / validator / month",
                      xaxis_title="Validator",yaxis_title="USD")
st.plotly_chart(fig_bar,use_container_width=True)

comp_df = pd.DataFrame({k:v[3] for k,v in st.session_state.results.items()}).T
st.header("Scenario comparison – averages")
st.dataframe(comp_df.drop(columns=["Sold tokens","Avg sale price $"]).style.format({
    "Avg APY":"{:.2%}","Avg shortfall $":"${:,.0f}",
    "Avg token-earn (tok/val/mo)":"{:,.0f}",
    "Avg USD-earn ($/val/mo)":"${:,.0f}"
}),use_container_width=True)

st.subheader("Guardian sales during simulation")
st.write(f"Total tokens sold: {summary['Sold tokens']:,.0f}")
st.write(f"Average sale price: ${summary['Avg sale price $']:.4f}")
