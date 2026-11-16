"""
European Bank - Customer Segmentation & Churn Analytics dashboard.
Run:  streamlit run app.py
Looks for European_Bank.csv next to this file (or in the working directory); otherwise offers an upload box.
All figures are calculated live from the data; CustomerId, Surname and Year are not used as analytical fields.
"""
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="European Bank | Churn Analytics", page_icon="🏦", layout="wide")

NAVY, RED, TEAL, GREY = "#1B365D", "#C8553D", "#2A9D8F", "#8A94A6"
REQUIRED = ["CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance", "NumOfProducts",
            "HasCrCard", "IsActiveMember", "EstimatedSalary", "Exited"]
AGE_BINS, AGE_LABELS = [0, 29, 39, 49, 59, 200], ["18-29", "30-39", "40-49", "50-59", "60+"]
CS_BINS = [0, 579, 669, 739, 799, 900]
CS_LABELS = ["Poor (<580)", "Fair (580-669)", "Good (670-739)", "Very good (740-799)", "Exceptional (800+)"]
BAL_BINS, BAL_LABELS = [-1, 0, 50000, 100000, 150000, 1e9], ["Zero", "1-50k", "50k-100k", "100k-150k", "150k+"]
SEG_ORDER = ["Product Overload", "Inactive 50+", "Active 50+", "Mid-Life Germany", "Mid-Life France & Spain", "Under 40"]
SEG_RULES = {
    "Product Overload": "3 or 4 products (checked first)",
    "Inactive 50+": "Age 50+ and not an active member",
    "Active 50+": "Age 50+ and an active member",
    "Mid-Life Germany": "Age 40-49, Germany, 1-2 products",
    "Mid-Life France & Spain": "Age 40-49, France or Spain, 1-2 products",
    "Under 40": "Age under 40, 1-2 products",
}

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.6rem; max-width: 1250px;}
    .hero {background: linear-gradient(120deg,#1B365D 0%,#2D4A7A 100%); color:white; padding:22px 28px;
           border-radius:14px; margin-bottom:18px;}
    .hero h1 {color:white; font-size:1.7rem; margin:0 0 4px 0;}
    .hero p {color:#D6E0F0; margin:0; font-size:0.95rem;}
    .kpi {background:#FFFFFF; border:1px solid #E3E8EF; border-left:5px solid #1B365D; border-radius:10px;
          padding:14px 18px; box-shadow:0 1px 3px rgba(0,0,0,.05);}
    .kpi.red {border-left-color:#C8553D;} .kpi.teal {border-left-color:#2A9D8F;}
    .kpi .l {font-size:.78rem; color:#5A6678; text-transform:uppercase; letter-spacing:.05em;}
    .kpi .v {font-size:1.9rem; font-weight:700; color:#1B365D; line-height:1.2;}
    .kpi .d {font-size:.8rem; color:#5A6678;}
    .obs {background:#F7F9FC; border:1px solid #E3E8EF; border-radius:10px; padding:10px 14px; margin-bottom:8px;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ data loading
def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    a = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - a) / d * 100, (c + a) / d * 100


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        raise ValueError(f"The file is missing required columns: {', '.join(missing)}")
    df = raw.dropna(subset=REQUIRED).copy()
    df["AgeGroup"] = pd.cut(df.Age, AGE_BINS, labels=AGE_LABELS)
    df["CreditBand"] = pd.cut(df.CreditScore, CS_BINS, labels=CS_LABELS)
    df["BalanceBand"] = pd.cut(df.Balance, BAL_BINS, labels=BAL_LABELS)
    df["Activity"] = np.where(df.IsActiveMember == 1, "Active", "Inactive")
    df["Products"] = df.NumOfProducts.astype(int).astype(str) + " product" + np.where(df.NumOfProducts == 1, "", "s")
    conds = [df.NumOfProducts >= 3,
             (df.Age >= 50) & (df.IsActiveMember == 0),
             (df.Age >= 50) & (df.IsActiveMember == 1),
             (df.Age >= 40) & (df.Geography == "Germany"),
             (df.Age >= 40)]
    df["Segment"] = np.select(conds, SEG_ORDER[:5], default="Under 40")
    return df


@st.cache_data(show_spinner=False)
def load_path(path: str) -> pd.DataFrame:
    return prepare(pd.read_csv(path))


def get_data():
    here = Path(__file__).resolve().parent
    for p in (here / "European_Bank.csv", Path.cwd() / "European_Bank.csv"):
        if p.exists():
            try:
                return load_path(str(p))
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not read {p.name}: {e}")
                st.stop()
    st.warning("European_Bank.csv was not found next to app.py. Upload it to continue.")
    up = st.file_uploader("Upload European_Bank.csv", type="csv")
    if up is None:
        st.stop()
    try:
        return prepare(pd.read_csv(up))
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not read the uploaded file: {e}")
        st.stop()


# ------------------------------------------------------------------ helpers
def rate_df(d, col, order=None):
    g = d.groupby(col, observed=True)["Exited"].agg(customers="count", churned="sum").reset_index()
    g["rate"] = g.churned / g.customers * 100
    ci = [wilson(k, n) for k, n in zip(g.churned, g.customers)]
    g["lo"], g["hi"] = [c[0] for c in ci], [c[1] for c in ci]
    g[col] = g[col].astype(str)
    if order:
        g[col] = pd.Categorical(g[col], [o for o in order if o in set(g[col])], ordered=True)
        g = g.sort_values(col)
        g[col] = g[col].astype(str)
    return g


def bar(t, col, title, base, order=None, height=290, horizontal=False):
    t = t.copy()
    t["flag"] = np.where(t.rate == t.rate.max(), "highest", "other")
    x_sort = order if order else list(t[col])
    tip = [alt.Tooltip(col, title=title), alt.Tooltip("customers:Q", title="Customers", format=","),
           alt.Tooltip("churned:Q", title="Churned", format=","), alt.Tooltip("rate:Q", title="Churn rate %", format=".1f"),
           alt.Tooltip("lo:Q", title="95% CI low %", format=".1f"), alt.Tooltip("hi:Q", title="95% CI high %", format=".1f")]
    color = alt.condition(alt.datum.flag == "highest", alt.value(RED), alt.value(NAVY))
    if horizontal:
        ch = alt.Chart(t).mark_bar().encode(y=alt.Y(f"{col}:N", sort=x_sort, title=None),
                                            x=alt.X("rate:Q", title="Churn rate (%)"), color=color, tooltip=tip)
        txt = alt.Chart(t).mark_text(align="left", dx=4, fontSize=11).encode(
            y=alt.Y(f"{col}:N", sort=x_sort), x="rate:Q", text=alt.Text("rate:Q", format=".1f"))
        rule = alt.Chart(pd.DataFrame({"b": [base]})).mark_rule(color=GREY, strokeDash=[5, 4]).encode(x="b:Q")
    else:
        ch = alt.Chart(t).mark_bar().encode(x=alt.X(f"{col}:N", sort=x_sort, title=None, axis=alt.Axis(labelAngle=-20)),
                                            y=alt.Y("rate:Q", title="Churn rate (%)"), color=color, tooltip=tip)
        txt = alt.Chart(t).mark_text(dy=-7, fontSize=11).encode(
            x=alt.X(f"{col}:N", sort=x_sort), y="rate:Q", text=alt.Text("rate:Q", format=".1f"))
        rule = alt.Chart(pd.DataFrame({"b": [base]})).mark_rule(color=GREY, strokeDash=[5, 4]).encode(y="b:Q")
    return (ch + txt + rule).properties(title=title, height=height).configure_title(fontSize=14, anchor="start", color=NAVY)


def kpi(label, value, sub, cls=""):
    return f'<div class="kpi {cls}"><div class="l">{label}</div><div class="v">{value}</div><div class="d">{sub}</div></div>'


def small_n_note(t, col):
    small = t[t.customers < 100]
    if len(small):
        st.caption("Small groups (under 100 customers) have wide uncertainty: " +
                   ", ".join(f"{r[col]} (n={r.customers})" for _, r in small.iterrows()) + ".")


# ------------------------------------------------------------------ app
df = get_data()
overall_rate = df.Exited.mean() * 100

st.markdown(
    '<div class="hero"><h1>🏦 European Bank: Customer Segmentation & Churn Analytics</h1>'
    '<p>Interactive view of who leaves, where risk concentrates, and which segments matter most. '
    'All figures are calculated live from the customer file.</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Filters")
    geos = st.multiselect("Country", sorted(df.Geography.unique()), default=sorted(df.Geography.unique()))
    genders = st.multiselect("Gender", sorted(df.Gender.unique()), default=sorted(df.Gender.unique()))
    amin, amax = int(df.Age.min()), int(df.Age.max())
    age_rng = st.slider("Age range", amin, amax, (amin, amax))
    activity = st.radio("Customer activity", ["All", "Active", "Inactive"], horizontal=True)
    prods = st.multiselect("Number of products", sorted(df.NumOfProducts.unique()), default=sorted(df.NumOfProducts.unique()))
    st.caption("Charts, KPIs, segments and risk observations update with every filter.")

mask = (df.Geography.isin(geos) & df.Gender.isin(genders) & df.Age.between(*age_rng) & df.NumOfProducts.isin(prods))
if activity != "All":
    mask &= df.Activity == activity
f = df[mask]

if f.empty:
    st.warning("No customers match the current filters. Widen the filters in the sidebar.")
    st.stop()

total, churned = len(f), int(f.Exited.sum())
retained = total - churned
rate = churned / total * 100
delta = rate - overall_rate
c1, c2, c3, c4 = st.columns(4)
c1.markdown(kpi("Total customers", f"{total:,}", f"{total / len(df) * 100:.1f}% of all {len(df):,} customers"), unsafe_allow_html=True)
c2.markdown(kpi("Churned customers", f"{churned:,}", "customers who exited", "red"), unsafe_allow_html=True)
c3.markdown(kpi("Retained customers", f"{retained:,}", f"{100 - rate:.1f}% retention", "teal"), unsafe_allow_html=True)
c4.markdown(kpi("Churn rate", f"{rate:.2f}%", f"{delta:+.1f} pp vs overall {overall_rate:.2f}%", "red"), unsafe_allow_html=True)
if total < 100:
    st.warning(f"Only {total} customers match the filters; rates may be unreliable.")
st.write("")

t1, t2, t3, t4, t5 = st.tabs(["Overview", "Customer profile", "Segments", "Risk observations", "Data"])

with t1:
    a, b = st.columns(2)
    with a:
        dist = pd.DataFrame({"Outcome": ["Retained", "Churned"], "Customers": [retained, churned]})
        pie = alt.Chart(dist).mark_arc(innerRadius=60).encode(
            theta="Customers:Q", color=alt.Color("Outcome:N", scale=alt.Scale(domain=["Retained", "Churned"], range=[TEAL, RED])),
            tooltip=["Outcome", alt.Tooltip("Customers:Q", format=",")]).properties(title="Churn distribution", height=290)
        st.altair_chart(pie.configure_title(fontSize=14, anchor="start", color=NAVY), width="stretch")
    with b:
        t = rate_df(f, "Geography")
        st.altair_chart(bar(t, "Geography", "Churn rate by country", rate), width="stretch")
    a, b = st.columns(2)
    with a:
        t = rate_df(f, "AgeGroup", AGE_LABELS)
        st.altair_chart(bar(t, "AgeGroup", "Churn rate by age group", rate, AGE_LABELS), width="stretch")
        small_n_note(t, "AgeGroup")
    with b:
        t = rate_df(f, "Gender")
        st.altair_chart(bar(t, "Gender", "Churn rate by gender", rate), width="stretch")

with t2:
    a, b = st.columns(2)
    with a:
        t = rate_df(f, "Activity")
        st.altair_chart(bar(t, "Activity", "Active vs inactive customers", rate), width="stretch")
    with b:
        t = rate_df(f, "Products", [f"{i} product" + ("" if i == 1 else "s") for i in range(1, 5)])
        st.altair_chart(bar(t, "Products", "Churn rate by number of products", rate), width="stretch")
        small_n_note(t, "Products")
    a, b = st.columns(2)
    with a:
        t = rate_df(f, "CreditBand", CS_LABELS)
        st.altair_chart(bar(t, "CreditBand", "Churn rate by credit score band", rate, CS_LABELS), width="stretch")
    with b:
        t = rate_df(f, "BalanceBand", BAL_LABELS)
        st.altair_chart(bar(t, "BalanceBand", "Churn rate by balance band", rate, BAL_LABELS), width="stretch")
        small_n_note(t, "BalanceBand")
    a, b = st.columns(2)
    with a:
        hm = f.groupby(["AgeGroup", "Activity"], observed=True).Exited.agg(rate="mean", customers="count").reset_index()
        hm["rate"] *= 100
        base_h = alt.Chart(hm).encode(x=alt.X("Activity:N", title=None), y=alt.Y("AgeGroup:N", sort=AGE_LABELS, title=None))
        heat = base_h.mark_rect().encode(color=alt.Color("rate:Q", scale=alt.Scale(scheme="reds", domain=[0, 100]), title="Churn %"),
                                         tooltip=["AgeGroup", "Activity", alt.Tooltip("rate:Q", format=".1f"), "customers"])
        lab = base_h.mark_text(fontSize=12).encode(text=alt.Text("rate:Q", format=".0f"),
                                                   color=alt.condition(alt.datum.rate > 55, alt.value("white"), alt.value("black")))
        st.altair_chart((heat + lab).properties(title="Churn % by age group and activity", height=290)
                        .configure_title(fontSize=14, anchor="start", color=NAVY), width="stretch")
    with b:
        cols = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary", "Exited"]
        if f.Exited.nunique() > 1:
            cm = f[cols].corr().stack().reset_index()
            cm.columns = ["a", "b", "r"]
            base_c = alt.Chart(cm).encode(x=alt.X("a:N", title=None, sort=cols, axis=alt.Axis(labelAngle=-40)),
                                          y=alt.Y("b:N", title=None, sort=cols))
            hc = base_c.mark_rect().encode(color=alt.Color("r:Q", scale=alt.Scale(scheme="redblue", domain=[1, -1]), title="r"),
                                           tooltip=["a", "b", alt.Tooltip("r:Q", format=".2f")])
            tc = base_c.mark_text(fontSize=9).encode(text=alt.Text("r:Q", format=".2f"))
            st.altair_chart((hc + tc).properties(title="Correlation heatmap", height=290)
                            .configure_title(fontSize=14, anchor="start", color=NAVY), width="stretch")
        else:
            st.info("The correlation heatmap needs both churned and retained customers in the filtered data.")

with t3:
    rows = []
    for s in SEG_ORDER:
        d = f[f.Segment == s]
        if len(d):
            rows.append({"Segment": s, "Rule": SEG_RULES[s], "Customers": len(d), "Churned": int(d.Exited.sum()),
                         "Churn rate %": d.Exited.mean() * 100, "Share of customers %": len(d) / total * 100,
                         "Share of churners %": (d.Exited.sum() / churned * 100) if churned else 0.0,
                         "Avg age": d.Age.mean(), "Avg balance": d.Balance.mean(),
                         "Balance of churned": d.loc[d.Exited == 1, "Balance"].sum()})
    seg = pd.DataFrame(rows)
    a, b = st.columns([1.1, 1])
    with a:
        sc = seg.rename(columns={"Churn rate %": "rate", "Customers": "customers", "Churned": "churned"})
        sc["lo"], sc["hi"] = 0, 0
        st.altair_chart(bar(sc, "Segment", "Churn rate by segment", rate, SEG_ORDER, height=320, horizontal=True),
                        width="stretch")
    with b:
        m = seg.melt(id_vars="Segment", value_vars=["Share of customers %", "Share of churners %"], var_name="Measure", value_name="Percent")
        st.altair_chart(alt.Chart(m).mark_bar().encode(
            y=alt.Y("Segment:N", sort=SEG_ORDER, title=None), x=alt.X("Percent:Q", title="Percent (%)"),
            yOffset="Measure:N", color=alt.Color("Measure:N", scale=alt.Scale(range=[GREY, RED]), legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["Segment", "Measure", alt.Tooltip("Percent:Q", format=".1f")]).properties(
            title="Share of customers vs share of churners", height=320).configure_title(fontSize=14, anchor="start", color=NAVY),
            width="stretch")
    st.dataframe(
        seg,
        column_config={
            "Churn rate %":        st.column_config.NumberColumn("Churn rate %",        format="%.1f"),
            "Share of customers %": st.column_config.NumberColumn("Share of customers %", format="%.1f"),
            "Share of churners %": st.column_config.NumberColumn("Share of churners %", format="%.1f"),
            "Avg age":             st.column_config.NumberColumn("Avg age",             format="%.1f"),
            "Avg balance":         st.column_config.NumberColumn("Avg balance",         format="%,.0f"),
            "Balance of churned":  st.column_config.NumberColumn("Balance of churned",  format="%,.0f"),
        },
        width="stretch", hide_index=True,
    )
    st.markdown("**How to read this.** Segments are mutually exclusive business rules applied in the order listed. Rates are "
                "in-sample descriptions of this file, not forecasts. Segments with few customers have wide uncertainty.")
    if len(seg) > 1 and churned:
        top = seg.sort_values("Churn rate %", ascending=False).iloc[0]
        st.info(f"Highest-churn segment in the current filter: **{top.Segment}** at {top['Churn rate %']:.1f}% "
                f"({int(top.Customers):,} customers, {top['Share of churners %']:.1f}% of churners).")

with t4:
    st.subheader("Automated churn-risk observations")
    st.caption("Calculated on the current filter. Groups with fewer than 50 customers are excluded. These are associations in "
               "the data, not proven causes.")
    dims = {"Country": "Geography", "Gender": "Gender", "Age group": "AgeGroup", "Activity": "Activity",
            "Products": "Products", "Credit score band": "CreditBand", "Balance band": "BalanceBand"}
    cand = []
    for lab, col in dims.items():
        g = rate_df(f, col)
        for _, r in g.iterrows():
            if r.customers >= 50:
                cand.append({"Dimension": lab, "Group": r[col], "Customers": int(r.customers), "Churn rate %": r.rate,
                             "Lift vs filtered avg": r.rate / rate if rate else np.nan,
                             "CI low": r.lo, "CI high": r.hi})
    cd = pd.DataFrame(cand)
    if cd.empty or churned == 0:
        st.info("Not enough churn events in this filter to rank risk groups.")
    else:
        hi = cd.sort_values("Churn rate %", ascending=False).head(5)
        lo = cd.sort_values("Churn rate %").head(3)
        st.markdown("**Highest-risk groups**")
        for _, r in hi.iterrows():
            st.markdown(f'<div class="obs">🔴 <b>{r.Dimension}: {r.Group}</b> churns at <b>{r["Churn rate %"]:.1f}%</b> '
                        f'({r["Lift vs filtered avg"]:.1f}x the filtered average of {rate:.1f}%), n={r.Customers:,}. '
                        f'95% CI {r["CI low"]:.1f}-{r["CI high"]:.1f}%.</div>', unsafe_allow_html=True)
        st.markdown("**Most stable groups**")
        for _, r in lo.iterrows():
            st.markdown(f'<div class="obs">🟢 <b>{r.Dimension}: {r.Group}</b> churns at <b>{r["Churn rate %"]:.1f}%</b> '
                        f'({r["Lift vs filtered avg"]:.1f}x the average), n={r.Customers:,}.</div>', unsafe_allow_html=True)
    zero = f[f.Balance == 0]
    bal_churn = f.loc[f.Exited == 1, "Balance"].sum()
    st.markdown("**Exposure**")
    st.markdown(f"- Balances held by churned customers in this filter: **{bal_churn:,.0f}** "
                f"({bal_churn / f.Balance.sum() * 100 if f.Balance.sum() else 0:.1f}% of balances in the filter). Currency is not stated in the file.")
    st.markdown(f"- Zero-balance customers: {len(zero):,} ({len(zero) / total * 100:.1f}% of the filter).")
    st.markdown("**Reading notes**")
    st.markdown("- In the full dataset, number of products, age, country, activity and gender show clear associations with churn; "
                "credit score, tenure, credit card ownership and salary do not.\n"
                "- Gender and age should be reviewed for fairness and legal compliance before they are used in any customer targeting.\n"
                "- Rates are descriptive of a single snapshot; test any intervention in a controlled pilot.")

with t5:
    st.subheader("Filtered data")
    show = f.drop(columns=["CreditBand", "BalanceBand", "Activity", "Products"], errors="ignore")
    st.dataframe(show.head(500), width="stretch", hide_index=True)
    st.caption(f"Showing up to 500 of {len(f):,} filtered rows.")
    st.download_button("Download filtered data (CSV)", show.to_csv(index=False).encode("utf-8"),
                       file_name="filtered_customers.csv", mime="text/csv")
