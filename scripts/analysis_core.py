"""
Core analysis for: Customer Segmentation & Churn Pattern Analytics in European Banking.
Every number used in the paper, executive summary and dashboard is computed here from
European_Bank.csv. CustomerId, Surname and Year are NOT used as analytical features.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve,
                             precision_recall_fscore_support, accuracy_score)
from sklearn.inspection import permutation_importance

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "European_Bank.csv"
CH = BASE / "charts"
CH.mkdir(exist_ok=True)

NAVY, RED, TEAL, GREY, GOLD = "#1B365D", "#C8553D", "#2A9D8F", "#8A94A6", "#E9B44C"
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "axes.titleweight": "bold",
                     "axes.titlesize": 12, "axes.labelsize": 10, "font.family": "DejaVu Sans"})

AGE_BINS, AGE_LABELS = [0, 29, 39, 49, 59, 200], ["18-29", "30-39", "40-49", "50-59", "60+"]
CS_BINS, CS_LABELS = [0, 579, 669, 739, 799, 900], ["Poor (<580)", "Fair (580-669)", "Good (670-739)",
                                                    "Very good (740-799)", "Exceptional (800+)"]
BAL_BINS, BAL_LABELS = [-1, 0, 50000, 100000, 150000, 1e9], ["Zero", "1-50k", "50k-100k", "100k-150k", "150k+"]
TEN_BINS, TEN_LABELS = [-1, 2, 5, 8, 10], ["0-2 yrs", "3-5 yrs", "6-8 yrs", "9-10 yrs"]

SEG_ORDER = ["Product Overload", "Inactive 50+", "Active 50+", "Mid-Life Germany",
             "Mid-Life France & Spain", "Under 40"]
SEG_RULES = {
    "Product Overload": "Holds 3 or 4 products (checked first)",
    "Inactive 50+": "Age 50 or over and not an active member",
    "Active 50+": "Age 50 or over and an active member",
    "Mid-Life Germany": "Age 40-49, Germany, 1-2 products",
    "Mid-Life France & Spain": "Age 40-49, France or Spain, 1-2 products",
    "Under 40": "Age under 40, 1-2 products",
}


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    a = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - a) / d, (c + a) / d


def rate_table(df, col):
    g = df.groupby(col, observed=True)["Exited"].agg(n="count", churned="sum").reset_index()
    g["rate"] = g.churned / g.n * 100
    ci = [wilson(k, n) for k, n in zip(g.churned, g.n)]
    g["lo"] = [c[0] * 100 for c in ci]
    g["hi"] = [c[1] * 100 for c in ci]
    g = g.rename(columns={col: "group"})
    g["group"] = g["group"].astype(str)
    return g


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def add_features(df):
    df = df.copy()
    df["AgeGroup"] = pd.cut(df.Age, AGE_BINS, labels=AGE_LABELS)
    df["CreditBand"] = pd.cut(df.CreditScore, CS_BINS, labels=CS_LABELS)
    df["BalanceBand"] = pd.cut(df.Balance, BAL_BINS, labels=BAL_LABELS)
    df["TenureBand"] = pd.cut(df.Tenure, TEN_BINS, labels=TEN_LABELS)
    conds = [
        df.NumOfProducts >= 3,
        (df.Age >= 50) & (df.IsActiveMember == 0),
        (df.Age >= 50) & (df.IsActiveMember == 1),
        (df.Age >= 40) & (df.Geography == "Germany"),
        (df.Age >= 40),
    ]
    df["Segment"] = np.select(conds, SEG_ORDER[:5], default="Under 40")
    return df


# ------------------------------------------------------------------ chart helpers
def _bar_rate(ax, t, title, xlabel, base, rotate=0, color=NAVY, highlight_max=True):
    cols = [color] * len(t)
    if highlight_max:
        cols[int(t.rate.values.argmax())] = RED
    ax.bar(t.group, t.rate, color=cols, width=0.65)
    ax.errorbar(t.group, t.rate, yerr=[np.clip(t.rate - t.lo, 0, None), np.clip(t.hi - t.rate, 0, None)], fmt="none", ecolor="#333", capsize=3, lw=1)
    ax.axhline(base, color=GREY, ls="--", lw=1.2, label=f"Overall churn {base:.1f}%")
    for i, (r, n, h) in enumerate(zip(t.rate, t.n, t.hi)):
        ax.text(i, h + 1.0, f"{r:.1f}%\n(n={n:,})", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, max(t.hi.max() * 1.25, base * 1.5))
    ax.set_ylabel("Churn rate (%)")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    if rotate:
        plt.setp(ax.get_xticklabels(), rotation=rotate, ha="right")


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(CH / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_charts(df, R):
    base = R["overall_rate"]
    # 1 churn distribution
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
    counts = [R["retained"], R["churned"]]
    ax[0].bar(["Retained", "Churned"], counts, color=[TEAL, RED], width=0.55)
    for i, c in enumerate(counts):
        ax[0].text(i, c + 120, f"{c:,}", ha="center", fontweight="bold")
    ax[0].set_ylabel("Customers")
    ax[0].set_title("Customer counts by outcome")
    ax[0].set_ylim(0, max(counts) * 1.15)
    ax[1].pie(counts, labels=["Retained", "Churned"], colors=[TEAL, RED], autopct="%1.2f%%", startangle=90,
              wedgeprops={"width": 0.45, "edgecolor": "white"}, textprops={"fontsize": 9})
    ax[1].set_title("Share of customers")
    _save(fig, "01_churn_distribution.png")

    # 2 geography
    fig, ax = plt.subplots(figsize=(6.5, 4))
    _bar_rate(ax, R["geo"], "Churn rate by country (95% CI)", "Country", base)
    _save(fig, "02_churn_by_country.png")

    # 3 age
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    _bar_rate(ax[0], R["age"], "Churn rate by age group (95% CI)", "Age group", base)
    for v, c, lab in [(0, TEAL, "Retained"), (1, RED, "Churned")]:
        sns.kdeplot(df[df.Exited == v].Age, ax=ax[1], color=c, fill=True, alpha=0.3, label=lab, bw_adjust=1.2)
    ax[1].set_title("Age distribution: churned vs retained")
    ax[1].set_xlabel("Age (years)")
    ax[1].legend(frameon=False)
    _save(fig, "03_churn_by_age.png")

    # 4 gender
    fig, ax = plt.subplots(figsize=(5.5, 4))
    _bar_rate(ax, R["gender"], "Churn rate by gender (95% CI)", "Gender", base)
    _save(fig, "04_churn_by_gender.png")

    # 5 activity
    fig, ax = plt.subplots(figsize=(5.5, 4))
    t = R["active"].copy()
    t["group"] = t.group.map({"0": "Inactive", "1": "Active"})
    _bar_rate(ax, t, "Churn rate by membership activity (95% CI)", "Membership status", base)
    _save(fig, "05_churn_by_activity.png")

    # 6 products
    fig, ax = plt.subplots(figsize=(6.5, 4))
    t = R["products"].copy()
    t["group"] = t.group.map(lambda x: f"{x} product" + ("s" if x != "1" else ""))
    _bar_rate(ax, t, "Churn rate by number of products (95% CI)", "Products held", base)
    _save(fig, "06_churn_by_products.png")

    # 7 credit score
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
    _bar_rate(ax[0], R["credit"], "Churn rate by credit score band (95% CI)", "Credit score band", base, rotate=25)
    for v, c, lab in [(0, TEAL, "Retained"), (1, RED, "Churned")]:
        sns.kdeplot(df[df.Exited == v].CreditScore, ax=ax[1], color=c, fill=True, alpha=0.3, label=lab)
    ax[1].set_title("Credit score distribution: churned vs retained")
    ax[1].set_xlabel("Credit score")
    ax[1].legend(frameon=False)
    _save(fig, "07_credit_score_analysis.png")

    # 8 balance
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
    _bar_rate(ax[0], R["balance"], "Churn rate by balance band (95% CI)", "Account balance band", base, rotate=25)
    nz = df[df.Balance > 0]
    for v, c, lab in [(0, TEAL, "Retained"), (1, RED, "Churned")]:
        sns.kdeplot(nz[nz.Exited == v].Balance, ax=ax[1], color=c, fill=True, alpha=0.3, label=lab)
    ax[1].set_title("Balance distribution (non-zero balances)")
    ax[1].set_xlabel("Balance")
    ax[1].legend(frameon=False)
    _save(fig, "08_balance_analysis.png")

    # 9 correlation heatmap
    cols = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard",
            "IsActiveMember", "EstimatedSalary", "Exited"]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6.2))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlation heatmap (Pearson)")
    _save(fig, "09_correlation_heatmap.png")

    # 10 tenure and card
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    _bar_rate(ax[0], R["tenure"], "Churn rate by tenure band (95% CI)", "Tenure", base, highlight_max=False)
    t = R["card"].copy()
    t["group"] = t.group.map({"0": "No credit card", "1": "Has credit card"})
    _bar_rate(ax[1], t, "Churn rate by credit card ownership (95% CI)", "Credit card", base, highlight_max=False)
    _save(fig, "10_tenure_and_card.png")

    # 11 age x activity heatmap
    piv = df.pivot_table(index="AgeGroup", columns="IsActiveMember", values="Exited", aggfunc="mean", observed=True) * 100
    cnt = df.pivot_table(index="AgeGroup", columns="IsActiveMember", values="Exited", aggfunc="count", observed=True)
    piv.columns, cnt.columns = ["Inactive", "Active"], ["Inactive", "Active"]
    lab = piv.round(1).astype(str) + "%\n(n=" + cnt.astype(str) + ")"
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    sns.heatmap(piv, annot=lab, fmt="", cmap="Reds", vmin=0, vmax=100, linewidths=0.5,
                cbar_kws={"label": "Churn rate (%)"}, ax=ax)
    ax.set_title("Churn rate by age group and activity")
    ax.set_xlabel("Membership status")
    ax.set_ylabel("Age group")
    _save(fig, "11_age_activity_heatmap.png")

    # 12 geography x gender
    gg = df.groupby(["Geography", "Gender"])["Exited"].mean().reset_index()
    gg["Exited"] *= 100
    fig, ax = plt.subplots(figsize=(6.5, 4))
    sns.barplot(data=gg, x="Geography", y="Exited", hue="Gender", palette=[RED, NAVY], ax=ax)
    for c in ax.containers:
        ax.bar_label(c, fmt="%.1f%%", fontsize=8, padding=2)
    ax.axhline(base, color=GREY, ls="--", lw=1.2)
    ax.set_ylabel("Churn rate (%)")
    ax.set_title("Churn rate by country and gender")
    ax.legend(frameon=False)
    _save(fig, "12_country_gender.png")

    # 13 segments
    seg = R["seg"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    s2 = seg.iloc[::-1]
    cols_ = [RED if r == seg.churn_rate.max() else NAVY for r in s2.churn_rate]
    ax[0].barh(s2.segment, s2.churn_rate, color=cols_)
    ax[0].axvline(base, color=GREY, ls="--", lw=1.2)
    for i, (r, n) in enumerate(zip(s2.churn_rate, s2.n)):
        ax[0].text(r + 1, i, f"{r:.1f}%  (n={n:,})", va="center", fontsize=8)
    ax[0].set_xlim(0, 110)
    ax[0].set_xlabel("Churn rate (%)")
    ax[0].set_title("Churn rate by customer segment")
    w = 0.38
    y = np.arange(len(seg))
    ax[1].barh(y + w / 2, seg.share_customers[::-1], w, color=GREY, label="Share of customers")
    ax[1].barh(y - w / 2, seg.share_churners[::-1], w, color=RED, label="Share of all churners")
    ax[1].set_yticks(y)
    ax[1].set_yticklabels(s2.segment)
    ax[1].set_xlabel("Percent (%)")
    ax[1].set_title("Segment share of customers vs churners")
    ax[1].legend(frameon=False, fontsize=8)
    _save(fig, "13_segments.png")

    # 14 model
    m = R["model"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, (fpr, tpr, auc) in m["roc"].items():
        ax[0].plot(fpr, tpr, lw=2, label=f"{name} (AUC {auc:.3f})", color=NAVY if "Gradient" in name else GOLD)
    ax[0].plot([0, 1], [0, 1], ls="--", color=GREY)
    ax[0].set_xlabel("False positive rate")
    ax[0].set_ylabel("True positive rate")
    ax[0].set_title("ROC curves on held-out test set")
    ax[0].legend(frameon=False, loc="lower right")
    imp = m["importance"].sort_values("importance").tail(10)
    ax[1].barh(imp.feature, imp.importance, xerr=imp["std"], color=NAVY, ecolor="#333", capsize=2)
    ax[1].set_xlabel("Drop in test ROC-AUC when feature is shuffled")
    ax[1].set_title("Permutation importance (gradient boosting)")
    _save(fig, "14_model_evaluation.png")


# ------------------------------------------------------------------ main analysis
def run(make_plots=True):
    df = pd.read_csv(DATA)
    R = {}
    R["shape"] = df.shape
    R["missing"] = int(df.isna().sum().sum())
    R["duplicates"] = int(df.duplicated().sum())
    R["unique_ids"] = int(df.CustomerId.nunique())
    R["years"] = sorted(df.Year.unique().tolist())
    df = add_features(df)

    R["total"] = len(df)
    R["churned"] = int(df.Exited.sum())
    R["retained"] = R["total"] - R["churned"]
    R["overall_rate"] = R["churned"] / R["total"] * 100
    R["retention_rate"] = 100 - R["overall_rate"]
    R["zero_balance_share"] = float((df.Balance == 0).mean() * 100)
    R["germany_zero_balance"] = int(((df.Geography == "Germany") & (df.Balance == 0)).sum())
    R["describe"] = df[["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]].describe().T

    R["geo"] = rate_table(df, "Geography")
    R["gender"] = rate_table(df, "Gender")
    R["age"] = rate_table(df, "AgeGroup")
    R["credit"] = rate_table(df, "CreditBand")
    R["tenure"] = rate_table(df, "TenureBand")
    R["balance"] = rate_table(df, "BalanceBand")
    R["products"] = rate_table(df, "NumOfProducts")
    R["card"] = rate_table(df, "HasCrCard")
    R["active"] = rate_table(df, "IsActiveMember")
    sal_q = pd.qcut(df.EstimatedSalary, 4, labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"])
    d2 = df.assign(SalaryQuartile=sal_q)
    R["salary"] = rate_table(d2, "SalaryQuartile")
    R["geo_gender"] = df.groupby(["Geography", "Gender"])["Exited"].agg(n="count", rate="mean").reset_index()
    R["geo_gender"]["rate"] *= 100
    R["age_active"] = df.groupby(["AgeGroup", "IsActiveMember"], observed=True)["Exited"].agg(n="count", rate="mean").reset_index()
    R["age_active"]["rate"] *= 100

    # ---- statistical tests
    chi_rows = []
    for var, lab in [("Geography", "Geography"), ("Gender", "Gender"), ("AgeGroup", "Age group"),
                     ("CreditBand", "Credit score band"), ("TenureBand", "Tenure band"),
                     ("BalanceBand", "Balance band"), ("NumOfProducts", "Number of products"),
                     ("HasCrCard", "Credit card ownership"), ("IsActiveMember", "Active membership")]:
        ct = pd.crosstab(df[var], df.Exited)
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
        chi_rows.append({"variable": lab, "chi2": chi2, "dof": dof, "p_value": p, "cramers_v": v})
    R["chi"] = pd.DataFrame(chi_rows).sort_values("cramers_v", ascending=False).reset_index(drop=True)

    mw_rows = []
    ch, rt = df[df.Exited == 1], df[df.Exited == 0]
    for var in ["Age", "CreditScore", "Tenure", "Balance", "EstimatedSalary"]:
        u, p = stats.mannwhitneyu(ch[var], rt[var], alternative="two-sided")
        mw_rows.append({"variable": var, "median_churned": ch[var].median(), "median_retained": rt[var].median(),
                        "mean_churned": ch[var].mean(), "mean_retained": rt[var].mean(),
                        "u_stat": u, "p_value": p, "prob_superiority": u / (len(ch) * len(rt))})
    R["mw"] = pd.DataFrame(mw_rows)
    corr = df[["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard",
               "IsActiveMember", "EstimatedSalary", "Exited"]].corr()["Exited"].drop("Exited")
    R["corr_exited"] = corr.sort_values(key=abs, ascending=False)
    # balance confound check: balance effect among non-German, non-zero etc.
    R["churn_zero_bal"] = float(df[df.Balance == 0].Exited.mean() * 100)
    R["churn_nonzero_bal"] = float(df[df.Balance > 0].Exited.mean() * 100)
    nong = df[df.Geography != "Germany"]
    R["churn_zero_bal_nong"] = float(nong[nong.Balance == 0].Exited.mean() * 100)
    R["churn_nonzero_bal_nong"] = float(nong[nong.Balance > 0].Exited.mean() * 100)

    # ---- segmentation
    total_bal = df.Balance.sum()
    churn_bal_total = df[df.Exited == 1].Balance.sum()
    R["total_balance"] = float(total_bal)
    R["churned_balance"] = float(churn_bal_total)
    R["churned_balance_share"] = float(churn_bal_total / total_bal * 100)
    rows = []
    for s in SEG_ORDER:
        d = df[df.Segment == s]
        rows.append({
            "segment": s, "rule": SEG_RULES[s], "n": len(d), "churned": int(d.Exited.sum()),
            "churn_rate": d.Exited.mean() * 100, "share_customers": len(d) / len(df) * 100,
            "share_churners": d.Exited.sum() / df.Exited.sum() * 100,
            "avg_age": d.Age.mean(), "avg_balance": d.Balance.mean(),
            "pct_active": d.IsActiveMember.mean() * 100,
            "total_balance": d.Balance.sum(),
            "churned_balance": d[d.Exited == 1].Balance.sum(),
            "share_churned_balance": d[d.Exited == 1].Balance.sum() / churn_bal_total * 100,
        })
    seg = pd.DataFrame(rows)
    lo_hi = [wilson(k, n) for k, n in zip(seg.churned, seg.n)]
    seg["lo"] = [x[0] * 100 for x in lo_hi]
    seg["hi"] = [x[1] * 100 for x in lo_hi]
    R["seg"] = seg

    # ---- baseline predictive model (proper stratified split; ids/surname/year excluded)
    feats = ["CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance", "NumOfProducts",
             "HasCrCard", "IsActiveMember", "EstimatedSalary"]
    cat = ["Geography", "Gender"]
    num = [f for f in feats if f not in cat]
    X, y = df[feats], df.Exited
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    models = {
        "Logistic regression": Pipeline([("pre", ColumnTransformer(
            [("c", OneHotEncoder(), cat), ("n", StandardScaler(), num)])),
            ("clf", LogisticRegression(max_iter=1000))]),
        "Gradient boosting": Pipeline([("pre", ColumnTransformer(
            [("c", OneHotEncoder(), cat)], remainder="passthrough")),
            ("clf", HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=4, random_state=42))]),
    }
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    mrows, roc = [], {}
    for name, pipe in models.items():
        cv = cross_val_score(pipe, Xtr, ytr, cv=skf, scoring="roc_auc")
        pipe.fit(Xtr, ytr)
        pr = pipe.predict_proba(Xte)[:, 1]
        pred = (pr >= 0.5).astype(int)
        p_, r_, f_, _ = precision_recall_fscore_support(yte, pred, average="binary", zero_division=0)
        order = np.argsort(-pr)
        k20, k10 = int(0.2 * len(yte)), int(0.1 * len(yte))
        mrows.append({"model": name, "cv_auc_mean": cv.mean(), "cv_auc_sd": cv.std(),
                      "test_auc": roc_auc_score(yte, pr), "test_pr_auc": average_precision_score(yte, pr),
                      "accuracy": accuracy_score(yte, pred), "precision": p_, "recall": r_, "f1": f_,
                      "top10_churn_rate": yte.values[order[:k10]].mean() * 100,
                      "top20_capture": yte.values[order[:k20]].sum() / yte.sum() * 100})
        fpr, tpr, _ = roc_curve(yte, pr)
        roc[name] = (fpr, tpr, roc_auc_score(yte, pr))
    gb = models["Gradient boosting"]
    pi = permutation_importance(gb, Xte, yte, scoring="roc_auc", n_repeats=10, random_state=42)
    imp = pd.DataFrame({"feature": Xte.columns, "importance": pi.importances_mean, "std": pi.importances_std})
    R["model"] = {"table": pd.DataFrame(mrows), "roc": roc, "importance": imp.sort_values("importance", ascending=False),
                  "n_train": len(Xtr), "n_test": len(Xte), "test_base_rate": yte.mean() * 100,
                  "majority_acc": (1 - yte.mean()) * 100}
    if make_plots:
        make_charts(df, R)
    return df, R


if __name__ == "__main__":
    d, R = run()
    pd.set_option("display.width", 200, "display.max_columns", 30)
    for k in ["geo", "gender", "age", "credit", "tenure", "balance", "products", "card", "active", "salary",
              "seg", "chi", "mw"]:
        print("==", k)
        print(R[k].round(3))
    print(R["corr_exited"].round(3))
    print(R["model"]["table"].round(3))
    print(R["model"]["importance"].round(4))
    print({k: R[k] for k in ["overall_rate", "total_balance", "churned_balance", "churned_balance_share",
                             "zero_balance_share", "germany_zero_balance", "churn_zero_bal", "churn_nonzero_bal",
                             "churn_zero_bal_nong", "churn_nonzero_bal_nong"]})
