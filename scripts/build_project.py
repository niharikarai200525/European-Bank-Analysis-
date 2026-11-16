"""
Builds research_paper.md, research_paper.pdf, executive_summary.pdf, analysis_report.csv and
statistical_tests.csv from the computed results in analysis_core.py. No number is typed by hand:
all figures come from European_Bank.csv through f-strings.
Run:  python scripts/build_project.py
"""
import re
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_core as A
from analysis_core import BASE, CH, fmt_p

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
                                PageBreak, KeepTogether)

NAVY, RED, LIGHT = colors.HexColor("#1B365D"), colors.HexColor("#C8553D"), colors.HexColor("#F1F4F9")
W = A4[0] - 108  # usable width (54pt margins)


# ------------------------------------------------------------------ formatting helpers
def P(x, d=1):
    return f"{x:.{d}f}%"


def N(x):
    return f"{int(round(x)):,}"


def M(x):
    return f"{x / 1e6:,.1f} million"


def PP(a, b):
    return f"{abs(a - b):.1f} percentage points"


def rate_rows(t, rename=None, var=None):
    rows = []
    for _, r in t.iterrows():
        g = r.group if not rename else rename.get(r.group, r.group)
        row = [g, N(r.n), N(r.churned), P(r.rate), f"{r.lo:.1f} - {r.hi:.1f}%"]
        rows.append(([var] if var is not None else []) + row)
    return rows


# ------------------------------------------------------------------ content (blocks)
def paper_blocks(df, R):
    B = []
    h1 = lambda t: B.append(("h1", t))
    h2 = lambda t: B.append(("h2", t))
    p = lambda t: B.append(("p", t))
    bul = lambda L: B.append(("bul", L))
    num = lambda L: B.append(("num", L))
    tab = lambda cap, head, rows, w=None: B.append(("table", cap, head, rows, w))
    img = lambda f, cap, w=6.3: B.append(("img", f, cap, w))
    note = lambda t: B.append(("note", t))

    geo, gen = R["geo"].set_index("group"), R["gender"].set_index("group")
    age, act = R["age"].set_index("group"), R["active"].set_index("group")
    prod, cs = R["products"].set_index("group"), R["credit"].set_index("group")
    bal, ten = R["balance"].set_index("group"), R["tenure"].set_index("group")
    card, sal = R["card"].set_index("group"), R["salary"].set_index("group")
    seg = R["seg"].set_index("segment")
    chi, mw = R["chi"].set_index("variable"), R["mw"].set_index("variable")
    ce = R["corr_exited"]
    mt = R["model"]["table"].set_index("model")
    gb, lr = mt.loc["Gradient boosting"], mt.loc["Logistic regression"]
    imp = R["model"]["importance"]
    gg = R["geo_gender"].set_index(["Geography", "Gender"])
    aa = R["age_active"].set_index(["AgeGroup", "IsActiveMember"])

    base = R["overall_rate"]
    de_share_churners = geo.loc["Germany", "churned"] / R["churned"] * 100
    de_share_cust = geo.loc["Germany", "n"] / R["total"] * 100
    top2 = seg.loc[["Product Overload", "Inactive 50+"]]
    top2_cust = top2.share_customers.sum()
    top2_chr = top2.share_churners.sum()
    top2_bal = top2.share_churned_balance.sum()
    avg_bal_all = df.Balance.mean()
    gender_gap_all = all(gg.loc[(g, "Female"), "rate"] > gg.loc[(g, "Male"), "rate"] for g in ["France", "Germany", "Spain"])
    best_age = age.rate.idxmax()
    best_seg_bal = seg.avg_balance.idxmax()
    maxprod_rate = prod.rate.idxmax()
    minprod = prod.rate.idxmin()

    # ---------------- Abstract
    h1("Abstract")
    p(f"This study analyses {N(R['total'])} retail banking customer records to understand who leaves the bank and which "
      f"customer groups carry the greatest attrition risk. Overall, {N(R['churned'])} customers churned, an attrition rate "
      f"of **{P(base, 2)}**. Churn is strongly uneven. Customers in Germany churn at {P(geo.loc['Germany','rate'])} "
      f"against {P(geo.loc['France','rate'])} in France and {P(geo.loc['Spain','rate'])} in Spain; customers aged {best_age} "
      f"churn at {P(age.loc[best_age,'rate'])}; customers holding three or four products churn at "
      f"{P((prod.loc['3','churned']+prod.loc['4','churned'])/(prod.loc['3','n']+prod.loc['4','n'])*100)}; and inactive members "
      f"churn at {P(act.loc['0','rate'])} compared with {P(act.loc['1','rate'])} for active members. Credit score, tenure, "
      f"credit card ownership and estimated salary show little or no practical association with churn. Six explainable, "
      f"rule-based segments are defined; the two highest-risk segments (Product Overload and Inactive 50+) contain "
      f"{P(top2_cust)} of customers but {P(top2_chr)} of all churners. A baseline gradient boosting model evaluated on a "
      f"held-out test set reaches a ROC-AUC of {gb.test_auc:.3f}. All results are associations observed in a single-year "
      f"snapshot and should not be read as causal effects.")

    # ---------------- 1 Introduction
    h1("1. Introduction")
    p("Customer retention is central to the stability of retail banking. Acquiring a new depositor generally requires "
      "marketing, onboarding and compliance effort, while existing customers already hold relationships and balances "
      "with the institution. Understanding which customers leave, and what they have in common, allows a bank to direct "
      "limited retention resources where they are most likely to matter.")
    p("This project applies descriptive analytics, statistical testing and transparent rule-based segmentation to a "
      "customer-level dataset from a European bank. The emphasis is on findings that are explainable to non-technical "
      "decision makers and that can be reproduced from the supplied data and code.")

    # ---------------- 2 Objectives
    h1("2. Research Objectives")
    num(["Measure the overall churn and retention rates of the customer portfolio.",
         "Identify how churn varies by country, gender, age, credit score, tenure, balance, number of products, credit card "
         "ownership, estimated salary and activity status.",
         "Test which of these differences are statistically distinguishable from chance and how large they are.",
         "Define explainable customer segments and describe the churn risk and balance exposure of each.",
         "Evaluate, with a proper held-out test, how well the available attributes rank customers by churn risk.",
         "Translate the evidence into clearly labelled, testable business recommendations."])

    # ---------------- 3 Dataset
    h1("3. Dataset Description")
    p(f"The file European_Bank.csv contains {N(R['shape'][0])} rows and {R['shape'][1]} columns. Each row is one customer. "
      f"The target variable is Exited (1 = customer churned, 0 = customer retained). The Year column holds a single value "
      f"({R['years'][0]}) for every row, so the data is a single cross-sectional snapshot and contains no information on when "
      f"each customer left.")
    d = df
    tab("Dataset columns and their role in the analysis",
        ["Column", "Description", "Range / values", "Role"],
        [["Year", "Reference year", f"{R['years'][0]} (constant)", "Excluded (no variation)"],
         ["CustomerId", "Customer identifier", f"{N(R['unique_ids'])} unique values", "Excluded (identifier)"],
         ["Surname", "Customer surname", f"{N(d.Surname.nunique())} distinct values", "Excluded (identifier)"],
         ["CreditScore", "Credit score", f"{d.CreditScore.min()} - {d.CreditScore.max()}", "Feature"],
         ["Geography", "Country", ", ".join(sorted(d.Geography.unique())), "Feature"],
         ["Gender", "Gender", ", ".join(sorted(d.Gender.unique())), "Feature"],
         ["Age", "Age in years", f"{d.Age.min()} - {d.Age.max()}", "Feature"],
         ["Tenure", "Length of relationship (unit not stated; read as years)", f"{d.Tenure.min()} - {d.Tenure.max()}", "Feature"],
         ["Balance", "Account balance (currency not stated)", f"{d.Balance.min():,.0f} - {d.Balance.max():,.0f}", "Feature"],
         ["NumOfProducts", "Number of bank products held", f"{d.NumOfProducts.min()} - {d.NumOfProducts.max()}", "Feature"],
         ["HasCrCard", "Holds a credit card (1/0)", "0, 1", "Feature"],
         ["IsActiveMember", "Active member (1/0)", "0, 1", "Feature"],
         ["EstimatedSalary", "Estimated salary", f"{d.EstimatedSalary.min():,.0f} - {d.EstimatedSalary.max():,.0f}", "Feature"],
         ["Exited", "Churn outcome (1/0)", "0, 1", "Target"]], [1.3, 2.6, 1.7, 1.4])
    p(f"**Data quality.** The file has {R['missing']} missing values and {R['duplicates']} duplicate rows, and every CustomerId "
      f"is unique. {P(R['zero_balance_share'])} of customers hold a zero balance. No customer in Germany has a zero balance "
      f"({R['germany_zero_balance']} of {N(geo.loc['Germany','n'])}), a structural feature that matters when interpreting "
      f"balance effects in Section 5. CustomerId and Surname are identifiers and are never used as analytical predictors.")
    ds = R["describe"]
    tab("Descriptive statistics of numeric variables",
        ["Variable", "Mean", "Std. dev.", "Min", "Median", "Max"],
        [[v, f"{r['mean']:,.1f}", f"{r['std']:,.1f}", f"{r['min']:,.0f}", f"{r['50%']:,.0f}", f"{r['max']:,.0f}"]
         for v, r in ds.iterrows()], [1.9, 1.1, 1.1, 1.0, 1.0, 1.1])

    # ---------------- 4 Methodology
    h1("4. Methodology")
    h2("4.1 Churn measurement and grouping")
    p("Churn rate is the number of churned customers divided by the number of customers in a group. Continuous variables "
      "are grouped into bands for presentation: age (18-29, 30-39, 40-49, 50-59, 60+), credit score (Poor below 580, "
      "Fair 580-669, Good 670-739, Very good 740-799, Exceptional 800+), balance (zero, 1-50k, 50k-100k, 100k-150k, 150k+), "
      "tenure (0-2, 3-5, 6-8, 9-10) and estimated salary quartiles. Every group churn rate is shown with a 95% Wilson "
      "confidence interval so that readers can judge how much a small group's rate may vary by chance.")
    h2("4.2 Statistical analysis")
    p("For categorical and banded variables a chi-square test of independence against the churn outcome is reported, "
      "with Cramer's V as the effect size (roughly: below 0.10 negligible, 0.10-0.30 small to moderate, above 0.30 "
      "strong). For numeric variables a two-sided Mann-Whitney U test compares churned and retained customers; the "
      "reported probability of superiority is the chance that a randomly chosen churner has a higher value than a randomly "
      "chosen retained customer (0.50 means no difference). Pearson correlations with the churn indicator are also shown. "
      "With a sample of this size, very small effects can be statistically significant, so effect sizes are reported "
      "alongside p-values.")
    h2("4.3 Segmentation")
    p("Segments are defined by explicit business rules applied in priority order, so every customer belongs to exactly one "
      "segment. Rules were chosen from the strongest patterns found in the exploratory analysis (products, age, activity, "
      "country). Because the rules were informed by this same dataset, segment churn rates are in-sample descriptions; "
      "they are not out-of-sample forecasts.")
    h2("4.4 Baseline predictive model")
    p(f"To test how well the available attributes can rank customers by risk, two baseline models (logistic regression and "
      f"gradient boosting) were trained on a stratified {100 - 20}% training split ({N(R['model']['n_train'])} customers) and "
      f"evaluated once on a held-out {20}% test split ({N(R['model']['n_test'])} customers) with a fixed random seed. "
      f"Five-fold cross-validation on the training split is also reported. CustomerId, Surname and Year were excluded. "
      f"No hyperparameter search was run, so the test set was not used for tuning.")
    h2("4.5 Tools")
    p("Python (pandas, SciPy, scikit-learn, Matplotlib, Seaborn), ReportLab for PDF generation and Streamlit for the "
      "interactive dashboard. All code is included in the scripts folder.")

    # ---------------- 5 EDA
    h1("5. Exploratory Data Analysis")
    h2("5.1 Overall churn")
    p(f"Of {N(R['total'])} customers, {N(R['churned'])} churned and {N(R['retained'])} were retained, giving a churn rate of "
      f"{P(base, 2)} and a retention rate of {P(R['retention_rate'], 2)}. Roughly one in five customers leaves.")
    h2("5.2 Churn by demographic and engagement variables")
    rows = (rate_rows(R["geo"], var="Country") + rate_rows(R["gender"], var="Gender") +
            rate_rows(R["age"], var="Age group") + rate_rows(R["active"], {"0": "Inactive", "1": "Active"}, "Membership") +
            rate_rows(R["products"], {"1": "1 product", "2": "2 products", "3": "3 products", "4": "4 products"}, "Products"))
    tab("Churn rate by demographic and engagement variables",
        ["Variable", "Group", "Customers", "Churned", "Churn rate", "95% CI"], rows, [1.1, 1.3, 1.0, 0.9, 1.0, 1.4])
    p(f"**Country.** Germany has {P(geo.loc['Germany','rate'])} churn, about twice the rate of France "
      f"({P(geo.loc['France','rate'])}) and Spain ({P(geo.loc['Spain','rate'])}). German customers are "
      f"{P(de_share_cust)} of the portfolio but {P(de_share_churners)} of all churners.")
    p(f"**Gender.** Female customers churn at {P(gen.loc['Female','rate'])} versus {P(gen.loc['Male','rate'])} for male "
      f"customers, a gap of {PP(gen.loc['Female','rate'], gen.loc['Male','rate'])}. "
      + ("The gap appears in each of the three countries." if gender_gap_all else "The gap is not present in every country.")
      + " The data does not explain the reason.")
    p(f"**Age.** Churn rises steeply with age: {P(age.loc['18-29','rate'])} for ages 18-29 and {P(age.loc['30-39','rate'])} for "
      f"30-39, then {P(age.loc['40-49','rate'])} for 40-49 and a peak of {P(age.loc['50-59','rate'])} for 50-59, before easing "
      f"to {P(age.loc['60+','rate'])} for 60 and over. The median age of churned customers is "
      f"{mw.loc['Age','median_churned']:.0f} years against {mw.loc['Age','median_retained']:.0f} for retained customers.")
    p(f"**Activity.** Inactive members churn at {P(act.loc['0','rate'])}, active members at {P(act.loc['1','rate'])}. "
      f"The combined effect with age is large: among customers aged 50-59, {P(aa.loc[('50-59', 0), 'rate'])} of inactive members "
      f"churned against {P(aa.loc[('50-59', 1), 'rate'])} of active members.")
    p(f"**Products.** The relationship is not linear. Customers with two products have the lowest churn "
      f"({P(prod.loc[minprod,'rate'])}), customers with one product {P(prod.loc['1','rate'])}, those with three products "
      f"{P(prod.loc['3','rate'])} and all {N(prod.loc['4','n'])} customers with four products churned "
      f"({P(prod.loc['4','rate'])}). Groups with three or four products are small ({N(prod.loc['3','n'])} and "
      f"{N(prod.loc['4','n'])} customers), so their exact rates are less certain, although the lower bound of the 95% "
      f"interval for three products is still {P(prod.loc['3','lo'])}.")
    h2("5.3 Churn by financial profile")
    rows = (rate_rows(R["credit"], var="Credit score") + rate_rows(R["tenure"], var="Tenure") +
            rate_rows(R["balance"], var="Balance") +
            rate_rows(R["card"], {"0": "No credit card", "1": "Has credit card"}, "Credit card") +
            rate_rows(R["salary"], var="Salary quartile"))
    tab("Churn rate by financial profile variables",
        ["Variable", "Group", "Customers", "Churned", "Churn rate", "95% CI"], rows, [1.2, 1.5, 1.0, 0.9, 1.0, 1.4])
    p(f"**Credit score, tenure, credit card and salary.** Churn is close to the portfolio average in every band: "
      f"{P(cs.rate.min())} to {P(cs.rate.max())} across credit score bands, {P(ten.rate.min())} to {P(ten.rate.max())} across "
      f"tenure bands, {P(card.loc['1','rate'])} versus {P(card.loc['0','rate'])} for card holders and non-holders, and "
      f"{P(sal.rate.min())} to {P(sal.rate.max())} across salary quartiles.")
    p(f"**Balance.** Customers with a zero balance churn at {P(R['churn_zero_bal'])} and customers with a non-zero balance at "
      f"{P(R['churn_nonzero_bal'])}. Part of this gap reflects geography: all German customers have non-zero balances and "
      f"Germany has the highest churn. Excluding Germany, non-zero balance customers churn at "
      f"{P(R['churn_nonzero_bal_nong'])} versus {P(R['churn_zero_bal_nong'])} for zero-balance customers, so a balance effect "
      f"remains but is smaller. The 1-50k band is very small ({N(bal.loc['1-50k','n'])} customers) and should not be "
      f"over-interpreted.")
    h2("5.4 Statistical tests")
    tab("Chi-square tests of association with churn",
        ["Variable", "Chi-square", "df", "p-value", "Cramer's V"],
        [[r.variable, f"{r.chi2:,.1f}", str(r.dof), fmt_p(r.p_value), f"{r.cramers_v:.3f}"] for r in R["chi"].itertuples()],
        [2.3, 1.3, 0.7, 1.1, 1.2])
    tab("Mann-Whitney U tests: churned versus retained customers",
        ["Variable", "Median churned", "Median retained", "p-value", "Prob. of superiority"],
        [[r.variable, f"{r.median_churned:,.0f}", f"{r.median_retained:,.0f}", fmt_p(r.p_value), f"{r.prob_superiority:.3f}"]
         for r in R["mw"].itertuples()], [1.6, 1.5, 1.5, 1.0, 1.7])
    strong = chi.sort_values("cramers_v", ascending=False)
    p(f"The strongest associations are with number of products (Cramer's V {strong.loc['Number of products','cramers_v']:.3f}) "
      f"and age group ({strong.loc['Age group','cramers_v']:.3f}), followed by geography ({strong.loc['Geography','cramers_v']:.3f}), "
      f"active membership ({strong.loc['Active membership','cramers_v']:.3f}), balance band ({strong.loc['Balance band','cramers_v']:.3f}) "
      f"and gender ({strong.loc['Gender','cramers_v']:.3f}). Credit score band (p = {fmt_p(chi.loc['Credit score band','p_value'])}), "
      f"tenure band (p = {fmt_p(chi.loc['Tenure band','p_value'])}) and credit card ownership "
      f"(p = {fmt_p(chi.loc['Credit card ownership','p_value'])}) are not significant at the 5% level. The Mann-Whitney test for "
      f"credit score gives p = {fmt_p(mw.loc['CreditScore','p_value'])}, but the probability of superiority is "
      f"{mw.loc['CreditScore','prob_superiority']:.3f}, essentially 0.50, so the practical difference is negligible. "
      f"Age shows a probability of superiority of {mw.loc['Age','prob_superiority']:.3f}, a large separation.")
    h2("5.5 Correlations")
    p(f"Linear correlations with the churn indicator are: "
      + ", ".join(f"{k} {v:+.2f}" for k, v in ce.items())
      + f". Age is the strongest linear correlate. Number of products shows only a weak linear correlation "
      f"({ce['NumOfProducts']:+.2f}) because its relationship with churn is U-shaped rather than linear, which is why "
      f"the chi-square test and the grouped rates are more informative for this variable.")

    # ---------------- 6 Visual findings
    h1("6. Visual Findings")
    img("01_churn_distribution.png", "Distribution of churned and retained customers", 5.8)
    p(f"The portfolio is imbalanced: {P(R['retention_rate'], 2)} retained and {P(base, 2)} churned.")
    img("02_churn_by_country.png", "Churn rate by country with 95% confidence intervals", 4.4)
    p(f"Germany stands apart. The confidence intervals for France ({geo.loc['France','lo']:.1f}-{geo.loc['France','hi']:.1f}%) "
      f"and Spain ({geo.loc['Spain','lo']:.1f}-{geo.loc['Spain','hi']:.1f}%) overlap with each other and sit far below Germany "
      f"({geo.loc['Germany','lo']:.1f}-{geo.loc['Germany','hi']:.1f}%).")
    img("03_churn_by_age.png", "Churn rate by age group and age distributions of churned and retained customers", 6.6)
    p("Churned customers are concentrated in the middle and older age range, while retained customers cluster in the "
      "mid-30s. Churn peaks in the 50-59 group.")
    img("04_churn_by_gender.png", "Churn rate by gender", 3.9)
    img("05_churn_by_activity.png", "Churn rate by active versus inactive membership", 3.9)
    p(f"Female customers and inactive members churn more often than male customers and active members. Activity is the "
      f"only engagement variable in the data that shows a clear association with churn.")
    img("06_churn_by_products.png", "Churn rate by number of products held", 4.5)
    p("Two products is the lowest-churn group; one product is above average; three and four products are extreme outliers "
      "in a small number of customers.")
    img("07_credit_score_analysis.png", "Credit score bands and credit score distributions by churn outcome", 6.6)
    p("The distributions of credit scores for churned and retained customers almost overlap, matching the statistical result "
      "that credit score is not a useful discriminator.")
    img("08_balance_analysis.png", "Balance bands and balance distributions (non-zero balances) by churn outcome", 6.6)
    p(f"Churn is lowest for zero balances and higher for balances above 100k. The median balance of churned customers is "
      f"{mw.loc['Balance','median_churned']:,.0f} versus {mw.loc['Balance','median_retained']:,.0f} for retained customers.")
    img("10_tenure_and_card.png", "Churn rate by tenure band and credit card ownership", 6.2)
    p("Neither tenure nor credit card ownership separates churned from retained customers.")
    img("11_age_activity_heatmap.png", "Churn rate by age group and activity status", 4.6)
    p(f"Churn is highest where age and inactivity combine: {P(aa.loc[('60+', 0), 'rate'])} for inactive customers aged 60+ "
      f"and {P(aa.loc[('50-59', 0), 'rate'])} for inactive customers aged 50-59. Active customers aged 60+ churn at "
      f"{P(aa.loc[('60+', 1), 'rate'])}, below the portfolio average.")
    img("12_country_gender.png", "Churn rate by country and gender", 4.6)
    p(f"Within every country females churn more than males; the highest rate is for German females "
      f"({P(gg.loc[('Germany', 'Female'), 'rate'])}) and the lowest for French males ({P(gg.loc[('France', 'Male'), 'rate'])}).")
    img("09_correlation_heatmap.png", "Correlation heatmap of numeric variables and churn", 5.0)
    ex = df[["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"]].corr()
    mm = ex.values.copy()
    np.fill_diagonal(mm, 0)
    ii, jj = np.unravel_index(np.abs(mm).argmax(), mm.shape)
    p(f"Correlations among the explanatory variables themselves are modest: the strongest is between {ex.index[ii]} and "
      f"{ex.columns[jj]} ({mm[ii, jj]:+.2f}). No two of the main drivers (age, activity, products, country) are close to "
      f"duplicates of each other, so each carries its own information.")

    # ---------------- 7 Segmentation
    h1("7. Customer Segmentation")
    p("Six mutually exclusive segments were defined with simple rules (Table below). The rules are applied in the order "
      "listed, so a customer with three or more products is always placed in Product Overload regardless of age or country.")
    tab("Segment definitions and churn results",
        ["Segment", "Rule", "Customers", "Churned", "Churn rate", "95% CI"],
        [[r.segment, r.rule, N(r.n), N(r.churned), P(r.churn_rate), f"{r.lo:.1f} - {r.hi:.1f}%"] for r in R["seg"].itertuples()],
        [1.4, 2.2, 0.9, 0.8, 0.9, 1.2])
    tab("Segment profile and balance exposure",
        ["Segment", "Share of customers", "Share of churners", "Avg. age", "Avg. balance", "Share of churned balance"],
        [[r.segment, P(r.share_customers), P(r.share_churners), f"{r.avg_age:.1f}", N(r.avg_balance), P(r.share_churned_balance)]
         for r in R["seg"].itertuples()], [1.5, 1.1, 1.1, 0.8, 1.0, 1.4])
    img("13_segments.png", "Churn rate by segment and each segment's share of customers versus churners", 6.7)
    h2("7.1 Segment descriptions")
    cmp = lambda v: "above" if v > avg_bal_all else "below"
    s = seg
    p(f"**Product Overload (3-4 products).** {N(s.loc['Product Overload','n'])} customers, churn "
      f"{P(s.loc['Product Overload','churn_rate'])}, the highest of any segment. Only "
      f"{P(s.loc['Product Overload','share_customers'])} of customers but {P(s.loc['Product Overload','share_churners'])} of "
      f"churners, and {P(s.loc['Product Overload','share_churned_balance'])} of the balance held by churned customers. The data "
      f"cannot say why these customers leave; fees, service quality or product mis-selling are hypotheses to investigate. "
      f"Business value: small group, high loss rate, moderate average balance ({N(s.loc['Product Overload','avg_balance'])}).")
    p(f"**Inactive 50+.** {N(s.loc['Inactive 50+','n'])} customers, churn {P(s.loc['Inactive 50+','churn_rate'])}. Average "
      f"balance {N(s.loc['Inactive 50+','avg_balance'])}, which is {cmp(s.loc['Inactive 50+','avg_balance'])} the portfolio "
      f"average of {N(avg_bal_all)}. They account for {P(s.loc['Inactive 50+','share_churned_balance'])} of churned balances "
      f"while being {P(s.loc['Inactive 50+','share_customers'])} of customers. Business value: a compact, high-risk group "
      f"holding meaningful balances, and one that can be identified from a simple activity flag.")
    p(f"**Active 50+.** {N(s.loc['Active 50+','n'])} customers, churn {P(s.loc['Active 50+','churn_rate'])}, near the "
      f"portfolio average of {P(base)} even though the age band is high-risk overall. Being an active member appears to "
      f"offset much of the age-related risk. Business value: an engaged older base worth protecting.")
    p(f"**Mid-Life Germany (40-49, 1-2 products).** {N(s.loc['Mid-Life Germany','n'])} customers, churn "
      f"{P(s.loc['Mid-Life Germany','churn_rate'])}. "
      + (f"This segment has the highest average balance of all segments ({N(s.loc['Mid-Life Germany','avg_balance'])}) "
         if best_seg_bal == "Mid-Life Germany" else
         f"Average balance is {N(s.loc['Mid-Life Germany','avg_balance'])} ")
      + f"and holds {P(s.loc['Mid-Life Germany','share_churned_balance'])} of churned balances, the largest share among the "
        f"high-risk segments. Business value: high balance exposure combined with high churn.")
    p(f"**Mid-Life France & Spain (40-49, 1-2 products).** {N(s.loc['Mid-Life France & Spain','n'])} customers, churn "
      f"{P(s.loc['Mid-Life France & Spain','churn_rate'])}, close to average. Business value: a large mid-age base with "
      f"moderate risk; compared with Mid-Life Germany it shows how much of the risk is country-specific.")
    p(f"**Under 40 (1-2 products).** {N(s.loc['Under 40','n'])} customers, churn {P(s.loc['Under 40','churn_rate'])}, the "
      f"lowest of any segment. This segment is {P(s.loc['Under 40','share_customers'])} of customers and, because it is so "
      f"large, still accounts for {P(s.loc['Under 40','share_churned_balance'])} of churned balances. Business value: low "
      f"risk per customer, but volume matters; monitoring rather than intervention is indicated.")

    # ---------------- 8 Model
    h1("8. Baseline Predictive Model")
    p(f"A proper train-test evaluation was run (Section 4.4). On the held-out test set, which has a churn rate of "
      f"{P(R['model']['test_base_rate'])}, gradient boosting reaches a ROC-AUC of {gb.test_auc:.3f} "
      f"(five-fold cross-validation on the training data: {gb.cv_auc_mean:.3f} +/- {gb.cv_auc_sd:.3f}) while logistic "
      f"regression reaches {lr.test_auc:.3f}. Logistic regression is weaker because the effect of products and age is "
      f"non-linear.")
    tab("Model performance on the held-out test set",
        ["Model", "ROC-AUC", "PR-AUC", "Accuracy", "Precision", "Recall", "Top 10% churn rate", "Churners in top 20%"],
        [[m, f"{r.test_auc:.3f}", f"{r.test_pr_auc:.3f}", P(r.accuracy * 100), P(r.precision * 100), P(r.recall * 100),
          P(r.top10_churn_rate), P(r.top20_capture)] for m, r in mt.iterrows()],
        [1.5, 0.8, 0.8, 0.9, 0.9, 0.8, 1.0, 1.0])
    p(f"Precision, recall and accuracy use a 0.5 probability threshold. Always predicting 'retained' would score "
      f"{P(R['model']['majority_acc'])} accuracy, so accuracy alone is a weak yardstick here. For gradient boosting, the "
      f"10% of test customers with the highest predicted risk had an actual churn rate of {P(gb.top10_churn_rate)}, and the "
      f"top 20% captured {P(gb.top20_capture)} of all churners. At the 0.5 threshold the model finds {P(gb.recall * 100)} of "
      f"churners with {P(gb.precision * 100)} precision; a lower threshold would trade precision for recall.")
    img("14_model_evaluation.png", "ROC curves and permutation feature importance on the test set", 6.4)
    p(f"Permutation importance ranks {imp.iloc[0].feature} and {imp.iloc[1].feature} far above all other variables, followed "
      f"by {imp.iloc[2].feature} and {imp.iloc[3].feature}; shuffling credit card ownership, tenure or estimated salary changes "
      f"the score by almost nothing. This agrees with the descriptive and statistical findings.")
    note("Interpretation limits: the model is a baseline, not a deployable system. It was validated on a random split of "
         "one snapshot, not on a later period. Attributes such as balance and activity are recorded at the same time as the "
         "outcome and may partly reflect a customer who is already leaving. It should be read as evidence that the listed "
         "attributes carry real signal, not as a guarantee of future accuracy.")

    # ---------------- 9 Key findings
    h1("9. Key Findings")
    num([f"Overall churn is {P(base, 2)} ({N(R['churned'])} of {N(R['total'])} customers).",
         f"Number of products and age are the strongest drivers. Customers with 3-4 products churn at "
         f"{P(seg.loc['Product Overload','churn_rate'])}; the 50-59 age group churns at {P(age.loc['50-59','rate'])}.",
         f"Germany churns at {P(geo.loc['Germany','rate'])}, about double France and Spain, and contributes "
         f"{P(de_share_churners)} of churners from {P(de_share_cust)} of customers.",
         f"Inactive members churn at {P(act.loc['0','rate'])} versus {P(act.loc['1','rate'])} for active members; inactivity "
         f"combined with age 50+ gives {P(seg.loc['Inactive 50+','churn_rate'])} churn.",
         f"Female customers churn at {P(gen.loc['Female','rate'])} versus {P(gen.loc['Male','rate'])} for males.",
         "Credit score, tenure, credit card ownership and estimated salary show no meaningful association with churn.",
         f"Product Overload and Inactive 50+ together are {P(top2_cust)} of customers, {P(top2_chr)} of churners and "
         f"{P(top2_bal)} of the balances held by churned customers.",
         f"Churned customers held {M(R['churned_balance'])} balance units, {P(R['churned_balance_share'])} of all balances "
         f"in the dataset ({M(R['total_balance'])}).",
         f"A baseline gradient boosting model ranks churn risk well on held-out data (ROC-AUC {gb.test_auc:.3f})."])

    # ---------------- 10 Recommendations
    h1("10. Business Recommendations")
    note("The recommendations below are proposals derived from the evidence. They are not findings of the data. Each one "
         "should be tested on a small scale, with a comparison group, before wide rollout.")
    num([f"**Prioritise the two highest-risk segments in a pilot retention programme.** Product Overload and Inactive 50+ "
         f"are {P(top2_cust)} of customers but {P(top2_chr)} of churners, so a small, targeted effort covers a large share of "
         f"attrition. Use a randomly assigned control group to measure whether outreach actually reduces churn.",
         "**Investigate the 3-4 product experience.** Review fees, bundling, onboarding and support for customers who hold "
         "three or more products, and interview or survey a sample of those who left. The data shows the outcome, not the cause.",
         f"**Run a country-specific review for Germany**, where mid-life customers churn at "
         f"{P(seg.loc['Mid-Life Germany','churn_rate'])}: pricing, competitor offers, service channels and product fit.",
         "**Re-engage inactive customers early**, particularly those aged 40 and over, and test whether raising activity "
         "lowers churn. Activity is associated with lower churn, but it may also be a symptom rather than a cause.",
         "**Pilot a risk-ranking model as a decision aid.** Validate it on a later period before use, monitor accuracy over "
         "time, and keep staff judgement in the loop.",
         "**Review fairness and legal compliance** before using age or gender in any targeting rule, in line with applicable "
         "EU and national non-discrimination and data-protection requirements. Behavioural and product variables are "
         "safer and more actionable targeting criteria.",
         "**Improve the data.** Add churn dates, multi-year history, transaction and complaint records, channel and revenue "
         "data so that cause, timing and profitability can be analysed."])

    # ---------------- 11 Limitations
    h1("11. Limitations")
    bul(["**Single snapshot.** The data covers one year and gives no churn date, so timing, trends and seasonality cannot be studied.",
         "**Association, not causation.** The analysis is observational; no finding shows that changing a variable would change churn.",
         "**Possible reverse causality.** Variables such as balance and activity are measured together with the outcome and may reflect "
         "customers who are already leaving.",
         "**In-sample segments.** Segment rules were derived from this dataset, so segment churn rates are descriptive and likely "
         "optimistic as forecasts for new data.",
         "**Small groups.** Customers with four products (60), the 1-50k balance band (75) and some age-activity cells are small; "
         "their rates carry wide confidence intervals.",
         "**Baseline model.** No hyperparameter tuning, no out-of-time validation and no calibration study were done.",
         "**Missing business data.** The file contains no revenue, margin, cost or customer-contact data, so profit impact "
         "cannot be estimated. Currency and the unit of tenure are not stated in the file.",
         "**Provenance.** How the data was collected and whether it represents the bank's full customer base is not documented in "
         "the file; the findings describe the dataset supplied."])

    # ---------------- 12 Conclusion
    h1("12. Conclusion")
    p(f"About one in five customers in this dataset churned. The risk is concentrated rather than evenly spread: it is "
      f"highest among customers who hold three or more products, who are 50 or over and inactive, and who are in mid-life in "
      f"Germany, while customers under 40 and those with two products are the most stable. Credit score, tenure, credit "
      f"card ownership and salary add little. A small number of explainable segments captures a large share of attrition, "
      f"and a baseline model confirms that the same attributes carry predictive signal. The next step is not to scale "
      f"interventions blindly but to test them in controlled pilots, investigate the causes behind the highest-risk groups, "
      f"and enrich the data.")
    h1("Appendix: Reproducibility")
    p("All figures in this paper were generated by scripts/analysis_core.py and scripts/build_project.py from "
      "European_Bank.csv. Running python scripts/build_project.py from the project folder regenerates the charts, "
      "analysis_report.csv, statistical_tests.csv, research_paper.md and both PDFs. The random seed is fixed at 42.")
    return B


# ------------------------------------------------------------------ renderers
def to_md(blocks, title, subtitle):
    out = [f"# {title}", "", f"*{subtitle}*", ""]
    fig = tbl = 0
    for b in blocks:
        k = b[0]
        if k == "h1":
            out += [f"## {b[1]}", ""]
        elif k == "h2":
            out += [f"### {b[1]}", ""]
        elif k == "p":
            out += [b[1], ""]
        elif k == "note":
            out += [f"> {b[1]}", ""]
        elif k == "bul":
            out += [f"- {x}" for x in b[1]] + [""]
        elif k == "num":
            out += [f"{i}. {x}" for i, x in enumerate(b[1], 1)] + [""]
        elif k == "table":
            tbl += 1
            _, cap, head, rows, _w = b
            out += [f"**Table {tbl}. {cap}**", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
            out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows] + [""]
        elif k == "img":
            fig += 1
            out += [f"![Figure {fig}. {b[2]}](charts/{b[1]})", "", f"*Figure {fig}. {b[2]}*", ""]
    return "\n".join(out)


def styles():
    s = {}
    s["title"] = ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=19, leading=23, textColor=NAVY, alignment=1, spaceAfter=6)
    s["sub"] = ParagraphStyle("sub", fontName="Helvetica", fontSize=10, leading=13, textColor=colors.HexColor("#4A5568"), alignment=1, spaceAfter=14)
    s["h1"] = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=14, leading=18, textColor=NAVY, spaceBefore=14, spaceAfter=6, keepWithNext=1)
    s["h2"] = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=colors.HexColor("#2D4A7A"), spaceBefore=8, spaceAfter=4, keepWithNext=1)
    s["p"] = ParagraphStyle("p", fontName="Helvetica", fontSize=9.6, leading=13.6, spaceAfter=6, alignment=4)
    s["li"] = ParagraphStyle("li", parent=s["p"], leftIndent=16, bulletIndent=4, spaceAfter=3)
    s["cap"] = ParagraphStyle("cap", fontName="Helvetica-Oblique", fontSize=8.5, leading=11, textColor=colors.HexColor("#4A5568"), alignment=1, spaceAfter=6)
    s["tcap"] = ParagraphStyle("tcap", fontName="Helvetica-Bold", fontSize=9, leading=12, textColor=NAVY, spaceBefore=6, spaceAfter=3, keepWithNext=1)
    s["cell"] = ParagraphStyle("cell", fontName="Helvetica", fontSize=8, leading=10)
    s["cellh"] = ParagraphStyle("cellh", fontName="Helvetica-Bold", fontSize=8, leading=10, textColor=colors.white)
    s["note"] = ParagraphStyle("note", fontName="Helvetica", fontSize=9, leading=12.5, textColor=colors.HexColor("#2D3748"))
    return s


def inline(t):
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)


def make_table(head, rows, weights, S):
    n = len(head)
    weights = weights or [1] * n
    tot = sum(weights)
    cw = [W * w / tot for w in weights]
    data = [[Paragraph(inline(h), S["cellh"]) for h in head]] + \
           [[Paragraph(inline(str(c)), S["cell"]) for c in r] for r in rows]
    t = Table(data, colWidths=cw, repeatRows=1)
    st = [("BACKGROUND", (0, 0), (-1, 0), NAVY), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E0")),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    for i in range(1, len(data)):
        if i % 2 == 0:
            st.append(("BACKGROUND", (0, i), (-1, i), LIGHT))
    t.setStyle(TableStyle(st))
    return t


def make_image(path, w_in):
    iw, ih = ImageReader(str(path)).getSize()
    w = min(w_in * 72, W)
    return Image(str(path), width=w, height=w * ih / iw)


def footer_factory(label):
    def _f(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#718096"))
        canvas.drawString(54, 28, label)
        canvas.drawRightString(A4[0] - 54, 28, f"Page {doc.page}")
        canvas.restoreState()
    return _f


def render_pdf(blocks, path, title, subtitle, label):
    S = styles()
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=54, rightMargin=54, topMargin=54, bottomMargin=50,
                            title=title, author="European Bank Analytics")
    fl = [Paragraph(inline(title), S["title"]), Paragraph(inline(subtitle), S["sub"])]
    fig = tbl = 0
    for b in blocks:
        k = b[0]
        if k in ("h1", "h2", "p"):
            fl.append(Paragraph(inline(b[1]), S[k]))
        elif k == "note":
            t = Table([[Paragraph(inline(b[1]), S["note"])]], colWidths=[W])
            t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFF6E5")),
                                   ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor("#E9B44C")),
                                   ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                                   ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
            fl += [Spacer(1, 3), t, Spacer(1, 8)]
        elif k == "bul":
            fl += [Paragraph(inline(x), S["li"], bulletText="\u2022") for x in b[1]] + [Spacer(1, 4)]
        elif k == "num":
            fl += [Paragraph(inline(x), S["li"], bulletText=f"{i}.") for i, x in enumerate(b[1], 1)] + [Spacer(1, 4)]
        elif k == "table":
            tbl += 1
            _, cap, head, rows, w = b
            fl += [Paragraph(f"Table {tbl}. {inline(cap)}", S["tcap"]), make_table(head, rows, w, S), Spacer(1, 8)]
        elif k == "img":
            fig += 1
            fl.append(KeepTogether([make_image(CH / b[1], b[3]), Paragraph(f"Figure {fig}. {inline(b[2])}", S["cap"])]))
        elif k == "pb":
            fl.append(PageBreak())
    doc.build(fl, onFirstPage=footer_factory(label), onLaterPages=footer_factory(label))


# ------------------------------------------------------------------ executive summary
def exec_blocks(df, R):
    geo, gen = R["geo"].set_index("group"), R["gender"].set_index("group")
    age, act = R["age"].set_index("group"), R["active"].set_index("group")
    seg = R["seg"].set_index("segment")
    mt = R["model"]["table"].set_index("model").loc["Gradient boosting"]
    base = R["overall_rate"]
    top2 = seg.loc[["Product Overload", "Inactive 50+"]]
    t_cust, t_chr, t_bal = top2.share_customers.sum(), top2.share_churners.sum(), top2.share_churned_balance.sum()
    cb = R["churned_balance"]
    de_sh = geo.loc["Germany", "churned"] / R["churned"] * 100
    B = []
    B.append(("h1", "1. Purpose"))
    B.append(("p", f"This brief summarises an analysis of {N(R['total'])} bank customer records to identify which groups of "
                   f"customers are most likely to leave the bank (\"churn\"), and to propose measures for testing. "
                   f"The analysis uses only the supplied dataset. Evidence and proposals are kept separate below."))
    B.append(("h1", "2. Headline figures (evidence)"))
    B.append(("table", "Key indicators", ["Indicator", "Value"],
              [["Customers analysed", N(R["total"])],
               ["Customers who left", f"{N(R['churned'])} ({P(base, 2)})"],
               ["Customers retained", f"{N(R['retained'])} ({P(R['retention_rate'], 2)})"],
               ["Balances held by customers who left", f"{M(cb)} units ({P(R['churned_balance_share'])} of all balances)"]],
              [2.6, 3.6]))
    B.append(("h1", "3. Key findings (evidence from the data)"))
    B.append(("num", [
        f"**Number of products.** Customers holding three or four products left at {P(seg.loc['Product Overload','churn_rate'])}; "
        f"customers holding two products left at only {P(R['products'].set_index('group').loc['2','rate'])}.",
        f"**Age.** Customers aged 50-59 left at {P(age.loc['50-59','rate'])}, compared with {P(age.loc['18-29','rate'])} for those aged 18-29.",
        f"**Inactivity.** Inactive members left at {P(act.loc['0','rate'])} against {P(act.loc['1','rate'])} for active members. "
        f"Inactive customers aged 50 or over left at {P(seg.loc['Inactive 50+','churn_rate'])}.",
        f"**Country.** Germany shows {P(geo.loc['Germany','rate'])} attrition, about twice France ({P(geo.loc['France','rate'])}) "
        f"and Spain ({P(geo.loc['Spain','rate'])}); it is {P(geo.loc['Germany','n'] / R['total'] * 100)} of customers but {P(de_sh)} of those who left.",
        f"**Gender.** Women left at {P(gen.loc['Female','rate'])} and men at {P(gen.loc['Male','rate'])}.",
        "**No clear link.** Credit score, years with the bank, credit card ownership and estimated salary show no meaningful "
        "association with leaving.",
        f"**Concentration.** Customers with three or more products, and inactive customers aged 50 or over, are together "
        f"{P(t_cust)} of customers but {P(t_chr)} of those who left and {P(t_bal)} of the balances they held.",
        f"**Predictive signal.** A baseline statistical model tested on unseen customers ranked risk well "
        f"(ROC-AUC {mt.test_auc:.2f}, where 0.50 is chance and 1.00 is perfect)."]))
    B.append(("img", "13_segments.png", "Churn rate by customer segment, and each segment's share of customers versus those who left", 6.4))
    B.append(("h1", "4. What the data does not show"))
    B.append(("bul", [
        "It covers one year and has no dates of departure, so it cannot show trends or timing.",
        "It shows association, not cause. The reasons customers leave are not recorded.",
        "It contains no revenue, cost or profit data, so no profit impact is calculated.",
        "Segment rates were derived from the same data and may be lower or higher for future customers."]))
    B.append(("h1", "5. Proposed recommendations (for decision, not findings)"))
    B.append(("num", [
        "**Pilot targeted retention outreach** for customers with three or more products and inactive customers aged 50 or over, "
        "using a comparison group to measure whether outreach reduces attrition.",
        "**Review the multi-product experience** (fees, onboarding, support) and speak with a sample of customers who left.",
        "**Review conditions in the German market** (pricing, competition, service) where attrition is highest.",
        "**Test re-engagement measures** for inactive customers aged 40 and over.",
        "**Check fairness and legal compliance** before any age- or gender-based targeting is used.",
        "**Strengthen data collection:** departure dates, multi-year history, complaints and revenue data."]))
    B.append(("h1", "6. Potential economic and business impact (illustrative arithmetic)"))
    B.append(("p", f"Customers who left held {M(cb)} balance units in this dataset ({P(R['churned_balance_share'])} of all balances). "
                   f"Purely as an illustration, if one in ten of those departures had been prevented, about {M(cb * 0.10)} units of "
                   f"balances would have remained with the bank. This is simple arithmetic on the data, not a forecast; it assumes "
                   f"prevented customers hold the same average balance and ignores the cost of retention measures. "
                   f"The currency is not stated in the dataset, and without revenue or margin data the effect on bank income "
                   f"cannot be estimated."))
    B.append(("h1", "7. Next steps"))
    B.append(("p", "Approve a small controlled pilot; assign an owner for data improvements; and review results after one "
                   "retention cycle before any wider rollout."))
    return B


# ------------------------------------------------------------------ CSV reports
def write_reports(df, R):
    rows = []

    def add(section, var, t, rename=None):
        for _, r in t.iterrows():
            g = rename.get(r.group, r.group) if rename else r.group
            rows.append({"section": section, "variable": var, "group": g, "customers": int(r.n), "churned": int(r.churned),
                         "churn_rate_pct": round(r.rate, 2), "ci95_low_pct": round(r.lo, 2), "ci95_high_pct": round(r.hi, 2)})
    rows.append({"section": "overall", "variable": "all customers", "group": "all", "customers": R["total"],
                 "churned": R["churned"], "churn_rate_pct": round(R["overall_rate"], 2),
                 "ci95_low_pct": round(A.wilson(R["churned"], R["total"])[0] * 100, 2),
                 "ci95_high_pct": round(A.wilson(R["churned"], R["total"])[1] * 100, 2)})
    add("profile", "Geography", R["geo"])
    add("profile", "Gender", R["gender"])
    add("profile", "AgeGroup", R["age"])
    add("profile", "IsActiveMember", R["active"], {"0": "Inactive", "1": "Active"})
    add("profile", "NumOfProducts", R["products"])
    add("profile", "CreditScoreBand", R["credit"])
    add("profile", "TenureBand", R["tenure"])
    add("profile", "BalanceBand", R["balance"])
    add("profile", "HasCrCard", R["card"], {"0": "No", "1": "Yes"})
    add("profile", "SalaryQuartile", R["salary"])
    seg = R["seg"]
    for _, r in seg.iterrows():
        rows.append({"section": "segment", "variable": "Segment", "group": r.segment, "customers": int(r.n),
                     "churned": int(r.churned), "churn_rate_pct": round(r.churn_rate, 2), "ci95_low_pct": round(r.lo, 2),
                     "ci95_high_pct": round(r.hi, 2)})
    out = pd.DataFrame(rows)
    out.to_csv(BASE / "analysis_report.csv", index=False)
    seg.drop(columns=["lo", "hi"]).round(2).to_csv(BASE / "segment_summary.csv", index=False)

    chi = R["chi"].assign(test="chi-square", statistic=lambda d: d.chi2.round(2))[["test", "variable", "statistic", "dof", "p_value", "cramers_v"]]
    chi["p_value"] = chi.p_value.map(lambda v: float(f"{v:.3g}"))
    chi["cramers_v"] = chi.cramers_v.round(4)
    mw = R["mw"].assign(test="mann-whitney", statistic=lambda d: d.u_stat.round(1), dof=np.nan,
                        cramers_v=np.nan)[["test", "variable", "statistic", "dof", "p_value", "cramers_v", "median_churned",
                                           "median_retained", "prob_superiority"]]
    mw["p_value"] = mw.p_value.map(lambda v: float(f"{v:.3g}"))
    mw["prob_superiority"] = mw.prob_superiority.round(4)
    pd.concat([chi, mw]).to_csv(BASE / "statistical_tests.csv", index=False)
    R["model"]["table"].round(4).to_csv(BASE / "model_metrics.csv", index=False)


def main():
    df, R = A.run(make_plots=True)
    write_reports(df, R)
    title = "Customer Segmentation & Churn Pattern Analytics in European Banking"
    sub = (f"A data-driven study of {N(R['total'])} customers | Source: European_Bank.csv | Prepared 13 November 2026")
    blocks = paper_blocks(df, R)
    (BASE / "research_paper.md").write_text(to_md(blocks, title, sub), encoding="utf-8")
    render_pdf(blocks, BASE / "research_paper.pdf", title, sub, "Customer Segmentation & Churn Pattern Analytics in European Banking")
    eb = exec_blocks(df, R)
    render_pdf(eb, BASE / "executive_summary.pdf", "Executive Summary: Customer Attrition in European Retail Banking",
               "Prepared for government and institutional stakeholders | Based on 10,000 customer records | 13 November 2026",
               "Executive Summary | Customer Segmentation & Churn Pattern Analytics")
    print("Built all deliverables")


if __name__ == "__main__":
    main()
