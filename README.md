# Customer Segmentation & Churn Pattern Analytics in European Banking

Analysis of 10,000 bank customers (`European_Bank.csv`): who churns, which segments carry the most risk,
and what to test next. Every number in the paper, summary and dashboard is calculated from the CSV.
`CustomerId`, `Surname` and `Year` are never used as analytical features.

## Folder contents

| Path | What it is |
|---|---|
| `app.py` | Streamlit dashboard (filters: country, gender, age, activity, products) |
| `research_paper.pdf` / `research_paper.md` | Full research paper with embedded charts and tables |
| `executive_summary.pdf` | 2-page brief for non-technical / government readers |
| `analysis_report.csv` | Churn rate, counts and 95% CI for every group and segment |
| `statistical_tests.csv` | Chi-square (with Cramer's V) and Mann-Whitney results |
| `segment_summary.csv`, `model_metrics.csv` | Segment profiles; baseline model scores |
| `charts/` | The 14 PNG charts used in the paper |
| `scripts/` | `analysis_core.py` (analysis) and `build_project.py` (rebuilds everything) |
| `European_Bank.csv` | Input data |
| `requirements.txt` | Python dependencies |

## Run the dashboard

```bash
pip install -r requirements.txt
streamlit run app.py
```

`app.py` reads `European_Bank.csv` from its own folder; if the file is missing it shows an upload box.

## Rebuild all outputs of project

```bash
python scripts/build_project.py
```

This regenerates the charts, CSV reports, `research_paper.md`, `research_paper.pdf` and `executive_summary.pdf`.
The random seed is fixed (42), so results are reproducible.

## Headline results

* Overall churn: **20.37%** (2,037 of 10,000 customers).
* Strongest associations: number of products, age, country (Germany), inactivity, gender.
* No meaningful association: credit score, tenure, credit card ownership, estimated salary.
* Six rule-based segments; the two highest-risk (Product Overload, Inactive 50+) are 7.6% of customers
  but 30.8% of churners.
* Baseline gradient boosting model: ROC-AUC 0.870 on a held-out test set (see paper, Section 8, for limits).

## Method notes and limits

Findings are associations from a single-year snapshot, not proven causes. Segment rules were derived from
the same data, so segment rates are descriptive. The currency and the unit of tenure are not stated in the file.
Add your name, institution and any required citations to the paper before submission.
