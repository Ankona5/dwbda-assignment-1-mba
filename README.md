# Assignment 1 – Market Basket Analysis (Apriori) on real-world data

**Name:** Ankona Mukherjee
**Registration No.:** 202400560

## Task
Apply Market Basket Analysis (MBA) on real-world data using Python.

## Data
`data/groceries.csv`: 9,835 real supermarket transactions (one basket per line) with 171 distinct
products. It is the *Groceries* dataset from the R package `arules` (Hahsler et al.), a standard
real-world dataset for association rule mining.

## Method
1. Load the baskets and one-hot encode them (`TransactionEncoder`).
2. Find frequent itemsets with **Apriori** (minimum support 1%).
3. Generate **association rules** (minimum confidence 30%) and rank them by lift.
   - *support*: share of baskets containing the items together
   - *confidence*: how often the consequent is bought when the antecedent is
   - *lift*: how much more often they are bought together than by chance (> 1 means a positive association)

## Run
```bash
pip install pandas mlxtend matplotlib
python mba.py
```
It prints the results and saves the charts in `images/` (plus the rules as CSV files).

## Results
9,835 transactions, 332 frequent itemsets, 125 rules. Top rules by lift:

| If a basket has | then it also has | support | confidence | lift |
|---|---|---|---|---|
| citrus fruit, other vegetables | root vegetables | 0.010 | 0.359 | 3.30 |
| other vegetables, tropical fruit | root vegetables | 0.012 | 0.343 | 3.15 |
| beef | root vegetables | 0.017 | 0.331 | 3.04 |
| citrus fruit, root vegetables | other vegetables | 0.010 | 0.586 | 3.03 |
| root vegetables, tropical fruit | other vegetables | 0.012 | 0.585 | 3.02 |
| curd, whole milk | yogurt | 0.010 | 0.385 | 2.76 |

**Takeaway:** fresh produce (root vegetables, other vegetables, citrus and tropical fruit) is bought
together far more often than by chance, and dairy items (curd, whole milk, yogurt) form another
cluster. A store could place these items near each other or bundle them in promotions.

### Charts
![Most frequently bought items](images/top_items.png)

![Association rules: support vs confidence, coloured by lift](images/rules_scatter.png)
