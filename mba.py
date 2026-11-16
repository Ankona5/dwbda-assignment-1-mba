"""Assignment 1 - Market Basket Analysis (Apriori) on real-world grocery data.

Data: data/groceries.csv - 9,835 real supermarket transactions (one basket per line,
items separated by commas). Source: the "Groceries" dataset from the R package arules
(Hahsler et al.), a standard teaching dataset for association rules.

Steps: load baskets -> one-hot encode -> Apriori frequent itemsets -> association rules
(support, confidence, lift) -> save results and charts.
"""
import os
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

MIN_SUPPORT = 0.01      # itemset must appear in at least 1% of baskets
MIN_CONFIDENCE = 0.3    # rule must hold in at least 30% of baskets containing the antecedent

# 1. Load transactions
with open("data/groceries.csv") as f:
    baskets = [line.strip().split(",") for line in f if line.strip()]
print(f"Transactions: {len(baskets)}")

# 2. One-hot encode
te = TransactionEncoder()
df = pd.DataFrame(te.fit(baskets).transform(baskets), columns=te.columns_)
print(f"Distinct items: {df.shape[1]}")

# 3. Frequent itemsets (Apriori)
itemsets = apriori(df, min_support=MIN_SUPPORT, use_colnames=True)
itemsets["size"] = itemsets["itemsets"].apply(len)
print(f"Frequent itemsets (support >= {MIN_SUPPORT}): {len(itemsets)}")

# 4. Association rules
rules = association_rules(itemsets, metric="confidence", min_threshold=MIN_CONFIDENCE)
rules = rules.sort_values("lift", ascending=False)
for col in ("antecedents", "consequents"):
    rules[col] = rules[col].apply(lambda s: ", ".join(sorted(s)))
cols = ["antecedents", "consequents", "support", "confidence", "lift"]
print(f"\nRules (confidence >= {MIN_CONFIDENCE}): {len(rules)}\nTop 10 by lift:")
print(rules[cols].head(10).round(3).to_string(index=False))

# 5. Save results
itemsets.sort_values("support", ascending=False).to_csv("frequent_itemsets.csv", index=False)
rules[cols].to_csv("association_rules.csv", index=False)

# 6. Charts
os.makedirs("images", exist_ok=True)
top = df.mean().sort_values(ascending=False).head(15)[::-1]
plt.figure(figsize=(8, 5))
top.plot.barh(color="#2a7f62")
plt.xlabel("Support (fraction of baskets)")
plt.title("15 most frequently bought items")
plt.tight_layout()
plt.savefig("images/top_items.png", dpi=120)

plt.figure(figsize=(6, 5))
plt.scatter(rules["support"], rules["confidence"], c=rules["lift"], cmap="viridis")
plt.colorbar(label="lift")
plt.xlabel("support")
plt.ylabel("confidence")
plt.title("Association rules")
plt.tight_layout()
plt.savefig("images/rules_scatter.png", dpi=120)
