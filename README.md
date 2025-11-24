# K-Means Clustering From Scratch

## 📌 Overview

This project demonstrates a complete implementation of the **K-Means Clustering algorithm from scratch**, using only **NumPy** for core computations. The goal is to deeply understand:

* How unsupervised clustering works
* Iterative optimization steps
* Sensitivity to initialization
* Convergence and evaluation metrics

The repository includes a full experimental workflow — dataset creation, clustering, evaluation, and visualization.

---

## 🎯 Project Objectives

✔ Implement K-Means algorithm manually (no scikit-learn for core logic)
✔ Compare **Random** vs **K-Means++ initialization**
✔ Evaluate results using **Inertia** and **Silhouette Score**
✔ Visualize cluster assignments, centroids, elbow plot, silhouette analysis
✔ Study initialization sensitivity and limitations of K-Means

---

## 🧠 Key Concepts Covered

| Concept                    | What You Learn                     |
| -------------------------- | ---------------------------------- |
| Unsupervised Learning      | Grouping unlabeled data            |
| Distance-based Clustering  | Euclidean space optimization       |
| Iterative Refinement       | Expectation-Maximization behavior  |
| Initialization Sensitivity | Random seeds impact results        |
| Evaluation Metrics         | Silhouette Score & Inertia         |
| Data Visualization         | Cluster plots & performance graphs |

---

## 📂 Project Structure

```
📁 KMeans_From_Scratch
│
├── kmeans_from_scratch_full.py   # Main implementation + demo
├── README.md                      # (this file)
└── results/                       # Generated plots & figures
```

---

## 🛠️ Installation & Requirements

Ensure Python 3.8+ and install dependencies:

```bash
pip install numpy matplotlib
```

---

## 🚀 How to Run

Run the full script:

```bash
python kmeans_from_scratch_full.py
```

This will:

* Generate synthetic datasets
* Apply K-Means clustering for multiple `k` values
* Plot results and print evaluation metrics

---

## 📊 Outputs & Visualization

You will see:

* Cluster scatter plots
* Centroid positions
* Elbow Curve (Inertia vs K)
* Silhouette Scores
* Comparison across seeds for initialization sensitivity

Example result visualizations:

> 🟢 Well-separated dataset
> 🟡 Overlapping dataset

---

## 📈 Evaluation Metrics Used

| Metric               | Purpose                              |
| -------------------- | ------------------------------------ |
| **Inertia (WCSS)**   | Measures compactness of clusters     |
| **Silhouette Score** | Measures separation between clusters |
| **Iterations Count** | Convergence performance              |

These metrics help decide the **best K** value.

---

## 🧩 Strengths of K-Means

* Simple & easy to implement
* Fast and scalable
* Works well with spherical clusters and large datasets

---

## ⚠️ Limitations Observed

| Limitation                  | Effect                                          |
| --------------------------- | ----------------------------------------------- |
| Sensitive to initialization | Different seeds → different results             |
| Assumes spherical clusters  | Poor performance on elongated or complex shapes |
| Must choose K manually      | Requires elbow/silhouette analysis              |
| Affected by outliers        | Centroids shift incorrectly                     |

This project includes experiments proving these limitations.

---

## 🔍 Future Improvements

* Handle outliers using **K-Medoids / DBSCAN**
* Support categorical features
* GPU-accelerated version
* Auto-estimation of **K** using Gap Statistic

---

## 🏁 Conclusion

This project builds a strong foundational understanding of:
✔ Optimization behavior of K-Means
✔ Impact of parameter tuning
✔ How to validate clustering results properly

You now have a fully working reference implementation useful for:

* Machine learning coursework
* Portfolio showcase
* Research learning

---

## 👤 Author

Manibharathi Nagarajan — **UX/UI Designer + AI Learner**

If you want, I can help you convert this into a GitHub repository with commits, code formatting, and portfolio-ready documentation.
