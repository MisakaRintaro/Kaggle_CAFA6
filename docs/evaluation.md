## Evaluation Metrics

### Overview

This competition evaluates protein function prediction using **Gene Ontology (GO)** annotations
across three subontologies:

- **Molecular Function (MF)**
- **Biological Process (BP)**
- **Cellular Component (CC)**

Predictions are evaluated only on proteins that **accumulate newly experimentally validated GO
annotations after the submission deadline**, following the CAFA-style prospective evaluation
protocol.

Separate test sets are constructed for MF, BP, and CC.
A single protein may appear in multiple test sets if it acquires new annotations in multiple
subontologies.

---

### Information Accretion (ia)

Because GO is a **hierarchical directed acyclic graph (DAG)**, predicting a more specific (deeper)
GO term inherently implies its ancestor terms.
Therefore, this competition uses **information-accretion–weighted precision and recall** instead
of unweighted metrics.

For a GO term `f`, the **information accretion** is defined as:
ia(f) = -log P(f | parents(f))


Intuitively, this measures **how much additional information is gained by predicting term `f`,
given that its parent terms are already known**.

In practice, this can be expressed using annotation frequencies:

ia(f) = -log ( count(f) / count(parents(f)) )


- Root GO terms have `ia(f) = 0`
- Deeper, rarer, and more specific GO terms have larger `ia(f)`

⚠️ **Important:**  
The `ia(f)` values are **provided by the competition organizers** and must not be recomputed by
participants.

---

### Weighted Precision and Recall

Using information accretion as weights, precision and recall are defined as:

- **Weighted Precision**: fraction of predicted information that is correct
- **Weighted Recall**: fraction of true information that is recovered

Predictions are evaluated by sweeping a probability threshold, and the **maximum F1-score (Fmax)**
is computed for each subontology.

---

### Final Score

1. Compute **Fmax** separately for:
   - Molecular Function (MF)
   - Biological Process (BP)
   - Cellular Component (CC)

2. The final score is the **arithmetic mean** of the three Fmax values:
Final Score = (F_MF + F_BP + F_CC) / 3


Additionally, evaluation includes three knowledge settings:

- **No-knowledge**
- **Limited-knowledge**
- **Partial-knowledge**

Scores from these settings are also averaged as specified in the CAFA evaluation protocol.

---

### Practical Implications for Modeling

- Models should output **confidence scores (probabilities)** for each GO term.
- Correct prediction of **deep, specific GO terms** is rewarded more than shallow terms.
- Predicting only high-level GO terms yields low evaluation scores.
- Threshold selection is handled implicitly by the evaluator via Fmax computation.

---

### Reference

The evaluation protocol follows:

Jiang Y. et al.  
*An expanded evaluation of protein function prediction methods shows an improvement in accuracy.*  
Genome Biology (2016), 17:184

