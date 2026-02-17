# Biological Basis of Oncology-GNN-Edge

## 1. Introduction

Cellular function is governed by interaction networks rather than isolated molecular events. Systems biology models these networks to understand disease processes.

Protein–protein interaction (PPI) networks provide a structural representation of cellular organization.

---

## 2. Protein–Protein Interaction Networks

PPI networks are constructed from:

* Yeast two-hybrid assays
* Co-immunoprecipitation
* Mass spectrometry
* Curated databases

Major databases include:

* STRING [1]
* BioGRID [2]
* Reactome [3]

These networks encode:

* Physical binding
* Functional associations
* Signaling pathways

---

## 3. Gene Expression as Molecular State

Gene expression profiling (RNA-seq, microarray) quantifies transcript abundance [4].

Log-transformation:

[
z = \log(x + 1)
]

is standard for stabilizing variance in RNA-seq data [4].

Standardization via per-gene z-score normalization enables comparative analysis.

---

## 4. Network-Level Dysregulation in Cancer

Cancer is characterized by:

* Pathway activation shifts
* Network rewiring
* Module-level perturbations

Network-based approaches outperform gene-level methods in capturing systemic dysregulation [5].

---

## 5. Why Graph Neural Networks?

Traditional differential expression treats genes independently.

Network models:

* Capture topological dependencies
* Incorporate neighborhood effects
* Reflect pathway-level organization

Graph-based modeling in biology has expanded significantly in recent years [6].

---

## 6. Biological Meaning of Embeddings

After graph convolution, each protein embedding reflects:

* Intrinsic molecular state
* Influence of interacting neighbors
* Local network topology

These embeddings act as structural molecular descriptors.

---

## 7. Pathway Aggregation

Proteins cluster into pathways such as:

* PI3K/AKT
* MAPK
* p53
* DNA repair pathways

Aggregating embeddings across pathway nodes allows structural pathway analysis.

---

## 8. Edge-Based Biological Computation

In decentralized or privacy-sensitive environments:

* Cloud-based genomic computation may be impractical
* Data transfer may be restricted
* Infrastructure may be limited

Edge-based execution enables:

* Local molecular modeling
* Reduced data movement
* Hardware-aware biological analysis

This systems-level approach is discussed in [7].

---

## 9. Research Scope

This framework:

* Does not diagnose disease
* Does not predict treatment outcomes
* Does not replace laboratory validation

It supports:

* Hypothesis generation
* Network-level exploratory modeling
* Computational biology research workflows

---

## References

[1] Szklarczyk, D., et al. (2021). STRING v11: protein–protein association networks. *Nucleic Acids Research*.

[2] Oughtred, R., et al. (2021). The BioGRID database. *Nucleic Acids Research*.

[3] Gillespie, M., et al. (2022). The Reactome pathway knowledgebase. *Nucleic Acids Research*.

[4] Love, M. I., Huber, W., & Anders, S. (2014). Moderated estimation of fold change and dispersion for RNA-seq data with DESeq2. *Genome Biology*.

[5] Barabási, A.-L., Gulbahce, N., & Loscalzo, J. (2011). Network medicine: a network-based approach to human disease. *Nature Reviews Genetics*.

[6] Zitnik, M., Agrawal, M., & Leskovec, J. (2018). Modeling polypharmacy side effects with graph convolutional networks. *Bioinformatics*.

[7] Vidya, S. (2026). *Edge-Based Execution of Graph Neural Networks for Protein Interaction Network Analysis in Clinical Oncology.* Research Square Preprint. [https://doi.org/10.21203/rs.3.rs-8645211/v1](https://doi.org/10.21203/rs.3.rs-8645211/v1)


