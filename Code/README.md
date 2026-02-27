## Environment Setup
- **For Python-based Code**
  - *conda env create -f DeepDisSNP.yml*: To run DeepDisSNP

- **For R-based Code**: To run Evidence Collection/Visualization and Pathway Enrichment Analysis
  - *Install R packages: ggplot2, igraph, biomaRt, clusterProfiler, MESHsim, gwasrapidd, PhenoScanner, ieugwasr, cowplot, hash*

## Experiments
- **Generate Embeddings**
  - *generate_embeddings_for_DiNet.py*: Generate embeddings for diseases from disease similarity.
  - *generate_embeddings_for_SNPNet_AllChrs.py*: Generate embeddings for SNPs from per-chromosome SNP LD networks.
 
- **Evaluate**:
  - *evaluate.py*: For various combinations of 1KGP Datasets (Phase 1 and 3), LD thresholds, and per-chromosome LD networks (chromosome 1-22)

- **Predict**:
  - *predict.py*: For prediction of novel disease-SNP associations

## Comparison
  - *DisSNPNet*: (https://github.com/hauldhut/DisSNPNet)
  


