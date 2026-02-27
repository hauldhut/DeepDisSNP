## Environment Setup
- **For Python-based Code**
  - *conda env create -f DeepDisSNP.yml*: To run DeepDisSNP

- **For R-based Code**: To run DisSNPNet, Evidence Collection/Visualization
  - *Install R packages: ggplot2, igraph, biomaRt, clusterProfiler, MESHsim, gwasrapidd, PhenoScanner, ieugwasr, cowplot, hash*

## Experiments
- **Generate Embeddings**
  - *generate_embeddings_for_DiNet.py*: Generate embeddings for diseases from disease similarity.
  - *generate_embeddings_for_SNPNet_AllChrs.py*: Generate embeddings for SNPs from per-chromosome SNP LD networks.
 
- **Evaluate**:
  - *evaluate.py*: For various combinations of disease and enhancer embeddings, embedding sizes, and epochs

- **Predict**:
  - *predict.py*: For prediction of novel disease-enhancer associations

## Summary
  - *summarize.py*: To summarize and create heatmaps for various combinations of disease and enhancer embeddings, embedding sizes, and epochs

## Comparison
  - *DisSNPNet*: (https://github.com/hauldhut/DisSNPNet)
  


