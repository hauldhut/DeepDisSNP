import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm
import itertools
import os
import sys
import random
import heapq

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

cls_method = "MLP"

# Custom class to redirect print output to both console and file
class Tee:
    def __init__(self, filename):
        self.file = open(filename, 'w')
        self.stdout = sys.stdout

    def write(self, message):
        self.file.write(message)
        self.stdout.write(message)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def close(self):
        self.file.close()

# Step 1: Load embeddings and disease-SNP pairs, filter invalid pairs
def load_data(disease_embeddings_file, SNP_embeddings_file, disease_SNP_file):
    print("Loading embeddings...")
    disease_embeddings_df = pd.read_csv(disease_embeddings_file)
    SNP_embeddings_df = pd.read_csv(SNP_embeddings_file)

    valid_diseases = set(disease_embeddings_df[disease_embeddings_df['type'] == 'disease']['node_id'])
    valid_SNPs = set(SNP_embeddings_df[SNP_embeddings_df['type'] == 'SNP']['node_id'])

    disease_emb = disease_embeddings_df[disease_embeddings_df['type'] == 'disease'].set_index('node_id')
    SNP_emb = SNP_embeddings_df[SNP_embeddings_df['type'] == 'SNP'].set_index('node_id')

    disease_emb_cols = [col for col in disease_embeddings_df.columns if col.startswith('dim_')]
    embedding_size_disease = len(disease_emb_cols)
    SNP_emb_cols = [col for col in SNP_embeddings_df.columns if col.startswith('dim_')]
    embedding_size_SNP = len(SNP_emb_cols)

    disease_emb = disease_emb[disease_emb_cols].to_numpy()
    SNP_emb = SNP_emb[SNP_emb_cols].to_numpy()

    diseases = sorted(valid_diseases)
    SNPs = sorted(valid_SNPs)

    print(f"Number of diseases with embeddings: {len(diseases)}")
    print(f"Number of SNPs with embeddings: {len(SNPs)}")

    print("Loading positive disease-SNP pairs...")
    disease_SNP_df = pd.read_csv(disease_SNP_file)

    positive_pairs = []
    total_pairs = len(disease_SNP_df)
    for _, row in disease_SNP_df.iterrows():
        disease, SNP = row['disease'], row['SNP']
        if disease in valid_diseases and SNP in valid_SNPs:
            positive_pairs.append((disease, SNP))

    positive_pairs = set(positive_pairs)
    skipped_pairs = total_pairs - len(positive_pairs)
    print(f"Number of positive pairs loaded: {len(positive_pairs)}")
    print(f"Number of pairs skipped (missing embeddings): {skipped_pairs}")

    if not positive_pairs:
        raise ValueError("No valid positive pairs found after filtering. Check embeddings_file and disease_SNP_file.")

    return diseases, SNPs, disease_emb, SNP_emb, positive_pairs, embedding_size_disease, embedding_size_SNP

# Step 2: Generate feature vectors and labels with balanced negative sampling
def generate_features_labels(
    diseases,
    SNPs,
    disease_emb,
    SNP_emb,
    positive_pairs,
    embedding_size_disease,
    embedding_size_SNP,
    selNeg_file=None
):
    print("Generating feature vectors and labels with balanced negative sampling...")

    disease_idx = {disease: i for i, disease in enumerate(diseases)}
    SNP_idx = {SNP: i for i, SNP in enumerate(SNPs)}

    positive_pairs_list = list(positive_pairs)
    num_positive = len(positive_pairs_list)
    print(f"Number of positive pairs: {num_positive}")

    # ---------- NEGATIVE PAIRS ----------
    if selNeg_file is not None and os.path.exists(selNeg_file):
        print(f"Loading selected negative pairs from: {selNeg_file}")
        neg_df = pd.read_csv(selNeg_file, sep="\t", header=None, names=["disease", "SNP"])
        selected_negative_pairs = list(zip(neg_df["disease"], neg_df["SNP"]))
    else:
        print("Sampling negative pairs...")
        all_pairs = itertools.product(diseases, SNPs)
        negative_pairs = [p for p in all_pairs if p not in positive_pairs]

        selected_negative_pairs = random.sample(negative_pairs, num_positive)

        if selNeg_file is not None:
            print(f"Saving selected negative pairs to: {selNeg_file}")
            pd.DataFrame(selected_negative_pairs).to_csv(
                selNeg_file, sep="\t", index=False, header=False
            )

    print(f"Number of negative pairs used: {len(selected_negative_pairs)}")

    # ---------- BUILD FEATURES ----------
    features, labels, pairs = [], [], []

    selected_pairs = positive_pairs_list + selected_negative_pairs
    random.shuffle(selected_pairs)

    for disease, SNP in tqdm(selected_pairs, desc="Generating features for pairs"):
        pairs.append((disease, SNP))
        label = 1 if (disease, SNP) in positive_pairs else 0
        labels.append(label)

        disease_vec = disease_emb[disease_idx[disease]]
        SNP_vec = SNP_emb[SNP_idx[SNP]]
        features.append(np.concatenate([disease_vec, SNP_vec]))

    return np.array(features), np.array(labels), pairs

# def logit(p, eps=1e-8):
#     p = np.clip(p, eps, 1 - eps)
#     return np.log(p / (1 - p))

# Step 3: Train MLP model and predict novel associations (memory safe)
def train_and_predict(diseases, SNPs, disease_emb, SNP_emb, positive_pairs, embedding_size_disease, embedding_size_SNP,
                      disease_embeddings_file, SNP_embeddings_file, base_name, topK=100, batch_size=500_000):

    print("Training MLP model on all labeled data...")
    mlp = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            solver='adam',
            learning_rate_init=1e-3,
            alpha=1e-4,   # regularization
            max_iter=500,
            early_stopping=False,
            random_state=42
        ))
    ])

    selNeg_file = f'../Results/Detail/{base_name}_selNeg_MatchedEE.txt'
    # Generate training data
    features, labels, pairs = generate_features_labels(diseases, SNPs, disease_emb, SNP_emb,
                                                positive_pairs, embedding_size_disease, embedding_size_SNP, selNeg_file)

    # Train model
    mlp.fit(features, labels)

    # Evaluate model on training data
    y_pred_proba = mlp.predict_proba(features)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    auroc = roc_auc_score(labels, y_pred_proba)
    auprc = average_precision_score(labels, y_pred_proba)
    f1 = f1_score(labels, y_pred)
    accuracy = accuracy_score(labels, y_pred)

    # if auroc > 0.999:
    #     print("Warning: AUROC=1.0000 or very high, indicating potential overfitting or data leakage.")

    print("\nTraining Data Performance:")
    print(f"AUROC: {auroc:.4f}")
    print(f"AUPRC: {auprc:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    metrics_df = pd.DataFrame([{
        'disease_emb_file': disease_embeddings_file,
        'SNP_emb_file': SNP_embeddings_file,
        'auroc_mean': auroc,
        'auroc_std': 0.0,
        'auprc_mean': auprc,
        'auprc_std': 0.0,
        'f1_mean': f1,
        'f1_std': 0.0,
        'accuracy_mean': accuracy,
        'accuracy_std': 0.0
    }])
    metrics_csv = f'../Prediction/{base_name}_top_{topK}_model_metrics.csv'
    metrics_df.to_csv(metrics_csv, index=False, float_format="%.4f")
    print(f"Saved model metrics to {metrics_csv}")

    # --- Predict for ALL novel pairs in batches ---
    print("Generating predictions for novel disease-SNP associations (batched)...")
    disease_idx = {disease: i for i, disease in enumerate(diseases)}
    SNP_idx = {SNP: i for i, SNP in enumerate(SNPs)}

    all_pairs_iter = itertools.product(diseases, SNPs)
    novel_pairs_iter = (pair for pair in all_pairs_iter if pair not in positive_pairs)

    top_k_heap = []  # min-heap for top topK
    batch_pairs = []
    batch_features = []

    total_novel = len(diseases) * len(SNPs) - len(positive_pairs)
    for pair in tqdm(novel_pairs_iter, total=total_novel, desc="Processing batches"):
        disease, SNP = pair
        disease_vec = disease_emb[disease_idx[disease]]
        SNP_vec = SNP_emb[SNP_idx[SNP]]
        batch_features.append(np.concatenate([disease_vec, SNP_vec]))
        batch_pairs.append(pair)

        if len(batch_features) >= batch_size:
            probs = mlp.predict_proba(np.array(batch_features))[:, 1]
            # probs = logit(probs)

            for p, prob in zip(batch_pairs, probs):
                if len(top_k_heap) < topK:
                    heapq.heappush(top_k_heap, (prob, p))
                else:
                    heapq.heappushpop(top_k_heap, (prob, p))
            batch_pairs.clear()
            batch_features.clear()

    # Process last batch
    if batch_features:
        probs = mlp.predict_proba(np.array(batch_features))[:, 1]
        # probs = logit(probs)

        for p, prob in zip(batch_pairs, probs):
            if len(top_k_heap) < topK:
                heapq.heappush(top_k_heap, (prob, p))
            else:
                heapq.heappushpop(top_k_heap, (prob, p))

    # Sort results
    top_k_heap.sort(reverse=True, key=lambda x: x[0])
    top_k_probs, top_k_pairs = zip(*top_k_heap)

    predictions_df = pd.DataFrame({
        'disease': [pair[0] for pair in top_k_pairs],
        'SNP': [pair[1] for pair in top_k_pairs],
        'predicted_probability': top_k_probs
    })
    predictions_csv = f'../Prediction/{base_name}_top_{topK}_predictions.csv'
    predictions_df.to_csv(predictions_csv, index=False, float_format="%.4f")
    print(f"Saved top {topK} predictions to {predictions_csv}")

    return auroc, 0.0, auprc, 0.0, f1, 0.0, accuracy, 0.0

def main():
    emb_method = "gat" #gat|gt|gcn|dw|n2v (only for H=4, o=0.5)|SapBERT (only for H=4, o=0.5)|Word2Vec (only for H=4, o=0.5)|MeSHHeading2vec
    print(f"emb_method: {emb_method}")

    DiSimNet = "MeSHID_Net" #MeSHID_Net|MeSH (only set this to SapBERT or Word2Vec or MeSHHeading2vec)
    
    epochs = 100
    embedding_size = 128
    
    phases = [1]
    ld_thresholds = ["r2def"]
    chromosomes = list(range(1, 23))
    
    
    H=4 #2|8
    o=0.5 #0|0.1|0.2|0.3|0.4

    topK = 100000
    
    for phase in phases:
        for ld in ld_thresholds:    
            for chr in chromosomes:
                print(f"phase: {phase}, ld_thres: {ld}, chr: {chr}")

                SNP_net = f"1kg_phase{phase}_chr{chr}_{ld}.ld_Net"
                print(f"SNP_net: {SNP_net}")

            
                if (H==4) & (o==0.5):
                    disease_emb_file = f"{DiSimNet}_{emb_method}_d_{embedding_size}_e_{epochs}"
                    SNP_emb_file = f"{SNP_net}_{emb_method}_d_{embedding_size}_e_{epochs}"
                    if (emb_method == "SapBERT") | (emb_method == "Word2Vec"):
                        disease_emb_file = f"{DiSimNet}_{emb_method}_d_{embedding_size}"
                        SNP_emb_file = f"{SNP_net}_gat_d_{embedding_size}_e_{epochs}"
                    if emb_method == "MeSHHeading2vec":
                        disease_emb_file = f"{DiSimNet}_{emb_method}_d_64"
                        SNP_emb_file = f"{SNP_net}_gat_d_{embedding_size}_e_{epochs}"
                else:
                    if o==0.5:
                        disease_emb_file = f"{DiSimNet}_{emb_method}_d_{embedding_size}_e_{epochs}_H_{H}"
                        SNP_emb_file = f"{SNP_net}_{emb_method}_d_{embedding_size}_e_{epochs}_H_{H}"
                    else:
                        disease_emb_file = f"{DiSimNet}_{emb_method}_d_{embedding_size}_e_{epochs}_H_{H}_o_{o}"
                        SNP_emb_file = f"{SNP_net}_{emb_method}_d_{embedding_size}_e_{epochs}_H_{H}_o_{o}"


                disease_embeddings_file = f"../Results/Embeddings/{disease_emb_file}.csv"
                SNP_embeddings_file = f"../Results/Embeddings/{SNP_emb_file}.csv"

                if DiSimNet == f"{DiSimNet}":
                    disease_SNP_file = os.path.expanduser(f"~/Data/GWAS/CAUSALdb/Chr_{chr}_Assoc.txt_BinaryInteraction.csv")
                else:
                    disease_SNP_file = os.path.expanduser(f"../Data/SNP2Disease_HMDD_MeSHUI_final.csv")
                
                base_name_disease = os.path.splitext(os.path.basename(disease_embeddings_file))[0]
                base_name_SNP = os.path.splitext(os.path.basename(SNP_embeddings_file))[0]
                
                base_name = base_name_disease + "_" + base_name_SNP + f"_Balanced_{cls_method}"

                print(f"\nProcessing pair:")
                print(f"disease_embeddings_file: {disease_embeddings_file}")
                print(f"SNP_embeddings_file: {SNP_embeddings_file}")
                print(f"disease-SNP file: {disease_SNP_file}")
                
                output_file = f'../Prediction/{base_name}_top_{topK}_output.txt'

                tee = Tee(output_file)
                sys.stdout = tee
                
                try:
                    diseases, SNPs, disease_emb, SNP_emb, positive_pairs, disease_emb_size, SNP_emb_size = load_data(
                        disease_embeddings_file, SNP_embeddings_file, disease_SNP_file
                    )

                    train_and_predict(
                        diseases, SNPs, disease_emb, SNP_emb, positive_pairs, disease_emb_size, SNP_emb_size,
                        disease_embeddings_file, SNP_embeddings_file, base_name, topK, batch_size=500_000
                    )
                
                except Exception as e:
                    print(f"Error processing: {str(e)}")
                
                finally:
                    sys.stdout = tee.stdout
                    tee.close()
    
# Execute
if __name__ == "__main__":
    main()

