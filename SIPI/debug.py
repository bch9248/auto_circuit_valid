import os
def save_training_samples(train_df, output_path='output/training_samples.txt'):
    """
    Save training samples to show contrastive learning pairs.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("CONTRASTIVE LEARNING - TRAINING SAMPLES\n")
        f.write("=" * 100 + "\n\n")
        f.write("CONCEPT:\n")
        f.write("- Anchor = Positive (same text)\n")
        f.write("- This teaches the model to create consistent embeddings for the same text\n")
        f.write("- Negatives = other texts in the batch (auto-generated)\n")
        f.write("=" * 100 + "\n\n")
        
        for idx, row in train_df.head(10).iterrows():  # Show first 10 samples
            f.write(f"{'='*100}\n")
            f.write(f"SAMPLE {idx + 1}\n")
            f.write(f"{'='*100}\n\n")
            
            f.write("🟦 ANCHOR (and POSITIVE - same text):\n")
            f.write(f"{'─'*100}\n")
            f.write(f"{row['text']}\n\n")
            
            # Show example negatives
            if idx < len(train_df) - 1:
                f.write(f"{'─'*100}\n")
                f.write("🟥 EXAMPLE NEGATIVES (other texts from batch):\n\n")
                for neg_idx in range(idx + 1, min(idx + 4, len(train_df))):
                    f.write(f"Negative {neg_idx - idx}:\n")
                    f.write(f"{train_df.iloc[neg_idx]['text'][:300]}...\n\n")
            
            f.write("\n")
    
    print(f"💾 Training samples saved to: {output_path}")