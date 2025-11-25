import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import os
import json
from datetime import datetime

def load_and_prepare_data(file_path):
    """
    Load Excel/CSV and prepare training data.
    
    Args:
        file_path: Path to the Excel or CSV file
    
    Returns:
        DataFrame with all columns preserved
    """
    print(f"📂 Loading data from: {file_path}")
    
    # Check file extension and load accordingly
    if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}. Use .xlsx, .xls, or .csv")
    
    print(f"✅ Loaded {len(df)} rows")
    print(f"📋 Columns: {list(df.columns)}")
    
    # Create the searchable query text
    df['query'] = df['Pass/Fail Criteria'].astype(str)
    # df['query'] = df['Description of Criteria'].astype(str) + '\t\n\n' + df['Pass/Fail Criteria'].astype(str)
    
    # The method column is the string query{index} with 4 digits (e.g., query0001)
    df['method'] = df.index.to_series().apply(lambda x: f'query{x+1:04d}') #indexed by 1
    
    # Remove rows with missing query data
    df_clean = df[df['query'].notna()].copy()
    df_clean = df_clean[df_clean['query'] != 'nan\t\n\nnan'].copy()
    
    print(f"✅ Clean data: {len(df_clean)} rows")
    print(f"💡 Added 'method' column: query{{index}}")
    
    return df_clean


def create_training_examples(df, split_ratio=0.8):
    """
    Create training examples for sentence transformer.
    Uses the query text as both anchor and positive for contrastive learning.
    
    Args:
        df: DataFrame with 'query' column
        split_ratio: Ratio for train/validation split
    
    Returns:
        train_examples, val_df, train_df
    """
    # Shuffle data
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split into train and validation
    split_idx = int(len(df_shuffled) * split_ratio)
    train_df = df_shuffled[:split_idx]
    val_df = df_shuffled[split_idx:]
    
    print(f"\n📊 Data split:")
    print(f"   Training: {len(train_df)} samples")
    print(f"   Validation: {len(val_df)} samples")
    
    # Create InputExample objects for contrastive learning
    # Each query is paired with itself (anchor = positive)
    train_examples = []
    for _, row in train_df.iterrows():
        train_examples.append(InputExample(texts=[row['query'], row['query']]))
    
    return train_examples, val_df, train_df


def save_corpus(df, output_dir='models/wwan_retrieval'):
    """
    Save the corpus with all metadata for retrieval during inference.
    
    Each corpus entry contains:
    - id: unique identifier
    - text: searchable text (query)
    - metadata: all other columns (method, images, etc.)
    
    Args:
        df: DataFrame with all columns
        output_dir: Directory to save corpus files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Build corpus with metadata
    corpus_entries = []
    for idx, row in df.iterrows():
        entry = {
            'id': f'doc_{idx}',
            'text': row['query'],
            'metadata': {}
        }
        
        # Add all other columns as metadata
        for col in df.columns:
            if col != 'query':
                value = row[col]
                # Convert to native Python types for JSON serialization
                if pd.isna(value):
                    entry['metadata'][col] = None
                elif isinstance(value, (pd.Timestamp, datetime)):
                    entry['metadata'][col] = str(value)
                else:
                    entry['metadata'][col] = value
        
        corpus_entries.append(entry)
    
    # Save full corpus with metadata as JSON
    corpus_data = {
        'corpus': corpus_entries,
        'corpus_size': len(corpus_entries),
        'created_at': datetime.now().isoformat(),
        'columns': list(df.columns)
    }
    
    corpus_json_path = os.path.join(output_dir, 'corpus.json')
    with open(corpus_json_path, 'w', encoding='utf-8') as f:
        json.dump(corpus_data, f, ensure_ascii=False, indent=2)
    print(f"📄 Corpus JSON saved to: {corpus_json_path}")
    print(f"   Entries: {len(corpus_entries)}")
    print(f"   Columns: {list(df.columns)}")
    
    # Save as CSV for easy inspection
    corpus_csv_path = os.path.join(output_dir, 'corpus.csv')
    df.to_csv(corpus_csv_path, index=False, encoding='utf-8')
    print(f"📄 Corpus CSV saved to: {corpus_csv_path}")
    
    # Save just the searchable texts (for quick loading)
    texts_only = [entry['text'] for entry in corpus_entries]
    texts_path = os.path.join(output_dir, 'corpus_texts.json')
    with open(texts_path, 'w', encoding='utf-8') as f:
        json.dump({'texts': texts_only}, f, ensure_ascii=False, indent=2)
    print(f"📄 Corpus texts saved to: {texts_path}")
    
    # Save metadata
    metadata = {
        'corpus_size': len(corpus_entries),
        'columns': list(df.columns),
        'created_at': datetime.now().isoformat()
    }
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f"📄 Metadata saved to: {metadata_path}")
    
    # Print sample entry structure
    print(f"\n📋 Sample corpus entry structure:")
    if corpus_entries:
        sample = corpus_entries[0]
        print(f"   ID: {sample['id']}")
        print(f"   Text (first 100 chars): {sample['text'][:100]}...")
        print(f"   Metadata keys: {list(sample['metadata'].keys())}")
    
    return corpus_entries


def train_model(train_examples, output_dir='models/wwan_retrieval', 
                base_model='sentence-transformers/all-MiniLM-L6-v2',
                num_epochs=10, batch_size=16):
    """
    Train sentence transformer model using contrastive learning.
    
    Args:
        train_examples: List of InputExample objects
        output_dir: Directory to save the model
        base_model: Base model to fine-tune
        num_epochs: Number of training epochs
        batch_size: Batch size for training
    
    Returns:
        Trained model
    """
    # Ensure output directory exists with proper permissions
    os.makedirs(output_dir, exist_ok=True)
    # os.chmod(output_dir, 0o777)  # Set full permissions
    
    # Set environment variable to control HuggingFace checkpoint location
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    # os.chmod(checkpoint_dir, 0o777)
    # os.environ['TRANSFORMERS_CACHE'] = checkpoint_dir
    # os.environ['HF_HOME'] = checkpoint_dir
    
    print(f"\n🤖 Loading base model: {base_model}")
    model = SentenceTransformer(base_model)
    
    # Create DataLoader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    
    # Use MultipleNegativesRankingLoss for contrastive learning
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    print(f"\n🏋️ Starting training...")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Training samples: {len(train_examples)}")
    print(f"   Loss: MultipleNegativesRankingLoss (contrastive)")
    print(f"   Checkpoint directory: {checkpoint_dir}")
    
    # Import TrainingArguments to control where checkpoints are saved
    from transformers import TrainingArguments
    
    # Create training arguments with explicit checkpoint path
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,  # Store checkpoints here
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        logging_steps=10,
        save_strategy='epoch',
        save_total_limit=2,
        logging_dir=os.path.join(checkpoint_dir, 'logs'),
        report_to='none',  # Disable wandb and other reporting
    )
    
    # Train the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=int(len(train_dataloader) * 0.1),
        output_path=output_dir,
        show_progress_bar=True,
        save_best_model=True,
        use_amp=False,  # Disable automatic mixed precision to avoid potential issues
    )
    
    print(f"\n✅ Training complete!")
    print(f"💾 Model saved to: {output_dir}")
    
    return model


def main():
    # Set umask to ensure all created files/directories have write permissions
    os.umask(0o000)  # This allows full permissions (777) for new files/directories
    
    # Disable wandb logging
    os.environ['WANDB_DISABLED'] = 'true'
    
    # Configuration
    file_path = './output_extracted_data.xlsx'
    output_dir = 'models/wwan_retrieval'
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    print("="*80)
    print("WWAN RETRIEVAL MODEL TRAINING")
    print("="*80)
    
    # Load and prepare data (preserves all columns)
    df = load_and_prepare_data(file_path)
    
    # Display sample
    print(f"\n{'='*80}")
    print("SAMPLE DATA")
    print("="*80)
    print(f"Columns: {list(df.columns)}")
    if len(df) > 0:
        print(f"\nFirst entry:")
        print(f"  Query: {df.iloc[0]['query'][:200]}...")
        print(f"  Other columns: {[col for col in df.columns if col != 'query']}")
    
    # Create training examples
    train_examples, val_df, train_df = create_training_examples(df, split_ratio=1.0)
    
    # Save corpus with ALL metadata BEFORE training
    print(f"\n{'='*80}")
    print("SAVING CORPUS WITH METADATA")
    print("="*80)
    corpus_entries = save_corpus(df, output_dir=output_dir)
    print(f"✅ Corpus saved: {len(corpus_entries)} entries with full metadata")
    
    # Train model
    print(f"\n{'='*80}")
    print("TRAINING MODEL")
    print("="*80)
    model = train_model(
        train_examples,
        output_dir=output_dir,
        num_epochs=3,
        batch_size=256
    )
    
    print(f"\n{'='*80}")
    print("✅ TRAINING COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"\n📦 Output files in '{output_dir}':")
    print(f"   - Model files (sentence transformer)")
    print(f"   - corpus.json (full corpus with metadata)")
    print(f"   - corpus_texts.json (texts only for fast loading)")
    print(f"   - corpus.csv (human-readable)")
    print(f"   - metadata.json (training info)")


if __name__ == "__main__":
    main()