# NeuroSAM: Hybrid Neuro-Symbolic Sentiment Analysis Model

This repository contains the implementation of NeuroSAM, a hybrid neuro-symbolic model for sentiment analysis that combines UFEN-MTFN, ECO-SAM, and Neuro Symbolic Transformer (NST) architecture with symbolic topic distributions from Latent Dirichlet Allocation (LDA). This README provides instructions for installing and running the software, as well as replicating the experiments described in the codebase.

## Overview

NeuroSAM integrates neural text representations from ALBERT with symbolic features derived from LDA topic modeling at multiple granularities (15 and 25 topics). The model employs advanced components like the MultiGranularityTopicModule, UFENModule, and ECOSAMModule for feature extraction and fusion, achieving robust sentiment classification across three classes: negative, neutral, and positive.

## Prerequisites

Before installing and running the software, ensure you have the following:

- Python 3.9
- A compatible GPU with CUDA support (recommended for training and inference)
- Access to a Hugging Face account with a valid token for model download/upload (optional for hosting or downloading pre-trained models)

## Installation

Follow these steps to set up the environment and install the necessary dependencies.

1. **Clone the Repository**  
   Clone this repository to your local machine or download the provided code files.

   ```bash
   git clone 
   cd 
   ```

2. **Create a Virtual Environment** (optional but recommended)  
   Set up a virtual environment to manage dependencies.

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**  
   Install the required Python packages. The codebase relies on libraries like `torch`, `transformers`, `pandas`, `numpy`, `sklearn`, and others.

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install transformers pandas numpy scikit-learn nltk nlpaug imbalanced-learn matplotlib seaborn tqdm huggingface_hub python-dotenv
   ```

4. **Download NLTK Resources**  

   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('vader_lexicon'); nltk.download('opinion_lexicon'); nltk.download('averaged_perceptron_tagger')"
   ```

5. **Set Up Hugging Face Token** (optional)  

   ```bash
   echo "HF_READ=" > .env
   ```

## Running the Software

### 1. Data Preprocessing

The preprocessing script (`preprocessing.ipynb`) prepares raw text data for training by cleaning text, extracting LDA features, balancing classes, and tokenizing inputs.

- **Prepare Raw Data**: Place your raw datasets in a directory named `raw_data`. The script expects datasets like Twitter sentiment data, YouTube comments, and climate text data with specific column names. Adjust file paths and column mappings in the script if your data differs.
- **Run Preprocessing**: Execute the preprocessing script to generate cleaned, tokenized, and balanced datasets with LDA features.

  ```bash
  python preprocessing.py
  ```

- **Output**: This generates CSV files (`train_data_balanced.csv`, `val_data.csv`, `test_data.csv`), tokenized NumPy arrays (e.g., `train_input_ids.npy`), LDA features, and pickled objects like `lda_model.pkl` and `sentiment_encoder.pkl`.

### 2. Model Training

The training script (`model train.ipynb`) implements the NeuroSAM model, loads preprocessed data, trains the model with mixed precision and early stopping, and evaluates performance on test data.

- **Ensure Preprocessed Data**: Verify that the preprocessing outputs are available in the specified `BASE_PATH` (default: `/kaggle/input/optimized-set`). Update the path in the script if necessary.
- **Run Training**: Execute the training script to train the model over multiple epochs.

  ```bash
  python train.py
  ```

- **Output**: The script saves the best model checkpoint as `hybrid_sentiment_model.pt`, plots training metrics (`training_metrics.png`), and confusion matrices (`confusion_matrix.png`). It also uploads the model to the Hugging Face Hub under `aiguy68/neurosam-model`.

### 3. Inference (Demo)

The demo script (`demo.py`) demonstrates how to use the pre-trained NeuroSAM model for sentiment prediction on new text.

- **Set Up Environment**: Ensure the `.env` file with your Hugging Face token is configured if downloading the model from the Hub.
- **Run Demo**: Execute the demo script with a sample text input.

  ```bash
  python demo.py
  ```

- **Output**: The script outputs the predicted sentiment (negative, neutral, or positive) and confidence score for the input text (e.g., "I absolutely loved this movie! The acting was superb.").

## Replicating Experiments

To replicate the experiments conducted in the development of NeuroSAM, follow these steps. The experiments focus on data preprocessing with class balancing, multi-granularity LDA feature extraction, and model training with hyperparameter optimization.

### Experiment 1: Data Preprocessing and Class Balancing

- **Objective**: Prepare a balanced dataset for sentiment analysis by cleaning text, augmenting minority classes, and applying SMOTE.
- **Steps**:
  1. Run the preprocessing script as described above.
  2. Verify class distribution in the output `train_data_balanced.csv` to ensure balance across sentiment classes.
- **Expected Results**: The script outputs class distribution statistics before and after balancing, showing improved ratios for minority classes.

### Experiment 2: Multi-Granularity LDA Feature Extraction

- **Objective**: Extract topic distributions at different granularities (15 and 25 topics) to enhance symbolic feature representation.
- **Steps**:
  1. During preprocessing, ensure LDA features are extracted for both 15 and 25 topics (handled automatically in the script).
  2. Check saved NumPy arrays (`train_lda_topics_25.npy`, etc.) for topic distributions.
- **Expected Results**: The preprocessing script logs top words for each topic, indicating thematic coherence in LDA features.

### Experiment 3: Model Training and Evaluation

- **Objective**: Train NeuroSAM with pre-determined hyperparameters and evaluate performance on validation and test sets.
- **Steps**:
  1. Run the training script as described above.
  2. Monitor training progress via logged metrics (accuracy, F1 score, etc.) and saved plots.
  3. Review test set evaluation metrics printed at the end of training.
- **Expected Results**: The model achieves competitive performance metrics (accuracy, precision, recall, F1) on the test set, with confusion matrices visualizing per-class performance.

### Experiment 4: Inference on New Data

- **Objective**: Test the pre-trained model on unseen text inputs to predict sentiment.
- **Steps**:
  1. Run the demo script with custom text inputs.
  2. Note the predicted sentiment and confidence scores.
- **Expected Results**: The model correctly predicts sentiment for sample inputs, aligning with intuitive sentiment interpretation.

## Configuration and Customization

- **Hyperparameters**: Adjust training parameters like `NUM_EPOCHS`, `LEARNING_RATE`, and `BATCH_SIZE` in the training script (`model train.ipynb`) to experiment with different settings.
- **Model Architecture**: Modify the `HybridSentimentModel` class to experiment with different fusion strategies or transformer backbones (default: `albert-base-v2`).
- **Data Paths**: Update `BASE_PATH` and file paths in scripts to match your local directory structure.

## Troubleshooting

- **GPU Memory Issues**: If you encounter out-of-memory errors, reduce `BATCH_SIZE` or disable mixed precision training by setting `USE_MIXED_PRECISION = False` in the training script.
- **Dependency Conflicts**: Ensure all libraries are installed with compatible versions. Use a clean virtual environment if issues persist.
- **Hugging Face Authentication**: Verify your token in the `.env` file if you face authentication errors while downloading or uploading models.


