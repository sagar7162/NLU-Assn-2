# NLU Assignment 2: Word Embeddings & Character-Level RNN

**Student ID:** B23CM1034
**Course:** Natural Language Understanding

This repository contains the implementation for Assignment 2, which covers:
- **Problem 1:** Learning Word Embeddings from IIT Jodhpur Data
- **Problem 2:** Character-Level Name Generation Using RNN Variants

---

## Repository Structure

```
Assn 2/
├── B23CM1034-A2.ipynb      # Main implementation notebook (run this)
├── B23CM1034_Report.tex     # Detailed assignment report
├── README.md               # This file
├── data/
│   ├── raw_corpus.txt      # Raw scraped text from IIT Jodhpur
│   ├── clean_corpus.txt    # Preprocessed corpus
│   └── TrainingNames.txt   # 1000 Indian names for RNN training
├── results/
    ├── wordcloud.png                    # Word cloud visualization
    ├── top_words_frequency.png          # Top 20 words bar chart
    ├── cbow_pca_visualization.png       # CBOW PCA projection
    ├── skipgram_pca_visualization.png   # Skip-gram PCA projection
    ├── cbow_tsne_visualization.png      # CBOW t-SNE projection
    ├── skipgram_tsne_visualization.png  # Skip-gram t-SNE projection
    ├── training_curves_and_params.png   # Model training curves
    ├── name_generation_metrics.csv      # Quantitative evaluation results
    ├── generated_name_samples.txt       # Generated name samples
    └── failure_modes.csv                # Failure mode analysis

```

---

## Requirements

### Python Version
- Python 3.8 or higher

### Dependencies

Install the required packages using pip:

```bash
pip install numpy pandas matplotlib seaborn torch scikit-learn wordcloud requests beautifulsoup4 scipy
```

Or install all dependencies at once:

```bash
pip install numpy pandas matplotlib seaborn torch scikit-learn wordcloud requests beautifulsoup4 scipy pypdf
```

### Optional Dependencies

- `pypdf` - For extracting text from PDF documents during web scraping (optional but recommended)

```bash
pip install pypdf
```

---

## How to Run

### Option 1: Run the Complete Notebook

1. **Open the notebook** in Jupyter Notebook, JupyterLab, or VS Code:
   ```bash
   jupyter notebook B23CM1034-A2.ipynb
   ```

2. **Run all cells sequentially** (Kernel → Run All) or run cells one by one.

### Option 2: Run Specific Sections

The notebook is organized into clearly marked sections:

#### Problem 1: Word Embeddings
1. **Task 1.1:** Web Scraping - Collects data from IIT Jodhpur website
2. **Task 1.2:** Preprocessing - Cleans and tokenizes the corpus
3. **Task 1.3:** Word Cloud - Generates visualization
4. **Task 2:** Model Training - Trains CBOW and Skip-gram models
5. **Task 3:** Semantic Analysis - Nearest neighbors and analogies
6. **Task 4:** Visualization - PCA and t-SNE projections

#### Problem 2: Name Generation
1. **Task 0:** Dataset - Loads Indian names
2. **Task 1:** Model Implementation - Defines RNN, BiLSTM, and Attention models
3. **Task 2:** Training - Trains all three models
4. **Task 3:** Evaluation - Generates names and computes metrics

---

## Data Files

### Pre-existing Data

The following data files are included and will be loaded automatically:

- `data/raw_corpus.txt` - Raw text scraped from IIT Jodhpur website
- `data/clean_corpus.txt` - Preprocessed and cleaned corpus
- `data/TrainingNames.txt` - 1000 Indian names for RNN training

### Regenerating Data

**To regenerate the corpus from scratch:**
1. Delete `data/raw_corpus.txt`
2. Run the web scraping cell (Task 1.1)
3. The script will scrape IIT Jodhpur website and save new data

**Note:** Web scraping depends on website availability and structure. Pre-scraped data is provided for reproducibility.

---

## Expected Output

### Problem 1 Outputs

1. **Dataset Statistics:**
   - Total Tokens: ~70,000
   - Vocabulary Size: ~4,800
   - Word cloud and frequency charts saved to `results/`

2. **Model Training:**
   - Multiple CBOW and Skip-gram models trained with different hyperparameters
   - Training curves saved to `results/training_curves_and_params.png`

3. **Semantic Analysis:**
   - Nearest neighbors for: research, student, phd, exam, course
   - Analogy results for UG:BTech::PG:?, Student:Learning::Professor:?, etc.

4. **Visualizations:**
   - PCA and t-SNE projections saved to `results/`

### Problem 2 Outputs

1. **Model Parameters:**
   - Vanilla RNN: ~19,000 parameters
   - BiLSTM: ~161,000 parameters
   - RNN with Attention: ~131,000 parameters

2. **Generation Metrics:**
   - Novelty Rate: 53-100% (varies by model)
   - Diversity: 100% (all models)
   - Results saved to `results/name_generation_metrics.csv`

3. **Generated Samples:**
   - 25 sample names per model saved to `results/generated_name_samples.txt`

---

### Quick Mode

For faster testing/debugging, set `QUICK_MODE_RNN = True` in the RNN training cell:
- Reduces hidden size, epochs, and batches
- Training completes in ~5 minutes instead of ~20

---

## Results Summary

### Word Embeddings (Problem 1)

| Model | Best Configuration | Final Loss | Parameters |
|-------|-------------------|------------|------------|
| CBOW | dim=200, window=5, lr=0.01 | 0.2596 | 1,950,063 |
| Skip-gram | dim=100, window=2, neg=15, lr=0.001 | 0.2840 | 972,600 |

**Dataset Statistics:**
- Total Tokens: 70,637
- Vocabulary Size: 4,863
- Type-Token Ratio: 0.0688

### Name Generation (Problem 2)

| Model | Novelty Rate | Diversity | Parameters |
|-------|--------------|-----------|------------|
| BiLSTM | 100% | 100% | ~161,000 |
| Vanilla RNN | 60.4% | 100% | ~19,000 |
| RNN with Attention | 53.0% | 100% | ~131,000 |

**Dataset Statistics:**
- Total Names: 1,000
- Character Vocabulary Size: 24 (21 unique characters + 3 special tokens)

---