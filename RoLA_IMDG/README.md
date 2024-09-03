# IMDG code Fine-tuning

This project demonstrates how to fine-tune the Google Gemma model for a specific compliance task using MLX. The fine-tuning process is tailored for the CHERRY Shipping Company's compliance requirements.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Fine-tuning](#fine-tuning)
  - [Inference](#inference)
- [Project Structure](#project-structure)
- [License](#license)

## Prerequisites

- Python 3.11+
- MLX
- PyTorch
- Transformers
- Pandas
- PyPDF2

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/gyjong/cherry-compliance-finetuning.git
   cd cherry-compliance-finetuning
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Data Preparation

1. Place your CHERRY compliance PDF file in the `data` directory.

2. Run the data preparation script:
   ```
   python prepare_data.py
   ```

   This script will:
   - Extract text from the PDF
   - Generate question-answer pairs
   - Create train and validation datasets

### Fine-tuning

To fine-tune the Gemma model:

```
python -m mlx_lm.lora \
    --model mlx-community/gemma-2-27b-it-4bit \
    --train \
    --data data \
    --iters 300 \
    --batch-size 4 \
    --learning-rate 1e-5 \
    --steps-per-report 10 \
    --steps-per-eval 10 \
    --adapter-path checkpoints/adapters \
    --save-every 10 \
    --max-seq-length 2048 \
    --seed 42 \
    --lora-layers 16
```

### Inference

To run inference with the fine-tuned model:

```
python -m mlx_lm.generate \
    --model mlx-community/gemma-2-27b-it-4bit \
    --adapter-path checkpoints/adapters \
    --prompt "Your question here" \
    --max-tokens 256 \
    --temp 0.5
```

## Project Structure

```
cherry-compliance-finetuning/
├── data/
│   ├── CHERRY_compliance.pdf
│   ├── train.jsonl
│   └── valid.jsonl
├── checkpoints/
│   └── adapters/
├── prepare_data.py
├── fine_tune.py
├── inference.py
├── requirements.txt
└── README.md
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.