# NameD Entity Extraction for Historical Text Similarity: A Case Study of Suvarnabhumi

## Overview
This project, hosted at [ThomasRieger/NER_Historical](https://github.com/ThomasRieger/NER_Historical), implements a Named Entity Recognition (NER) and Part-of-Speech (POS) tagging system designed for Thai historical texts, with a specific focus on the historical context of Suvarnabhumi. The system extracts named entities (e.g., persons, locations, organizations) and linguistic features from Thai texts to facilitate text similarity analysis and support historical research.

The project addresses the challenges of processing Thai text, such as its lack of word boundaries and complex script, using NLP tools optimized for the Thai language. It is intended for researchers, historians, and NLP practitioners interested in analyzing Thai historical documents.

## Features
- **Named Entity Recognition (NER):** Extracts entities like persons, locations, and organizations from Thai historical texts.
- **Part-of-Speech (POS) Tagging:** Assigns grammatical roles (e.g., noun, verb, adjective) to words for linguistic analysis.
- **Text Similarity Analysis:** Computes similarity scores between texts based on extracted entities and POS features.
- **Case Study Focus:** Optimized for Suvarnabhumi-related texts, with potential applications to other Thai historical corpora.

## Installation

### Prerequisites
- Python 3.8 or higher
- `pip` (Python package manager)
- Virtual environment (recommended)
- Git (for cloning the repository)

### Steps
1. **Clone the Repository**
   ```bash
   git clone https://github.com/ThomasRieger/NER_Historical.git
   cd NER_Historical
   ```

2. **Set Up a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   The `requirements.txt` file with the following content:
   ```
   streamlit
   torch
   transformers
   datasets
   pandas
   attacut
   ```
   Then run:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the NER and POS Tagger

## TESTING
1. Add the Text in **line 166** in **Final_v1\testfinalmodel.py** then run the file

### Example
**Input Text**
```
สุวรรณภูมิเป็นเมืองโบราณที่มีความสำคัญในประวัติศาสตร์ไทย
```

## Project Structure
```
NER_Historical/
├── Final_v1                                    # Main script for NER and POS tagging
|  ├── AIFORTHAI-LST20Corpus/LST20_Corpus_final # Corpus For Training
|  ├── ner_modelfinal                           # Model_v1 (Not the best)
|  ├── out                                      
|  ├── testfinalmodel.py                        # Model Testing File
|  ├── train.py                                 # Model Training File
|  └── ผลลัพธ์ตอนเทรน                            # v1 Result
├── Website                                     # Script for text similarity analysis
├── requirements.txt                            # Dependencies
├── .gitignore                                  # Directory for input text files
├── .gitattribute                               # Directory for output results
└── README.md                                   # This file
```

**Note:** If your repository includes additional files or folders (e.g., pre-trained models, scripts, or datasets), update the structure above to reflect the actual contents. The current structure is inferred based on typical NLP project layouts.

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Make your changes and commit:
   ```bash
   git commit -m "Add your feature description"
   ```
4. Push to the branch:
   ```bash
   git push origin feature/your-feature-name
   ```
5. Open a pull request on the [ThomasRieger/NER_Historical](https://github.com/ThomasRieger/NER_Historical) repository.

Please ensure your code adheres to PEP 8 standards and includes relevant tests.

## Contact
For questions, suggestions, or issues, please:
- Open an issue on the [GitHub repository](https://github.com/ThomasRieger/NER_Historical).
- Contact the maintainer at [Thomasrieger32@gmail.com](mailto:Thomasrieger32@gmail.com)

## Acknowledgments
- [LST20 Dataset](https://huggingface.co/datasets/lst-nectec/lst20) for Thai Language Datasets
- [attacut](https://github.com/PyThaiNLP/attacut) for word tokenizer
- [WangchanBERTa base model](https://huggingface.co/airesearch/wangchanberta-base-att-spm-uncased) for base model
- [Named Entity Recognition for Thai Historical Data](https://ieeexplore.ieee.org/document/10613644) Historical researchers and contributors to the Suvarnabhumi case study.
