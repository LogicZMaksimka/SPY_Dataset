# SPY: Enhancing Privacy with Synthetic PII Detection Dataset

We proudly present the **SPY Dataset**, a novel synthetic dataset for the task of **Personal Identifiable Information (PII)** detection. This dataset highlights the importance of safeguarding PII in modern data processing and serves as a benchmark for advancing privacy-preserving technologies.

## Key Highlights

- **Innovative Generation:** We present a methodology for developing a SPY dataset and compare it to other methodologies used for creating a synthetic PII datasets. Our approach does not require any external data and can be applied to any knowledge domain.
- **Benchmark for PII Detection:** We open-source a SPY dataset containing 4,491  medical consultations and 4,197 questions in legal domain, that is specifically developed to highlight the contrast between an average task of named entity recognition and more fine-grained tasks of PII detection. 



<div style="text-align: center;">
    <img src="img/prompting_pipeline.png" alt="Generation Pipeline" width="50%">
</div>

## Dataset Statistics

| **Entity type** | **Legal questions** | **Medical Consultations** |
|----------------|----------------------|---------------------------|
| url            | 3281                 | 3388                      |
| email          | 3707                 | 4065                      |
| username       | 3854                 | 3968                      |
| address        | 3592                 | 4121                      |
| name           | 4046                 | 4276                      |
| phone number   | 3304                 | 3847                      |
| id_num         | 3186                 | 4024                      |
| **Total**      | **28170**            | **30789**                 |
*Table: Number of generated PII entities by type.*


## Data Format

The dataset is divided into two files, categorized by domain:

1. **Legal Questions Dataset**
2. **Medical Consultations Dataset**

Additionally, we provide small sample-size files for both domains.

### **File Structure**
The dataset is stored in **.jsonl** format, where each line contains a dictionary with the following keys:

- **text:** The generated text.
- **tokens:** A list of tokens created by splitting the generated text by whitespace.
- **trailing_whitespace:** A boolean indicating whether a token has trailing whitespace.
- **labels:** A list of labels for each token in **BIO** format.
