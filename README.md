## NameGuess: Column Name Expansion for Tabular Data

The NameGuess task seeks to expand abbreviated column names into their full forms.
For instance, it would expand "D_ID" into "Department ID" and "E_NAME" into "Employee Name."

### Getting Started
Create a conda environment
```
conda create -n nameguess python=3.11 --file requirements.txt
conda activate nameguess
python -m spacy download en_core_web_sm
```

### Training Data Creation
Scripts for two key compoenents in the training data creation.

#### Logical Name Identification
Example
```
python src/cryptic_identifier.py --text "nycpolicedepartment"
```

#### Abbreviation Generation
Example
```
python src/cryptic_generator.py --text "customer name"
```

### Benchmarks
We provide a human-annotated evaluation benchmark including 9,218 column names on 895 tables. [Link (TBA)]

### Evaluation Scripts
```
python run_eval.py --model_name gpt-4
```

### Cite
Please cite the paper if you use the codebase in your work:

```
@article{nameguess,
  title     = {NameGuess: Column Name Expansion for Tabular Data},
  author    = {Zhang, Jiani and Shen, Zhengyuan and Srinivasan, Balasubramaniam and Wang, Shen and Rangwala, Huzefa and Karypis, George},
  year      = {2023}
}
```

### Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

### License

This project is licensed under the Apache-2.0 License.

