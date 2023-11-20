# NameGuess Evaluation Data

Contains the evaluation datasets in JSON format, which are designed to test the performance of models in expanding abbreviated column names in tabular data. They include annotated examples from different city open data: Chicago, Los Angeles, and San Francisco.

## Dataset Overview

- `eval_chicago.json`: Evaluation data for Chicago Open Data (https://data.cityofchicago.org/)
- `eval_la.json`: Evaluation data for Los Angeles Open Data (https://data.lacity.org/)
- `eval_sf.json`: Evaluation data for San Francisco Open Data (https://datasf.org/opendata/)

Each file contains annotated examples of abbreviated column names based on their corresponding original forms, along with additional contextual information.

## File Structure

Each JSON file is structured with multiple entries, where each entry represents a case of column name abbreviation. The structure of an entry is as follows:

- `table_partition`: The data source partition, indicating the origin of the dataset.
- `table_id`: A unique identifier for the source table.
- `table_prompt`: A string containing a list of column names and a sample row from the table, providing context.
- `query`: The formatted prompt query representing the task of expanding the abbreviated column names.
- `gt_label`: An array of the expanded names (ground truth labels) of the abbreviated column headers.
- `technical_name`: An array of the abbreviated column names as they appear in the dataset.
- `difficulty`: An array of integers indicating the difficulty level for expanding each column name.

Example entry:
```json
{
  {
    "table_partition": "chicago_opendata",
    "table_id": "bc6b-sq4u",
    "table_prompt": "column names : make, lst_inspec_mnth, Mdl, veh_color, mnthRprtd, num_trips\nrow 1 : Scion, 2021-04, Tacoma, White, 2022-04, 319.0\n",
    "query": "As abbreviations of column names from a table, make | lst_inspec_mnth | Mdl | veh_color | mnthRprtd | num_trips stand for",
    "gt_label": [
      "Make",
      "last inspection month",
      "Model",
      "Color",
      "MONTH REPORTED",
      "number of trips"
    ],
    "technical_name": [
      "make",
      "lst_inspec_mnth",
      "Mdl",
      "veh_color",
      "mnthRprtd",
      "num_trips"
    ],
    "difficulty": [
      0,
      1,
      1,
      2,
      2,
      2
    ]
  }
}

