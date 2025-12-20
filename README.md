# mlproj-masters

## Group Members

- Diogo Carvalho - 20221935
- Luiza Salum - 20221902
- Ricardo Pereira - 20250343

## Installation Guide

This project is managed with [uv](https://github.com/astral-sh/uv), pip can
still be used but uv is recommended.

1. Run the following from the root of the repository:

   ```bash
   # If using uv:
   uv sync

   # Or, if you don't have uv, create a virtual environment, then run
   pip install -r requirements.txt
   ```

1. To start the predictions interface, run:

   ```bash
   poe serve
   ```

1. Head to localhost:8000 after the app is done loading

## File Locations

- [Main file](./src/main.ipynb)
- [Custom pipeline transformers](./src/funcs/custom_transformers.py)
- [Data importing and exploration](./src/funcs/data_import.py)
- [Pipeline building functions](./src/funcs/pipeline.py)
- [Preprocessing (Excluding custom transformers)](./src/funcs/preprocessing.py)
- [API for the interface to get new predictions](./src/mlproj_web/api.py)
