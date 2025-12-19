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
