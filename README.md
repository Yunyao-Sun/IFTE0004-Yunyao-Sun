# Volatility Forecasting Project

This project builds one equal-weight intraday index from:

- AAPL.txt
- AMZN.txt
- JPM.txt

Then it forecasts realized volatility using HAR-family, regularized linear models, tree models, gradient boosting, and neural networks.

## How to run

Put these files in the same folder as `main.py`:

- AAPL.txt
- AMZN.txt
- JPM.txt

Then run:

```bash
python main.py
```

Outputs are saved in the `outputs` folder.

## Main setting

Edit `config.py`:

```python
NN_SEEDS = 5
```

Use `5` for a quick test. Use `20` or `100` for the final run.
# IFTE0004-Yunyao-Sun
Reference paper: A Machine Learning Approach to Volatility Forecasting

Raw high-frequency data files and generated output files are excluded due to file size limits. To reproduce the results, place AAPL.txt, AMZN.txt, and JPM.txt in the same folder as main.py, then run:

python main.py
