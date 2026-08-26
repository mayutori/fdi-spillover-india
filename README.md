# FDI Financial Spillover & Credit Access Analysis (India)

A Python-based empirical framework designed to estimate the causal impact of FDI inflows on local Indian firms' financial access and credit conditions using Input-Output (IO) supply chain spillovers and Bartik (Shift-Share) Instrumental Variables (2SLS).

## Overview
Evaluating FDI impacts often suffers from endogeneity due to unobserved regional credit demand shocks. This repository addresses these empirical challenges by:
1. Disentangling **Forward/Backward supply chain spillovers** using Input-Output coefficient matrices.
2. Constructing **Bartik (Shift-Share) Instrumental Variables** combining baseline industry shares with national FDI shocks.
3. Automatically generating publication-ready executive PDF reports comparing naive OLS and IV-2SLS estimates.

## Tech Stack
- **Language**: Python 3.12+
- **Package Manager**: `uv`
- **Econometrics**: `linearmodels` (2SLS/IV), `statsmodels`, `pandas`, `numpy`
- **Reporting**: `matplotlib`

## Quick Start

1. Clone the repository:
```bash
git clone [https://github.com/mayutori/fdi-spillover-india.git](https://github.com/mayutori/fdi-spillover-india.git)
cd fdi-spillover-india