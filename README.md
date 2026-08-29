# IF-Pareto: A Risk-Based Pareto Scheduling Approach

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![C++17](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/)

This repository contains the source code for the paper: **"From Heavy-Tailed Volatility to Multi-Resource Stability: A Risk-Based Pareto Scheduling Approach"** (Submitted to IEEE TPDS).

It provides the complete trace-driven simulator (Python) and the real-world distributed testbed (C++ & gRPC) to evaluate the **IF-Pareto** scheduling framework against heavy-tailed cloud workloads.

## 🗂 Repository Structure
*   `if_pareto_simulator.py`: The core trace-driven simulator evaluating P2C, MeanRiskMulti, LBMBP, Prob, and IF-Pareto. Includes ablation studies and generates academic-grade PDF plots.
*   `run_gcp_testbed.sh`: Automated bash script to provision Google Cloud Platform (GCP) VMs, compile the C++17 gRPC framework, and measure genuine sub-millisecond end-to-end deployment latency.
*   `google.csv` / `alibaba.csv`: Sample traces containing extracted resource means and variances *(Note: Please ensure the trace data is placed in the root directory before execution)*.

## 🚀 1. Running the Trace-Driven Simulator

The simulator relies on standard data science packages.
```bash
# Install dependencies
pip install numpy pandas scikit-learn seaborn matplotlib scipy tqdm

# Execute the simulation
python3 if_pareto_simulator.py

Upon completion, the simulator will automatically generate high-resolution, rasterized .pdf files inside the simulation_results/ directory, including P95 Boxplots, Load Imbalance Degree (LID), CDF curves, and Multi-dimensional Scatter Plots.