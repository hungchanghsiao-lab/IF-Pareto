# IF-Pareto: A Risk-Based Pareto Scheduling Approach

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![C++17](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/)

This repository contains the source code for the paper: **"From Heavy-Tailed Volatility to Multi-Resource Stability: A Risk-Based Pareto Scheduling Approach"** (Submitted to IEEE TPDS).

It provides the complete trace-driven simulator (Python) and the real-world distributed testbed (C++ & gRPC) to evaluate the **IF-Pareto** scheduling framework against heavy-tailed cloud workloads.

---

## 🗂 Repository Structure
* `if_pareto_simulator.py`: The core trace-driven simulator evaluating P2C, MeanRiskMulti, LBMBP, Prob, and IF-Pareto. Includes ablation studies and generates academic-grade PDF plots.
* `run_gcp_testbed.sh`: Automated bash script to provision Google Cloud Platform (GCP) VMs, compile the C++17 gRPC framework, and measure genuine sub-millisecond end-to-end deployment latency.
* `google.csv` / `alibaba.csv`: Sample traces containing extracted resource means and variances.

---

## 🚀 1. Running the Trace-Driven Simulator

The simulator relies on standard data science packages and automatically adapts to your running environment (Local or Google Colab).

### Dependencies Installation
```bash
pip install numpy pandas scikit-learn seaborn matplotlib scipy tqdm
```

### Execution Options

* **Option A: Local Environment**
  1. Ensure `google.csv` and `alibaba.csv` are placed in the same directory as `if_pareto_simulator.py` (which acts as the root path `.` by default). *(Note: If trace files are missing, the simulator will automatically generate synthetic fallback data).*
  2. Run the simulator:
     ```bash
     python3 if_pareto_simulator.py
     ```

* **Option B: Google Colab Environment**
  1. The script automatically detects if it is running inside Google Colab and will prompt you to mount your Google Drive.
  2. Ensure your workload traces (`google.csv` and `alibaba.csv`) are placed inside your Google Drive folder at: `/MyDrive/IFexp/`.
  3. Run the cells to generate all simulation results, metrics, and plots.

Upon completion, the simulator will automatically generate high-resolution, rasterized `.pdf` files inside the `simulation_results/` directory, including P95 Boxplots, Load Imbalance Degree (LID), CDF curves, and Multi-dimensional Scatter Plots.

---

## ☁️ 2. Deploying the C++ GCP End-to-End Testbed

To reproduce the sub-millisecond latency results highlighted in our paper, we provide a fully automated script to deploy a master-worker architecture on GCP.

**Prerequisites:** Ensure you have the `gcloud` CLI installed and authenticated.

```bash
# Execute the automated deployment
chmod +x run_gcp_testbed.sh
./run_gcp_testbed.sh
```

### What the script does:
1. Provisions 4 `e2-medium` instances (1 Master, 3 Workers) in the same Virtual Private Cloud (VPC).
2. Automatically installs the `gRPC` and `Protocol Buffers` C++ toolchains.
3. Compiles `master.cc` and `worker.cc` using `c++17`.
4. Executes 5,000 asynchronous task placements, logging the genuine wall-clock latency encompassing algorithmic scanning ($O(N)$), serialization, and VPC network transmission.
5. Downloads the `latency_results_cpp.csv` to your local environment for CDF plotting.

---
