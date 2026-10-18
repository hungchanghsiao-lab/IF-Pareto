# IF-Pareto: A Risk-Based Pareto Scheduling Approach

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![C++17](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/)

This repository contains the source code for the paper: **"From Heavy-Tailed Volatility to Multi-Resource Stability: A Risk-Based Pareto Scheduling Approach"** (Submitted to IEEE TPDS).

It provides the complete trace-driven simulator (Python) and the distributed cloud testbed (C++17 & gRPC) to evaluate the **IF-Pareto** scheduling framework against heavy-tailed cloud workloads.

---

## Repository Structure
* `if_pareto_simulator.py`: The core trace-driven simulator evaluating P2C, MeanRiskMulti, LBMBP, Prob, IF-Pareto, and ablation variants (`IF-SafeOnly`, `IF-RiskyOnly`). Generates high-resolution, rasterized PDF plots.
* `run_gcp_testbed.sh`: Automated bash script to provision Google Cloud Platform (GCP) VMs, compile the C++17/gRPC control plane and worker services, and measure end-to-end deployment latency over VPC networks.
* `google.csv` / `alibaba.csv`: Extracted workload trace profiles containing resource utilization means and variances.

---

## 1. Running the Trace-Driven Simulator

The simulator relies on standard scientific computing packages and automatically adapts to your execution environment (Local or Google Colab).

### Dependencies Installation
```bash
pip install numpy pandas scikit-learn seaborn matplotlib scipy tqdm
```

### Execution Options

* **Option A: Local Environment**
  1. Ensure `google.csv` and `alibaba.csv` are placed in the same directory as `if_pareto_simulator.py` (which uses `.` as the default root directory). *(Note: If trace files are absent, the simulator generates synthetic fallback data for verification).*
  2. Run the simulator:
     ```bash
     python3 if_pareto_simulator.py
     ```

* **Option B: Google Colab Environment**
  1. The script automatically detects if it is running inside Google Colab and prompts you to mount your Google Drive.
  2. Place your workload traces (`google.csv` and `alibaba.csv`) inside your Google Drive folder at: `/MyDrive/IFexp/`.
  3. Run the script to generate all simulation results, statistical logs, and figures.

Upon completion, the simulator outputs 300 DPI `.pdf` figures inside the `simulation_results/` directory, including Workload CCDFs, P95 Boxplots, Load Imbalance Degree (LID), CDF curves, Sensitivity Analyses, and Multi-dimensional Scatter Plots.

---

## 2. Deploying the C++17/gRPC Distributed Testbed on GCP

To reproduce the sub-millisecond end-to-end latency measurements reported in Section IV-F.3 of our paper, we provide an automated script to deploy a master-worker testbed on GCP.

**Prerequisites:** Ensure you have the `gcloud` CLI installed, authenticated, and configured with an active project.

```bash
# Execute the automated deployment and benchmark
chmod +x run_gcp_testbed.sh
./run_gcp_testbed.sh
```

### What the script does:
1. Provisions 4 `e2-medium` VM instances (1 Master node, 3 physical Worker nodes) within the `asia-east1-b` zone.
2. Installs the `gRPC`, `Protocol Buffers`, and `C++17` build toolchains on all nodes.
3. Compiles `master.cc` and `worker.cc` with `-O3` optimization.
4. Executes 5,000 continuous task placements, measuring wall-clock end-to-end decision latency encompassing $O(1)$ pre-computed tag lookup, $O(N)$ 50-server state scanning, Protobuf RPC serialization, and internal VPC network transmission.
5. Retrieves `latency_results_cpp.csv` to your local environment for empirical CDF plotting.