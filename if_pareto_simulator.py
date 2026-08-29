#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IF-Pareto Trace-Driven Simulator
================================
Evaluates the Dual-Regime IF-Pareto scheduler against state-of-the-art baselines
(P2C, MeanRiskMulti, LBMBP, Prob) using real-world heavy-tailed traces.

Submission: IEEE TPDS
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import os
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import sys
from scipy.stats import norm
import time
import warnings

# Suppress minor warnings for clean execution output
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# =============================================================================
# Global Configuration & Hyperparameters
# =============================================================================
SCRIPT_VERSION = "v4.0_Release"
N_SERVERS = 50                 # Logical cluster size
M_TASKS = 4000                 # Tasks per simulation round (High contention density)
R_ROUNDS = 100                 # Independent simulation rounds for steady-state
Z_SCORE_P95 = 1.645            # Gaussian quantile proxy for 95th percentile risk
N_ROUNDS_SENSITIVITY = 100     # Rounds for hyperparameter sensitivity analysis

# Evaluation switches
RUN_COMPARISON = True
RUN_LATENCY_TEST = True
RUN_SENSITIVITY_ANALYSIS = True

SELECTED_WORKLOADS = ['google', 'alibaba']
ENABLED_SCHEDULERS_LIST = ['P2C', 'MeanRiskMulti', 'IF_Pareto', 'LBMBP', 'PROB', 'IF_SafeOnly', 'IF_RiskyOnly']
ENABLE_HARD_CAPACITY_CHECK = True

# Isolation Forest (IF) Hyperparameters
IF_CONTAMINATION = 0.1
IF_N_ESTIMATORS = 100
IF_MAX_SAMPLES = 256
IF_MAX_FEATURES = 1.0
IF_BOOTSTRAP = False
ENABLE_NORMALIZATION = False
IF_VERBOSE = 0
IF_N_JOBS = -1

# Directory Setup
if 'google.colab' in sys.modules:
    print("Colab environment detected. Mounting Google Drive...")
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("Google Drive mounted successfully.")
    except Exception as e:
        print(f"Failed to mount Google Drive: {e}")
    ROOT_DIR = "/content/drive/MyDrive/IFexp"
else:
    print("Local environment detected. Using current working directory.")
    ROOT_DIR = "."

BASE_OUTPUT_DIR = os.path.join(ROOT_DIR, "simulation_results")
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

ALL_WORKLOAD_PROFILES = {
    'alibaba': os.path.join(ROOT_DIR, "alibaba.csv"),
    'google': os.path.join(ROOT_DIR, "google.csv")
}

# =============================================================================
# Data Loading & ML Pipeline
# =============================================================================
def load_raw_data(filepath):
    """Loads workload trace and normalizes multi-dimensional features if required."""
    logging.info(f"Loading trace: '{filepath}'...")
    try:
        if not os.path.exists(filepath):
            logging.warning(f"File {filepath} not found. Generating synthetic data.")
            data = {
                'cpu_mean': np.random.rand(5000), 'mem_mean': np.random.rand(5000),
                'cpu_variance': np.random.rand(5000) * 0.05, 'mem_variance': np.random.rand(5000) * 0.05
            }
            task_pool = pd.DataFrame(data)
        else:
            task_pool = pd.read_csv(filepath)
    except Exception as e:
        logging.error(f"Failed to load trace: {e}")
        return None

    features = ['cpu_mean', 'mem_mean', 'cpu_variance', 'mem_variance']
    if not all(f in task_pool.columns for f in features):
        logging.error(f"Missing mandatory features in CSV: {features}")
        return None

    if ENABLE_NORMALIZATION:
        logging.info("Applying Min-Max Scaling [0, 1] to task features...")
        scaler = MinMaxScaler()
        task_pool[features] = scaler.fit_transform(task_pool[features])

    return task_pool

def apply_isolation_forest(df, contamination):
    """Phase 1: Offline Risk Analysis. Segregates tasks into Safe(1) and Risky(-1)."""
    features = ['cpu_mean', 'mem_mean', 'cpu_variance', 'mem_variance']
    df_labeled = df.copy()

    if_task_analyzer = IsolationForest(
        contamination=contamination,
        n_estimators=IF_N_ESTIMATORS,
        max_samples=IF_MAX_SAMPLES,
        max_features=IF_MAX_FEATURES,
        bootstrap=IF_BOOTSTRAP,
        n_jobs=IF_N_JOBS,
        verbose=IF_VERBOSE,
        random_state=42
    )
    df_labeled['if_risk_label'] = if_task_analyzer.fit_predict(df_labeled[features])
    return df_labeled, if_task_analyzer

# =============================================================================
# Cluster State Management
# =============================================================================
def initialize_servers(n):
    return [{'id': i, 'task_count': 0, 
             'total_cpu_mean': 0.0, 'total_cpu_variance': 0.0, 
             'total_mem_mean': 0.0, 'total_mem_variance': 0.0} for i in range(n)]

def update_server(server, task):
    server['task_count'] += 1
    server['total_cpu_mean'] += task['cpu_mean']
    server['total_cpu_variance'] += task['cpu_variance']
    server['total_mem_mean'] += task['mem_mean']
    server['total_mem_variance'] += task['mem_variance']

def get_fallback_server(servers):
    best_s = None
    min_load = float('inf')
    for s in servers:
        load = max(s['total_cpu_mean'], s['total_mem_mean'])
        if load < min_load:
            min_load = load
            best_s = s
    return best_s

# =============================================================================
# Scheduling Algorithms (Baselines & Proposed)
# =============================================================================
def schedule_p2c(task, servers):
    """Baseline 1: Power of Two Choices (Workload-Oblivious)"""
    if not ENABLE_HARD_CAPACITY_CHECK:
        s1, s2 = np.random.choice(servers, 2, replace=False)
        target = s1 if s1['task_count'] <= s2['task_count'] else s2
        update_server(target, task)
        return

    valid_pair_found = False
    for _ in range(10):
        s1, s2 = np.random.choice(servers, 2, replace=False)
        valid_s1 = (s1['total_cpu_mean'] + task['cpu_mean'] <= 1.0) and (s1['total_mem_mean'] + task['mem_mean'] <= 1.0)
        valid_s2 = (s2['total_cpu_mean'] + task['cpu_mean'] <= 1.0) and (s2['total_mem_mean'] + task['mem_mean'] <= 1.0)

        if valid_s1 and valid_s2:
            target = s1 if s1['task_count'] <= s2['task_count'] else s2
            update_server(target, task)
            valid_pair_found = True
            break
        elif valid_s1:
            update_server(s1, task)
            valid_pair_found = True
            break
        elif valid_s2:
            update_server(s2, task)
            valid_pair_found = True
            break

    if not valid_pair_found:
        target = get_fallback_server(servers)
        update_server(target, task)

def schedule_meanrisk_multi(task, servers):
    """Baseline 2: MeanRiskMulti (Multi-dimensional Statistical Optimizer)"""
    best_server = None
    min_future_risk = float('inf')
    target_pool = servers

    if ENABLE_HARD_CAPACITY_CHECK:
        valid_servers = [s for s in servers if (s['total_cpu_mean'] + task['cpu_mean'] <= 1.0) and (s['total_mem_mean'] + task['mem_mean'] <= 1.0)]
        target_pool = valid_servers if valid_servers else servers

    for s in target_pool:
        cpu_risk = (s['total_cpu_mean'] + task['cpu_mean']) + Z_SCORE_P95 * np.sqrt(s['total_cpu_variance'] + task['cpu_variance'] + 1e-9)
        mem_risk = (s['total_mem_mean'] + task['mem_mean']) + Z_SCORE_P95 * np.sqrt(s['total_mem_variance'] + task['mem_variance'] + 1e-9)
        future_risk = max(cpu_risk, mem_risk)

        if future_risk < min_future_risk:
            min_future_risk = future_risk
            best_server = s

    if best_server is None: best_server = get_fallback_server(servers)
    update_server(best_server, task)

def schedule_lbmbp(task, servers, current_global_load):
    """Baseline 3: Load Balanced Multi-Dimensional Bin-Packing (Static Geometric)"""
    n_servers = len(servers)
    ideal_cpu_load = (current_global_load['cpu'] + task['cpu_mean']) / n_servers
    ideal_mem_load = (current_global_load['mem'] + task['mem_mean']) / n_servers

    best_server = None
    min_displacement = float('inf')
    w_cpu, w_mem = 0.5, 0.5

    target_pool = servers
    if ENABLE_HARD_CAPACITY_CHECK:
        valid_servers = [s for s in servers if (s['total_cpu_mean'] + task['cpu_mean'] <= 1.0) and (s['total_mem_mean'] + task['mem_mean'] <= 1.0)]
        target_pool = valid_servers if valid_servers else servers

    for s in target_pool:
        displacement = w_cpu * abs(ideal_cpu_load - (s['total_cpu_mean'] + task['cpu_mean'])) + \
                       w_mem * abs(ideal_mem_load - (s['total_mem_mean'] + task['mem_mean']))
        if displacement < min_displacement:
            min_displacement = displacement
            best_server = s

    if best_server is None: best_server = get_fallback_server(servers)
    update_server(best_server, task)
    current_global_load['cpu'] += task['cpu_mean']
    current_global_load['mem'] += task['mem_mean']

def schedule_prob(task, servers):
    """Baseline 4: Probability Density Optimizer"""
    best_server = None
    max_safety_margin = float('-inf')
    capacity = 1.0

    target_pool = servers
    if ENABLE_HARD_CAPACITY_CHECK:
        valid_servers = [s for s in servers if (s['total_cpu_mean'] + task['cpu_mean'] <= 1.0) and (s['total_mem_mean'] + task['mem_mean'] <= 1.0)]
        target_pool = valid_servers if valid_servers else servers

    for s in target_pool:
        mu_cpu = s['total_cpu_mean'] + task['cpu_mean']
        std_cpu = np.sqrt(s['total_cpu_variance'] + task['cpu_variance']) + 1e-9
        
        mu_mem = s['total_mem_mean'] + task['mem_mean']
        std_mem = np.sqrt(s['total_mem_variance'] + task['mem_variance']) + 1e-9

        z_cpu = (capacity - mu_cpu) / std_cpu
        z_mem = (capacity - mu_mem) / std_mem
        min_z = min(z_cpu, z_mem)

        if min_z > max_safety_margin:
            max_safety_margin = min_z
            best_server = s

    if best_server is None: best_server = get_fallback_server(servers)
    update_server(best_server, task)

def schedule_if_pareto(task, servers, z_score=Z_SCORE_P95):
    """Proposed Method: IF-Pareto Dual-Regime Scheduler"""
    best_server = None
    target_pool = servers

    if ENABLE_HARD_CAPACITY_CHECK:
        valid_servers = [s for s in servers if (s['total_cpu_mean'] + task['cpu_mean'] <= 1.0) and (s['total_mem_mean'] + task['mem_mean'] <= 1.0)]
        target_pool = valid_servers if valid_servers else servers 

    if task['if_risk_label'] == -1:
        # Policy B: Stability First (Cantelli-inspired Variance Minimization via Gaussian Proxy)
        min_combined_risk = float('inf')
        for s in target_pool:
            cpu_risk = s['total_cpu_mean'] + z_score * np.sqrt(s['total_cpu_variance'] + 1e-9)
            mem_risk = s['total_mem_mean'] + z_score * np.sqrt(s['total_mem_variance'] + 1e-9)
            combined_risk = cpu_risk + mem_risk
            if combined_risk < min_combined_risk:
                min_combined_risk = combined_risk
                best_server = s
    else:
        # Policy A: Pareto Efficiency (Min-Max Majorization)
        min_future_max_load = float('inf')
        for s in target_pool:
            future_max_load = max(s['total_cpu_mean'] + task['cpu_mean'], s['total_mem_mean'] + task['mem_mean'])
            if future_max_load < min_future_max_load:
                min_future_max_load = future_max_load
                best_server = s

    if best_server is None: best_server = get_fallback_server(servers)
    update_server(best_server, task)

# Ablation Study Variants
def schedule_if_pareto_safe_only(task, servers, z_score=Z_SCORE_P95):
    """Ablation Baseline: Forces all tasks to undergo Policy A."""
    fake_task = task.copy()
    fake_task['if_risk_label'] = 1 
    schedule_if_pareto(fake_task, servers, z_score)

def schedule_if_pareto_risky_only(task, servers, z_score=Z_SCORE_P95):
    """Ablation Baseline: Forces all tasks into Policy B."""
    fake_task = task.copy()
    fake_task['if_risk_label'] = -1 
    schedule_if_pareto(fake_task, servers, z_score)

# =============================================================================
# Simulation Routines
# =============================================================================
def run_latency_benchmark(task_pool, if_model, output_dir, n_rounds=10, m_tasks_per_round=1000, profile_name=""):
    logging.info(f"--- Running Algorithmic Latency Benchmark: {profile_name} ---")
    warnings.filterwarnings("ignore", category=UserWarning)

    latency_data = {'P2C': [], 'MeanRiskMulti': [], 'LBMBP': [], 'PROB': [], 'IF-Pareto': [], 'IF-Inference': []}
    features = ['cpu_mean', 'mem_mean', 'cpu_variance', 'mem_variance']

    for r in tqdm(range(n_rounds), desc="Latency Test Progress"):
        schedulers = {
            'P2C': initialize_servers(N_SERVERS), 'MeanRiskMulti': initialize_servers(N_SERVERS),
            'IF-Pareto': initialize_servers(N_SERVERS), 'LBMBP': initialize_servers(N_SERVERS),
            'PROB': initialize_servers(N_SERVERS)
        }
        lbmbp_global_load = {'cpu': 0.0, 'mem': 0.0}
        tasks = task_pool.sample(n=m_tasks_per_round, replace=True)

        for i in range(len(tasks)):
            task = tasks.iloc[i]

            t0 = time.perf_counter_ns()
            schedule_p2c(task, schedulers['P2C'])
            latency_data['P2C'].append(time.perf_counter_ns() - t0)

            t0 = time.perf_counter_ns()
            schedule_meanrisk_multi(task, schedulers['MeanRiskMulti'])
            latency_data['MeanRiskMulti'].append(time.perf_counter_ns() - t0)

            t0 = time.perf_counter_ns()
            schedule_lbmbp(task, schedulers['LBMBP'], lbmbp_global_load)
            latency_data['LBMBP'].append(time.perf_counter_ns() - t0)

            t0 = time.perf_counter_ns()
            schedule_prob(task, schedulers['PROB'])
            latency_data['PROB'].append(time.perf_counter_ns() - t0)

            t0 = time.perf_counter_ns()
            schedule_if_pareto(task, schedulers['IF-Pareto'])
            latency_data['IF-Pareto'].append(time.perf_counter_ns() - t0)

            # Phase-1 Inference Measurement (Decoupled)
            feat_vector = task[features].values.reshape(1, -1)
            t0 = time.perf_counter_ns()
            if_model.predict(feat_vector)
            latency_data['IF-Inference'].append(time.perf_counter_ns() - t0)

    df_latency = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in latency_data.items()])) / 1000.0
    filename_prefix = f"{profile_name}_" if profile_name else ""
    df_latency.to_csv(os.path.join(output_dir, f"{filename_prefix}latency_benchmark_raw.csv"), index=False)
    return df_latency

def plot_latency_comparison(df, output_dir, profile_name=""):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 18, 'axes.labelsize': 20})
    means, stds = df.mean(), df.std()
    plot_cols = ['P2C', 'MeanRiskMulti', 'LBMBP', 'PROB', 'IF-Pareto']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    x_pos = np.arange(len(plot_cols))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(plot_cols)))

    ax.bar(x_pos, means[plot_cols], yerr=stds[plot_cols], align='center', alpha=0.9, ecolor='black', capsize=10, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel(r'Avg. Decision Latency ($\mu$s)')
    ax.set_xticks(x_pos); ax.set_xticklabels(plot_cols)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    for i, v in enumerate(means[plot_cols]):
        ax.text(i, v + stds[plot_cols].iloc[i] + (v * 0.05), f"{v:.1f}", ha='center', va='bottom', fontweight='bold')

    info_text = (f"IF-Inference Cost: {means['IF-Inference']:.2f} $\mu$s\n(Decoupled Phase 1)")
    ax.text(0.95, 0.95, info_text, transform=ax.transAxes, fontsize=14, va='top', ha='right', bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.9))

    plt.tight_layout()
    filename_prefix = f"{profile_name}_" if profile_name else ""
    plt.savefig(os.path.join(output_dir, f"{filename_prefix}latency_comparison_plot.pdf"), bbox_inches='tight', dpi=300)
    plt.close()

def run_comparison_simulation(task_pool):
    all_results = []
    for r in tqdm(range(R_ROUNDS), desc="Simulation Progress", unit="round"):
        tasks_for_this_round = task_pool.sample(n=M_TASKS, replace=True)
        schedulers = {name: initialize_servers(N_SERVERS) for name in ENABLED_SCHEDULERS_LIST}
        lbmbp_global_load = {'cpu': 0.0, 'mem': 0.0}

        for i in range(M_TASKS):
            task = tasks_for_this_round.iloc[i]
            if 'P2C' in schedulers: schedule_p2c(task, schedulers['P2C'])
            if 'MeanRiskMulti' in schedulers: schedule_meanrisk_multi(task, schedulers['MeanRiskMulti'])
            if 'IF_Pareto' in schedulers: schedule_if_pareto(task, schedulers['IF_Pareto'])
            if 'IF_SafeOnly' in schedulers: schedule_if_pareto_safe_only(task, schedulers['IF_SafeOnly'])
            if 'IF_RiskyOnly' in schedulers: schedule_if_pareto_risky_only(task, schedulers['IF_RiskyOnly'])
            if 'LBMBP' in schedulers: schedule_lbmbp(task, schedulers['LBMBP'], lbmbp_global_load)
            if 'PROB' in schedulers: schedule_prob(task, schedulers['PROB'])

        for name, servers in schedulers.items():
            for s in servers:
                all_results.append({'round': r, 'scheduler': name, 'server_id': s['id'], 'cpu_load': s['total_cpu_mean'], 'mem_load': s['total_mem_mean'], 'task_count': s['task_count']})
    return pd.DataFrame(all_results)

def generate_comparison_plots(results_df, output_path_base, profile_name):
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 24, 'axes.labelsize': 26, 
        'xtick.labelsize': 22, 'ytick.labelsize': 22, 'legend.fontsize': 20, 
        'xtick.direction': 'out', 'ytick.direction': 'out', 
        'axes.linewidth': 1.5, 'axes.edgecolor': 'black', 
        'xtick.major.size': 5, 'ytick.major.size': 5,
        'xtick.major.width': 1.5, 'ytick.major.width': 1.5
    })

    box_props = {'linewidth': 1.5}; line_props = {'linewidth': 2}
    cdf_props = {'linewidth': 4}; scatter_props = {'s': 80, 'rasterized': True}

    p95_cpu = results_df.groupby(['round', 'scheduler'])['cpu_load'].quantile(0.95)
    p95_mem = results_df.groupby(['round', 'scheduler'])['mem_load'].quantile(0.95)

    def calculate_lid(x): return np.std(x) / np.mean(x) if np.mean(x) != 0 else 0

    lid_cpu = results_df.groupby(['round', 'scheduler'])['cpu_load'].apply(calculate_lid)
    lid_mem = results_df.groupby(['round', 'scheduler'])['mem_load'].apply(calculate_lid)

    metrics_df = pd.DataFrame({'P95_CPU': p95_cpu, 'P95_MEM': p95_mem, 'LID_CPU': lid_cpu, 'LID_MEM': lid_mem}).reset_index()

    for target in ['IF_Pareto', 'IF_SafeOnly', 'IF_RiskyOnly']:
        metrics_df['scheduler'] = metrics_df['scheduler'].str.replace(target, target.replace('_', '-'))

    scaler = MinMaxScaler()
    metrics_df[['P95_CPU', 'P95_MEM', 'LID_CPU', 'LID_MEM']] = scaler.fit_transform(metrics_df[['P95_CPU', 'P95_MEM', 'LID_CPU', 'LID_MEM']])

    scheduler_order = [s.replace('_', '-') for s in ENABLED_SCHEDULERS_LIST]

    try:
        # Boxplots
        fig, axes = plt.subplots(1, 2, figsize=(28, 7))
        sns.boxplot(ax=axes[0], data=metrics_df, x='scheduler', y='P95_CPU', order=scheduler_order, boxprops=box_props, whiskerprops=line_props, capprops=line_props, medianprops=line_props)
        axes[0].set_ylabel('P95 Server CPU Load (Normalized)'); axes[0].tick_params(axis='x', rotation=45); axes[0].set_xlabel('')
        sns.boxplot(ax=axes[1], data=metrics_df, x='scheduler', y='P95_MEM', order=scheduler_order, boxprops=box_props, whiskerprops=line_props, capprops=line_props, medianprops=line_props)
        axes[1].set_ylabel('P95 Server MEM Load (Normalized)'); axes[1].tick_params(axis='x', rotation=45); axes[1].set_xlabel('')
        plt.tight_layout(); plt.savefig(f"{output_path_base}_p95_boxplots.pdf", bbox_inches='tight', dpi=300); plt.close()

        fig, axes = plt.subplots(1, 2, figsize=(28, 7))
        sns.boxplot(ax=axes[0], data=metrics_df, x='scheduler', y='LID_CPU', order=scheduler_order, boxprops=box_props, whiskerprops=line_props, capprops=line_props, medianprops=line_props)
        axes[0].set_ylabel('CPU LID (Normalized)'); axes[0].tick_params(axis='x', rotation=45); axes[0].set_xlabel('')
        sns.boxplot(ax=axes[1], data=metrics_df, x='scheduler', y='LID_MEM', order=scheduler_order, boxprops=box_props, whiskerprops=line_props, capprops=line_props, medianprops=line_props)
        axes[1].set_ylabel('MEM LID (Normalized)'); axes[1].tick_params(axis='x', rotation=45); axes[1].set_xlabel('')
        plt.tight_layout(); plt.savefig(f"{output_path_base}_lid_boxplots.pdf", bbox_inches='tight', dpi=300); plt.close()

        # Scatter Plots
        scatter_df = results_df.copy()
        for target in ['IF_Pareto', 'IF_SafeOnly', 'IF_RiskyOnly']:
            scatter_df['scheduler'] = scatter_df['scheduler'].str.replace(target, target.replace('_', '-'))
        scatter_df[['cpu_load', 'mem_load']] = MinMaxScaler().fit_transform(scatter_df[['cpu_load', 'mem_load']])

        g = sns.FacetGrid(scatter_df, col="scheduler", hue="scheduler", col_order=scheduler_order, height=8, aspect=1)
        g.fig.set_size_inches(len(scheduler_order) * 7, 8)
        g.map(sns.scatterplot, "cpu_load", "mem_load", alpha=0.7, **scatter_props)
        g.set_axis_labels("Load (CPU) (Normalized)", "Load (MEM) (Normalized)")
        g.set_titles("{col_name}")
        for ax in g.axes.flat:
            ax.set_frame_on(True)
            for spine in ax.spines.values():
                spine.set_visible(True); spine.set_linewidth(2.0); spine.set_edgecolor('black')
        plt.tight_layout(); plt.savefig(f"{output_path_base}_pareto_scatter.pdf", bbox_inches='tight', dpi=300); plt.close()

        # CDF Plots
        fig, axes = plt.subplots(1, 2, figsize=(20, 7))
        for s in scheduler_order: sns.ecdfplot(ax=axes[0], data=metrics_df[metrics_df['scheduler'] == s], x='P95_CPU', label=s, **cdf_props)
        axes[0].set_xlabel('P95 CPU Load (Normalized)'); axes[0].set_ylabel('Cumulative Probability'); axes[0].legend(loc='lower right', fontsize='small'); axes[0].grid(True)
        for s in scheduler_order: sns.ecdfplot(ax=axes[1], data=metrics_df[metrics_df['scheduler'] == s], x='P95_MEM', label=s, **cdf_props)
        axes[1].set_xlabel('P95 MEM Load (Normalized)'); axes[1].set_ylabel('Cumulative Probability'); axes[1].legend(loc='lower right', fontsize='small'); axes[1].grid(True)
        plt.tight_layout(); plt.savefig(f"{output_path_base}_cdf_plots.pdf", bbox_inches='tight', dpi=300); plt.close()

    except Exception as e:
        logging.error(f"Plotting Error: {e}")

def run_sensitivity_analysis(raw_df, output_dir, profile_name):
    """Executes sensitivity tests for Contamination Factor and Z-score threshold."""
    sns.set_theme(style="ticks")
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 24, 'axes.labelsize': 20,
        'xtick.labelsize': 24, 'ytick.labelsize': 24, 'legend.fontsize': 20,
        'xtick.direction': 'out', 'ytick.direction': 'out',
        'axes.linewidth': 3.0, 'axes.edgecolor': 'black',
        'xtick.major.size': 10, 'ytick.major.size': 10,
        'xtick.major.width': 3, 'ytick.major.width': 3
    })

    cont_levels = np.linspace(0.01, 0.20, 10)
    exp1_results = []
    for cont in tqdm(cont_levels, desc="Sensitivity Exp 1"):
        df_labeled, _ = apply_isolation_forest(raw_df, cont)
        for _ in range(N_ROUNDS_SENSITIVITY):
            tasks = df_labeled.sample(n=M_TASKS, replace=True)
            servers = initialize_servers(N_SERVERS)
            for _, task in tasks.iterrows(): schedule_if_pareto(task, servers, z_score=Z_SCORE_P95)
            loads = [s['total_cpu_mean'] for s in servers]
            exp1_results.append({'Contamination': cont, 'P95 Load': np.percentile(loads, 95), 'LID': np.std(loads) / np.mean(loads) if np.mean(loads) > 0 else 0})

    df_exp1 = pd.DataFrame(exp1_results)
    df_exp1[['P95 Load', 'LID']] = MinMaxScaler().fit_transform(df_exp1[['P95 Load', 'LID']])

    fig, ax1 = plt.subplots(figsize=(12, 8))
    sns.lineplot(data=df_exp1, x='Contamination', y='P95 Load', ax=ax1, marker='o', color='tab:blue', linewidth=4, markersize=12, errorbar='sd', err_style='bars', err_kws={'capsize': 5})
    ax1.set_xlabel('Contamination Factor'); ax1.set_ylabel('P95 CPU Load (Normalized)', color='tab:blue'); ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    sns.lineplot(data=df_exp1, x='Contamination', y='LID', ax=ax2, marker='s', color='tab:red', linewidth=4, markersize=12, linestyle='--', errorbar='sd', err_style='bars', err_kws={'capsize': 5})
    ax2.set_ylabel('LID (Normalized)', color='tab:red'); ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.tight_layout(); plt.savefig(os.path.join(output_dir, f"{profile_name}_exp1_contamination.pdf"), bbox_inches='tight', dpi=300); plt.close()

    risk_levels = {'P80': norm.ppf(0.80), 'P85': norm.ppf(0.85), 'P90': norm.ppf(0.90), 'P95': norm.ppf(0.95), 'P99': norm.ppf(0.99)}
    df_labeled_fixed, _ = apply_isolation_forest(raw_df, IF_CONTAMINATION)
    exp2_results = []
    
    for label, z_val in tqdm(risk_levels.items(), desc="Sensitivity Exp 2"):
        for _ in range(N_ROUNDS_SENSITIVITY):
            tasks = df_labeled_fixed.sample(n=M_TASKS, replace=True)
            servers = initialize_servers(N_SERVERS)
            for _, task in tasks.iterrows(): schedule_if_pareto(task, servers, z_score=z_val)
            loads = [s['total_cpu_mean'] for s in servers]
            exp2_results.append({'Risk Level': label, 'Metric': 'P95 Load', 'Value': np.percentile(loads, 95)})
            exp2_results.append({'Risk Level': label, 'Metric': 'LID', 'Value': np.std(loads) / np.mean(loads) if np.mean(loads) > 0 else 0})

    df_exp2 = pd.DataFrame(exp2_results)
    for m in ['P95 Load', 'LID']:
        mask = df_exp2['Metric'] == m
        if mask.any(): df_exp2.loc[mask, ['Value']] = MinMaxScaler().fit_transform(df_exp2.loc[mask, ['Value']])

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    sns.barplot(data=df_exp2[df_exp2['Metric']=='P95 Load'], x='Risk Level', y='Value', ax=axes[0], hue='Risk Level', legend=False, palette='viridis', edgecolor='black', linewidth=2, errorbar='sd', capsize=0.1)
    axes[0].set_ylabel('P95 CPU Load (Normalized)'); axes[0].set_xlabel('Risk Threshold (Z-score)')
    sns.barplot(data=df_exp2[df_exp2['Metric']=='LID'], x='Risk Level', y='Value', ax=axes[1], hue='Risk Level', legend=False, palette='magma', edgecolor='black', linewidth=2, errorbar='sd', capsize=0.1)
    axes[1].set_ylabel('LID (Normalized)'); axes[1].set_xlabel('Risk Threshold (Z-score)')

    plt.tight_layout(); plt.savefig(os.path.join(output_dir, f"{profile_name}_exp2_zscore.pdf"), bbox_inches='tight', dpi=300); plt.close()

def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir_name = f"sim_run_{SCRIPT_VERSION}_{timestamp}"
    run_output_dir_path = os.path.join(BASE_OUTPUT_DIR, run_output_dir_name)

    try:
        os.makedirs(run_output_dir_path, exist_ok=True)
    except Exception as e:
        print(f"Critical Error: Cannot create output directory {e}")
        return

    log_file_path = os.path.join(run_output_dir_path, f"sim_run.log")
    logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()])

    logging.info("=== IF-Pareto Simulator Initiated ===")
    params_to_log = {"VERSION": SCRIPT_VERSION, "SCHEDULERS": ENABLED_SCHEDULERS_LIST, "WORKLOADS": SELECTED_WORKLOADS}
    for k, v in params_to_log.items(): logging.info(f"{k}: {v}")

    target_workloads = {k: v for k, v in ALL_WORKLOAD_PROFILES.items() if k in SELECTED_WORKLOADS}

    for profile_name, filepath in target_workloads.items():
        raw_data = load_raw_data(filepath)
        if raw_data is None: continue

        output_path_base = os.path.join(run_output_dir_path, f"{profile_name}")
        labeled_data, if_model = apply_isolation_forest(raw_data, IF_CONTAMINATION)

        if RUN_COMPARISON:
            logging.info(f"\n--- Comparison Simulation: {profile_name} ---")
            results_df = run_comparison_simulation(labeled_data)
            results_df.to_csv(f"{output_path_base}_results.csv", index=False)
            generate_comparison_plots(results_df, output_path_base, profile_name)

        if RUN_LATENCY_TEST:
            df_latency = run_latency_benchmark(labeled_data, if_model, run_output_dir_path, profile_name=profile_name)
            plot_latency_comparison(df_latency, run_output_dir_path, profile_name=profile_name)

        if RUN_SENSITIVITY_ANALYSIS:
            logging.info(f"\n--- Sensitivity Analysis: {profile_name} ---")
            run_sensitivity_analysis(raw_data, run_output_dir_path, profile_name)

if __name__ == "__main__":
    main()