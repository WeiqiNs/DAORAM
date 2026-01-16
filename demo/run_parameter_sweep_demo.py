
import subprocess
import time
import os
import sys
import argparse
import signal

"""
Parameter Sweep Benchmark Demo
------------------------------
This script runs a comparative benchmark across different configurations of:
- Cache Size
- B+ Tree Order

Protocols tested:
1. Bottom-Up SOMAP
2. Top-Down SOMAP
3. Baseline B+ OMAP
"""

NUM_DATA = 16383  # 2^14 - 1
VALUE_SIZE = 64
OPS = 100

# Configurations to test
# Format: (Cache Size, Tree Order)
# User requested: Cache=[2^10, 2^11], Order=[8, 16]
CONFIGS = [
    (1024, 8),
    (1024, 16),
    (2048, 8),
    (2048, 16)
]

def get_python_cmd():
    return sys.executable

def run_command(cmd, description):
    print(f"    [*] {description}...", end="", flush=True)
    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f" [FAILED]\n    Error: {result.stderr.strip()}")
            return None
        print(f" [Done] {time.time() - start:.2f}s")
        return result.stdout
    except Exception as e:
        print(f" [Error] {e}")
        return None

def extract_metric(output, metric_name):
    if not output: return "N/A"
    for line in output.split('\n'):
        if metric_name in line:
            return line.strip()
    return "N/A"

def extract_bandwidth(output):
    if not output: return "N/A"
    sent = "0"
    recv = "0"
    for line in output.split('\n'):
        if "Bandwidth Sent:" in line:
            sent = line.split(':')[1].strip().split(' ')[0]
        if "Bandwidth Recv:" in line:
            recv = line.split(':')[1].strip().split(' ')[0]
    try:
        total_kb = float(sent) + float(recv)
        return f"{total_kb/1024:.2f} MB"
    except:
        return "N/A"

def main():
    python_cmd = get_python_cmd()
    script_dir = os.path.dirname(os.path.abspath(__file__)) 
    root_dir = os.path.dirname(script_dir)
    
    results = []
    
    print("="*80)
    print("PARAMETER SWEEP BENCHMARK")
    print(f"N={NUM_DATA}, Ops={OPS}")
    print("Configs (Cache, Order):", CONFIGS)
    print("="*80)

    # Base port to avoid conflicts if run repeatedly quickly
    base_port = 30000

    for idx, (cache, order) in enumerate(CONFIGS):
        print(f"\n>>> CONFIGURATION {idx+1}/{len(CONFIGS)}: Cache={cache}, Order={order}")
        
        current_protocols = [
            ("Bottom-Up", "bottom_up", f"sweep_bu_{cache}_{order}.pkl", base_port + 0),
            ("Top-Down", "top_down", f"sweep_td_{cache}_{order}.pkl", base_port + 1),
            ("Baseline", "baseline", f"sweep_base_{cache}_{order}.pkl", base_port + 2),
        ]
        
        base_port += 10 # Increment for next batch

        # 1. Pre-build
        print("  [Phase 1] Pre-building Storage")
        for name, proto, filename, _ in current_protocols:
            cmd = [
                python_cmd, os.path.join(root_dir, "scripts", "prebuild_server.py"),
                "--protocol", proto,
                "--num-data", str(NUM_DATA),
                "--value-size", str(VALUE_SIZE),
                "--output", filename,
                "--order", str(order)
            ]
            if proto != "baseline":
                cmd.extend(["--cache-size", str(cache)])
                
            run_command(cmd, f"Building {name}")

        # 2. Benchmark
        print("  [Phase 2] Executing Benchmarks")
        for name, proto, filename, port in current_protocols:
            # Start Server
            server_cmd = [python_cmd, os.path.join(root_dir, "demo", "server.py"), "--port", str(port)]
            server_proc = subprocess.Popen(server_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            try:
                time.sleep(1.5)
                client_cmd = [
                    python_cmd, os.path.join(root_dir, "scripts", "benchmark_remote.py"),
                    "--protocol", proto,
                    "--server-ip", "localhost",
                    "--port", str(port),
                    "--load-storage", filename,
                    "--num-data", str(NUM_DATA),
                    "--value-size", str(VALUE_SIZE),
                    "--ops", str(OPS),
                    "--order", str(order),
                    "--mode", "mix"
                ]
                if proto != "baseline":
                    client_cmd.extend(["--cache-size", str(cache)])
                
                stdout = run_command(client_cmd, f"Running {name}")
                
                # Parse
                time_str = extract_metric(stdout, "Total Time:")
                if "Total Time:" in time_str:
                     clean_time = time_str.replace("Total Time:", "").strip().split('(')[0].strip()
                     latency = time_str.split('(')[1].strip(')') if '(' in time_str else "N/A"
                else:
                     clean_time, latency = "Error", "Error"

                bw = extract_bandwidth(stdout)
                
                results.append({
                    "Config": f"C={cache}, O={order}",
                    "Protocol": name,
                    "Time": clean_time,
                    "Latency": latency,
                    "Bandwidth": bw
                })

            finally:
                os.kill(server_proc.pid, signal.SIGTERM)
                server_proc.wait()
                
            # Cleanup file immediately to save space
            if os.path.exists(filename):
                os.remove(filename)

    # 3. Summary
    print("\n" + "="*95)
    print(f"{'CONFIG':<15} | {'PROTOCOL':<12} | {'TIME':<10} | {'LATENCY':<15} | {'BANDWIDTH':<12}")
    print("-" * 95)
    for res in results:
        print(f"{res['Config']:<15} | {res['Protocol']:<12} | {res['Time']:<10} | {res['Latency']:<15} | {res['Bandwidth']:<12}")
    print("="*95)

if __name__ == "__main__":
    main()
