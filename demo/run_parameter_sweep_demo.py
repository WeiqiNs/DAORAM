
import subprocess
import time
import os
import sys
import argparse
import signal

"""
Parameter Sweep Benchmark Demo (Extended Metrics)
-------------------------------------------------
This script runs a comparative benchmark across different configurations of:
- Cache Size
- B+ Tree Order

Protocols tested:
1. Bottom-Up SOMAP
2. Top-Down SOMAP
3. Baseline B+ OMAP

Metrics collected:
- Total Latency per Op
- Calculation Time (CPU) vs Network Transfer Time
- Interaction Rounds (RTT) per Op
- Bandwidth
- Peak Client Memory (Stash Usage)
"""

NUM_DATA = 16383  # 2^14 - 1
VALUE_SIZE = 64
OPS = 100
LATENCY_MS = 0  # WAN latency simulation in ms (0 = no simulation, 50 = typical WAN)

# Configurations to test
# Format: (Cache Size, Tree Order)
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

def parse_benchmark_output(output):
    """Parse detailed metrics from benchmark_remote.py output."""
    metrics = {
        "Latency": "N/A",
        "Calc": "N/A", 
        "Net": "N/A",
        "Rounds": "N/A",
        "BW": "N/A",
        "Mem": "N/A",
        "ServerStorage": "N/A"
    }
    
    if not output: return metrics

    sent_kb = 0.0
    recv_kb = 0.0
    
    for line in output.split('\n'):
        line = line.strip()
        
        # Latency (Total Time)
        # format: Total Time:     5.23s (52.30ms/op)
        if "Total Time:" in line and "ms/op" in line:
            try:
                metrics["Latency"] = line.split('(')[1].split(')')[0]
            except: pass
            
        # Calc Time (Processing)
        # format: - Processing: 1.23ms/op (10.0%)
        if "- Processing:" in line:
            try:
                metrics["Calc"] = line.split(':')[1].split('(')[0].strip()
            except: pass
            
        # Net Time (Commun.)
        # format: - Commun.:    1.23ms/op (10.0%)
        if "- Commun.:" in line:
            try:
                metrics["Net"] = line.split(':')[1].split('(')[0].strip()
            except: pass
            
        # Rounds
        # format: Total Rounds:   1400 (14.00/op)
        if "Total Rounds:" in line:
            try:
                metrics["Rounds"] = line.split(':')[1].split('(')[0].strip()
            except: pass
            
        # Memory
        # format: Peak Stash Entries: 123
        if "Peak Stash Entries:" in line:
            try:
                metrics["Mem"] = line.split(':')[1].strip() + " entries"
            except: pass
            
        # Server Storage
        # format: Total Server:   123.45 MB
        if "Total Server:" in line:
            try:
                metrics["ServerStorage"] = line.split(':')[1].strip()
            except: pass
            
        # Bandwidth accumulation
        if "Bandwidth Sent:" in line:
            try:
                # 123.45 KB
                val = line.split(':')[1].strip().split(' ')[0]
                sent_kb = float(val)
            except: pass
        if "Bandwidth Recv:" in line:
            try:
                val = line.split(':')[1].strip().split(' ')[0]
                recv_kb = float(val)
            except: pass
            
    if sent_kb > 0 or recv_kb > 0:
        total = sent_kb + recv_kb
        if total > 1024:
            metrics["BW"] = f"{total/1024:.2f} MB"
        else:
            metrics["BW"] = f"{total:.2f} KB"
            
    return metrics

def main():
    python_cmd = get_python_cmd()
    script_dir = os.path.dirname(os.path.abspath(__file__)) 
    root_dir = os.path.dirname(script_dir)
    
    results = []
    
    print("="*120)
    print("PARAMETER SWEEP BENCHMARK (EXTENDED METRICS)")
    print(f"N={NUM_DATA}, Ops={OPS}, Latency={LATENCY_MS}ms")
    print("Configs (Cache, Order):", CONFIGS)
    print("="*120)

    # Base port to avoid conflicts if run repeatedly quickly
    base_port = 30000

    # Track generated DS files: key=(protocol, order) -> filename
    generated_ds_files = {}

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
        
        # We might modify the filename for Baseline if reusing
        updated_protocols = []

        for name, proto, filename, port in current_protocols:
            # Check for reuse source
            reuse_src = None
            if proto in ["bottom_up", "top_down"]:
                ds_key = (proto, order)
                if ds_key in generated_ds_files:
                    reuse_src = generated_ds_files[ds_key]
            
            # Special case for Baseline: Reuse logic
            if proto == "baseline":
                ds_key = (proto, order)
                if ds_key in generated_ds_files:
                    print(f"    [*] Skipping build for {name} (Reusing {generated_ds_files[ds_key]})")
                    updated_protocols.append((name, proto, generated_ds_files[ds_key], port))
                    continue
            
            # Construct build command
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
                if reuse_src:
                    cmd.extend(["--reuse-ds", reuse_src])
                
            run_command(cmd, f"Building {name}" + (f" (Reusing DS from {reuse_src})" if reuse_src else ""))
            
            # Register this file as a future source if it's the first for this Order
            if (proto, order) not in generated_ds_files:
                 generated_ds_files[(proto, order)] = filename
            
            updated_protocols.append((name, proto, filename, port))

        current_protocols = updated_protocols

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
                    "--mode", "mix",
                    "--latency", str(LATENCY_MS)
                ]
                if proto != "baseline":
                    client_cmd.extend(["--cache-size", str(cache)])
                
                stdout = run_command(client_cmd, f"Running {name}")
                
                # Parse
                metrics = parse_benchmark_output(stdout)
                
                results.append({
                    "Config": f"C={cache}, O={order}",
                    "Protocol": name,
                    "Total Time": metrics["Latency"],
                    "Calc Time": metrics["Calc"],
                    "Net Time": metrics["Net"],
                    "Rounds": metrics["Rounds"],
                    "Bandwidth": metrics["BW"],
                    "Memory": metrics["Mem"],
                    "ServerStorage": metrics["ServerStorage"]
                })

            finally:
                os.kill(server_proc.pid, signal.SIGTERM)
                server_proc.wait()
                
            # Cleanup file immediately to save space
            # Optimization: Don't delete files that are reused as Base DS
            is_base_ds = filename in generated_ds_files.values()
            if os.path.exists(filename) and not is_base_ds:
                os.remove(filename)

    # 3. Report
    print("\n" + "="*140)
    print("FINAL RESULTS SUMMARY")
    print("="*140)
    
    # Headers
    headers = [
        "Config", "Protocol", "Total/op", "Calc/op", "Net/op", 
        "Rounds", "Bandwidth", "Peak Mem", "Server"
    ]
    
    # Format string (fixed width)
    row_fmt = "{:<16}  {:<12}  {:<12}  {:<12}  {:<12}  {:<8}  {:<12}  {:<14}  {:<12}"
    
    print(row_fmt.format(*headers))
    print("-" * 140)
    
    for r in results:
        print(row_fmt.format(
            r["Config"],
            r["Protocol"],
            r["Total Time"],
            r["Calc Time"],
            r["Net Time"],
            r["Rounds"],
            r["Bandwidth"],
            r["Memory"],
            r["ServerStorage"]
        ))
    
    print("-" * 140)
    print("\nNote: 'Calc/op' is CPU processing time, 'Net/op' is Network transmission time.")
    print("Peak Mem is client-side storage (stash + local + pending ops). Server is total server storage.")

    # Save to file
    with open("sweep_results_extended.txt", "w") as f:
        f.write(f"Parameter Sweep Results - {time.ctime()}\n")
        f.write("-" * 140 + "\n")
        f.write(row_fmt.format(*headers) + "\n")
        f.write("-" * 140 + "\n")
        for r in results:
            f.write(row_fmt.format(
                r["Config"],
                r["Protocol"],
                r["Total Time"],
                r["Calc Time"],
                r["Net Time"],
                r["Rounds"],
                r["Bandwidth"],
                r["Memory"],
                r["ServerStorage"]
            ) + "\n")

if __name__ == "__main__":
    main()
