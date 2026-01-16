
import subprocess
import time
import os
import sys
import argparse
import signal

"""
Full Comparison Demo Script
---------------------------
This script demonstrates the complete workflow for benchmarking three OMAP protocols:
1. Bottom-Up SOMAP
2. Top-Down SOMAP
3. Baseline B+ OMAP

Workflow:
1. [Pre-build Phase]: Generates server storage files (.pkl) for each protocol.
   This simulates the offline setup phase.
   
2. [Benchmark Phase]: 
   - Starts a server instance.
   - Runs the client benchmark which loads the pre-built storage.
   - Measures performance (Time, Bandwidth, Rounds).
   - Shuts down the server.

3. [Summary]: Prints a comparison table.
"""

# Configuration
DEFAULT_N = 16383  # 2^14 - 1
DEFAULT_CACHE = 1023 # 2^10 - 1
DEFAULT_VALUE_SIZE = 64
DEFAULT_OPS = 100
DEFAULT_ORDER = 4

def get_python_cmd():
    return sys.executable

def run_command(cmd, description):
    print(f"[*] {description}...")
    # print(f"    Command: {' '.join(cmd)}")
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    [FAILED] Error: {result.stderr.strip()}")
        sys.exit(1)
    print(f"    [Done] Time: {time.time() - start:.2f}s")
    return result.stdout

def extract_metric(output, metric_name):
    """Simple parser to extract values from benchmark output logs."""
    for line in output.split('\n'):
        if metric_name in line:
            # Example: "  Total Time:     4.11s (41.12ms/op)"
            return line.strip()
    return "N/A"

def extract_bandwidth(output):
    sent = "0"
    recv = "0"
    for line in output.split('\n'):
        if "Bandwidth Sent:" in line:
            sent = line.split(':')[1].strip().split(' ')[0] # 56481.99
        if "Bandwidth Recv:" in line:
            recv = line.split(':')[1].strip().split(' ')[0]
    try:
        total_kb = float(sent) + float(recv)
        return f"{total_kb/1024:.2f} MB"
    except:
        return "N/A"

def main():
    parser = argparse.ArgumentParser(description="Run Full Comparison Demo")
    parser.add_argument("--num-data", type=int, default=DEFAULT_N, help="Number of data items (N)")
    parser.add_argument("--cache-size", type=int, default=DEFAULT_CACHE, help="Cache size (C)")
    parser.add_argument("--ops", type=int, default=DEFAULT_OPS, help="Number of operations to run")
    parser.add_argument("--keep-files", action="store_true", help="Don't delete generated .pkl files")
    args = parser.parse_args()

    print("="*60)
    print(f"OMAP PROTOCOL COMPARISON DEMO")
    print(f"N={args.num_data}, Cache={args.cache_size}, Ops={args.ops}, ValueSize={DEFAULT_VALUE_SIZE}B")
    print("="*60)

    # Define protocols
    protocols = [
        # (Display Name, Protocol Arg, Filename, Port)
        ("Bottom-Up SOMAP", "bottom_up", f"demo_bu_{args.num_data}.pkl", 20001),
        ("Top-Down SOMAP", "top_down", f"demo_td_{args.num_data}.pkl", 20002),
        ("Baseline B+ Tree", "baseline", f"demo_base_{args.num_data}.pkl", 20003),
    ]
    
    python_cmd = get_python_cmd()
    script_dir = os.path.dirname(os.path.abspath(__file__)) # location of this script
    root_dir = os.path.dirname(script_dir) # parent (DAORAM root)
    
    results = []

    # 1. Pre-build Phase
    print("\n--- Phase 1: Pre-building Server Storage ---")
    for name, proto, filename, port in protocols:
        cmd = [
            python_cmd, os.path.join(root_dir, "scripts", "prebuild_server.py"),
            "--protocol", proto,
            "--num-data", str(args.num_data),
            "--value-size", str(DEFAULT_VALUE_SIZE),
            "--output", filename,
            "--order", str(DEFAULT_ORDER)
        ]
        if proto != "baseline":
            cmd.extend(["--cache-size", str(args.cache_size)])
            
        run_command(cmd, f"Building storage for {name}")

    # 2. Benchmark Phase
    print("\n--- Phase 2: Running Benchmarks ---")
    for name, proto, filename, port in protocols:
        print(f"\n--> Protocol: {name}")
        
        # Start Server
        server_cmd = [python_cmd, os.path.join(root_dir, "demo", "server.py"), "--port", str(port)]
        # Start server in background
        server_proc = subprocess.Popen(server_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        try:
            time.sleep(1.5) # Wait for server startup
            
            # Run Client
            client_cmd = [
                python_cmd, os.path.join(root_dir, "scripts", "benchmark_remote.py"),
                "--protocol", proto,
                "--server-ip", "localhost",
                "--port", str(port),
                "--load-storage", filename,
                "--num-data", str(args.num_data),
                "--value-size", str(DEFAULT_VALUE_SIZE),
                "--ops", str(args.ops),
                "--order", str(DEFAULT_ORDER),
                "--mode", "mix"
            ]
            if proto != "baseline":
                client_cmd.extend(["--cache-size", str(args.cache_size)])
            
            stdout = run_command(client_cmd, "Running client benchmark")
            
            # Parse Results
            time_str = extract_metric(stdout, "Total Time:")
            rounds_str = extract_metric(stdout, "Total Rounds:")
            bandwidth_str = extract_bandwidth(stdout)
            
            # Clean up the strings to just get the value
            time_cleaned = time_str.replace("Total Time:", "").strip()
            rounds_cleaned = rounds_str.replace("Total Rounds:", "").strip()

            results.append({
                "Protocol": name,
                "Time": time_cleaned.split('(')[0].strip() if '(' in time_cleaned else time_cleaned,
                "Latency": time_cleaned.split('(')[1].strip(')') if '(' in time_cleaned else "N/A",
                "Rounds": rounds_cleaned.split('(')[0].strip() if '(' in rounds_cleaned else rounds_cleaned,
                "Bandwidth": bandwidth_str
            })
            
        finally:
            # Kill server
            os.kill(server_proc.pid, signal.SIGTERM)
            server_proc.wait()
            time.sleep(0.5)

    # 3. Cleanup Phase
    if not args.keep_files:
        print("\n--- Phase 3: Cleanup ---")
        for _, _, filename, _ in protocols:
            if os.path.exists(filename):
                os.remove(filename)
                print(f"Removed {filename}")
    else:
        print("\n--- Skipping Cleanup (Files Kept) ---")

    # 4. Final Report
    print("\n" + "="*85)
    print(f"{'PROTOCOL':<20} | {'TOTAL TIME':<12} | {'LATENCY (AVG)':<15} | {'ROUNDS':<10} | {'BANDWIDTH':<10}")
    print("-" * 85)
    for res in results:
        print(f"{res['Protocol']:<20} | {res['Time']:<12} | {res['Latency']:<15} | {res['Rounds']:<10} | {res['Bandwidth']:<10}")
    print("="*85)

if __name__ == "__main__":
    main()
