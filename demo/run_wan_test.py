
import subprocess
import time
import os
import sys
import argparse
import signal

"""
WAN Simulation Benchmark
------------------------
Simulates 50ms One-Way Latency (100ms RTT).
Configuration: N=16383, Cache=1024, Order=8.
"""

NUM_DATA = 16383
VALUE_SIZE = 64
OPS = 50 
LATENCY = 50.0  # ms (One-way)

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
    """Parse detailed metrics."""
    metrics = {
        "Latency": "N/A",
        "Calc": "N/A", 
        "Net": "N/A",
        "Rounds": "N/A",
        "BW": "N/A",
        "Mem": "N/A"
    }
    
    if not output: return metrics

    sent_kb = 0.0
    recv_kb = 0.0
    
    for line in output.split('\n'):
        line = line.strip()
        if "Total Time:" in line and "ms/op" in line:
            try: metrics["Latency"] = line.split('(')[1].split(')')[0]
            except: pass
        if "- Processing:" in line:
            try: metrics["Calc"] = line.split(':')[1].split('(')[0].strip()
            except: pass
        if "- Commun.:" in line:
            try: metrics["Net"] = line.split(':')[1].split('(')[0].strip()
            except: pass
        if "Total Rounds:" in line:
            try: metrics["Rounds"] = line.split(':')[1].split('(')[0].strip()
            except: pass
        if "Peak Stash Entries:" in line:
            try: metrics["Mem"] = line.split(':')[1].strip()
            except: pass
        if "Bandwidth Sent:" in line:
            try: sent_kb = float(line.split(':')[1].strip().split(' ')[0])
            except: pass
        if "Bandwidth Recv:" in line:
            try: recv_kb = float(line.split(':')[1].strip().split(' ')[0])
            except: pass
            
    if sent_kb > 0 or recv_kb > 0:
        total = sent_kb + recv_kb
        metrics["BW"] = f"{total/1024:.2f} MB" if total > 1024 else f"{total:.2f} KB"
            
    return metrics

def main():
    python_cmd = get_python_cmd()
    script_dir = os.path.dirname(os.path.abspath(__file__)) 
    root_dir = os.path.dirname(script_dir)
    
    # Files from previous run
    files = {
        "Bottom-Up": "sweep_bu_1024_8.pkl",
        "Top-Down": "sweep_td_1024_8.pkl",
        "Baseline": "sweep_base_1024_8.pkl"
    }
    
    protocols = [
        ("Bottom-Up", "bottom_up", 40000),
        ("Top-Down", "top_down", 40001),
        ("Baseline", "baseline", 40002)
    ]
    
    results = []
    
    print("="*60)
    print(f"WAN SIMULATION (Latency={LATENCY}ms)")
    print(f"N={NUM_DATA}, Ops={OPS}, Cache=1024, Order=8")
    print("="*60)

    for name, proto, port in protocols:
        filename = files[name]
        if not os.path.exists(filename):
            print(f"Error: {filename} not found. Run parameter sweep first.")
            continue
            
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
                "--order", "8",
                "--cache-size", "1024",
                "--latency", str(LATENCY)
            ]
            
            stdout = run_command(client_cmd, f"Running {name}")
            metrics = parse_benchmark_output(stdout)
            
            results.append({
                "Protocol": name,
                "Total": metrics["Latency"],
                "Calc": metrics["Calc"],
                "Net": metrics["Net"],
                "Rounds": metrics["Rounds"],
                "BW": metrics["BW"]
            })

        finally:
            os.kill(server_proc.pid, signal.SIGTERM)
            server_proc.wait()

    print("\n" + "="*80)
    print("WAN RESULTS (50ms One-Way / 100ms RTT)")
    print("="*80)
    print(f"{'Protocol':<15} {'Total/op':<12} {'Calc/op':<12} {'Net/op':<12} {'Rounds':<8} {'BW':<10}")
    print("-" * 80)
    for r in results:
        print(f"{r['Protocol']:<15} {r['Total']:<12} {r['Calc']:<12} {r['Net']:<12} {r['Rounds']:<8} {r['BW']:<10}")

if __name__ == "__main__":
    main()
