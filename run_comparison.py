
import subprocess
import time
import os
import signal
import sys

# Protocol Configs
# (Protocol Name, Benchmark Protocol Arg, Pickle File, Port)
CONFIGS = [
    ("Bottom-Up SOMAP", "bottom_up", "bu_16k_m1.pkl", 10010),
    ("Top-Down SOMAP", "top_down", "td_16k_m1.pkl", 10011),
    ("Baseline B+ OMAP", "baseline", "base_16k_m1.pkl", 10012),
]

COMMON_ARGS = [
    "--num-data", "16383",
    "--cache-size", "1023",
    "--value-size", "64",
    "--ops", "100",
    "--order", "4",
    "--server-ip", "localhost",
    "--mode", "mix"
]

def run_server(port):
    print(f"--- Starting Server on Port {port} ---")
    # Start server in background
    cmd = [sys.executable, "demo/server.py", "--port", str(port)]
    process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2) # Wait for startup
    return process

def run_benchmark(protocol, load_file, port):
    print(f"--- Running Benchmark: {protocol} ---")
    cmd = [
        sys.executable, "scripts/benchmark_remote.py",
        "--protocol", protocol,
        "--port", str(port),
        "--load-storage", load_file
    ] + COMMON_ARGS
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()
    
    if result.returncode != 0:
        print(f"Error running {protocol}:")
        print(result.stderr)
        return None
    
    print(f"Benchmark finished in {end_time - start_time:.2f}s")
    return result.stdout

def parse_results(output):
    # Just print the summary table or specific lines from output
    lines = output.split('\n')
    summary_started = False
    
    relevant_lines = []
    
    # Simple parsing: Find lines with "rounds" or the summary table at the end
    # The benchmark script prints a "COMPARISON SUMMARY" if len > 1, but we run 1 by 1.
    # It prints 'print_results' output: 
    # e.g. "Protocol: X, Rounds: Y, Time: Z..."
    
    for line in lines:
        if "Comparison:" in line or "rounds" in line or "Time/op" in line:
            relevant_lines.append(line)
        # Also catch the nicely formatted dict print if any
        if line.strip().startswith("{'rounds'"):
            relevant_lines.append(line)
            
    return "\n".join(relevant_lines)

def main():
    results_log = []
    
    for name, proto, pkl, port in CONFIGS:
        server_proc = run_server(port)
        try:
            output = run_benchmark(proto, pkl, port)
            if output:
                # Capture the full output to a file for review
                with open(f"bench_{proto}.log", "w") as f:
                    f.write(output)
                
                # Extract summary (manual check is better but let's try)
                # We can just rely on the log files for the user.
                print(f"Output saved to bench_{proto}.log")
                results_log.append(f"=== {name} ===\n{output[-500:]}\n") 
        finally:
            server_proc.terminate()
            server_proc.wait()
            time.sleep(1) # Cooldown

    print("\n\nALL BENCHMARKS COMPLETE. TAILS OF LOGS:")
    for log in results_log:
        print(log)

if __name__ == "__main__":
    main()
