import subprocess
import time
import os
import sys
import signal
import json
from math import pow

"""
Parameter Sweep Benchmark (Real Case, Server Version)
-----------------------------------------------------
- num_data = 2^24-1
- cache_size: [2^3-1, 2^4-1, ..., 2^23-1]
- order: 4, 8, 16
- key_size: 16 bytes, value_size: 4096 bytes
- ops: 100
- latency: 30ms, 50ms, 80ms
- 实时写入 sweep_results_real.txt，断点续跑
- 详细进度提示
"""

NUM_DATA = int(pow(2,14)) - 1
KEY_SIZE = 16
VALUE_SIZE = 4096
OPS = 100
ORDERS = [4, 8, 16]
CACHE_SIZES = [int(pow(2,i))-1 for i in range(3,14)]
LATENCIES = [30, 50, 80]
RESULT_FILE = "sweep_results_real.txt"

python_cmd = sys.executable
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)

# 断点续跑：已完成的case
completed = set()
if os.path.exists(RESULT_FILE):
    with open(RESULT_FILE) as f:
        for line in f:
            if line.startswith("#CASE:"):
                completed.add(line.strip())

def run_command(cmd, desc):
    print(f"    [*] {desc}...", flush=True)
    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"      [FAILED] {result.stderr.strip()}")
            return None
        print(f"      [Done] {time.time()-start:.2f}s", flush=True)
        return result.stdout
    except Exception as e:
        print(f"      [Error] {e}")
        return None

def parse_benchmark_output(output):
    metrics = {}
    if not output: return metrics
    for line in output.split('\n'):
        if "Total Time:" in line and "ms/op" in line:
            metrics["Total Time"] = line.split('(')[1].split(')')[0]
        if "- Processing:" in line:
            metrics["Calc Time"] = line.split(':')[1].split('(')[0].strip()
        if "- Commun.:" in line:
            metrics["Net Time"] = line.split(':')[1].split('(')[0].strip()
        if "Total Rounds:" in line:
            metrics["Rounds"] = line.split(':')[1].split('(')[0].strip()
        if "Peak Stash Entries:" in line:
            metrics["Mem"] = line.split(':')[1].strip() + " entries"
        if "Total Server:" in line:
            metrics["ServerStorage"] = line.split(':')[1].strip()
        if "Bandwidth Sent:" in line:
            try: metrics["Sent"] = float(line.split(':')[1].strip().split(' ')[0])
            except: pass
        if "Bandwidth Recv:" in line:
            try: metrics["Recv"] = float(line.split(':')[1].strip().split(' ')[0])
            except: pass
    if "Sent" in metrics or "Recv" in metrics:
        total = metrics.get("Sent",0) + metrics.get("Recv",0)
        metrics["BW"] = f"{total/1024:.2f} MB" if total>1024 else f"{total:.2f} KB"
    return metrics

# 主 sweep 循环
total_cases = len(CACHE_SIZES)*len(ORDERS)*len(LATENCIES)*3

# 复用 DS：bottom_up static ORAM 与 top_down 的 _Tree/_Ds 仅依赖 num_data/order
# 每个 case 仍生成独立可加载的 full 文件（包含 cache_size 相关缓存）
base_ds_files = {}      # (protocol, order) -> ds_only file
full_state_files = {}   # (protocol, order, cache_size) -> full file

# 分阶段 sweep：先 sweep latency=50ms 且 order=8 的所有 case，再 sweep 其它 case
def sweep_cases(filter_func, base_ds_files, full_state_files, completed, start_idx=0, total_cases=None):
    case_idx = start_idx
    for cache_size in CACHE_SIZES:
        for order in ORDERS:
            for latency in LATENCIES:
                for proto_name, proto in [("Bottom-Up","bottom_up"), ("Top-Down","top_down"), ("Baseline","baseline")]:
                    if not filter_func(cache_size, order, latency, proto):
                        continue
                    case_tag = f"#CASE: cache={cache_size},order={order},latency={latency},proto={proto}"
                    if case_tag in completed:
                        case_idx += 1
                        continue
                    if total_cases:
                        print(f"\n==== [{case_idx+1}/{total_cases}] {case_tag} ====")
                    else:
                        print(f"\n==== {case_tag} ====")
                    sys.stdout.flush()
                    # 1. prebuild：先生成可复用 DS（ds_only），再为每个 cache_size 生成 full
                    if proto == "baseline":
                        static_key = (proto, order)
                        if static_key not in full_state_files:
                            full_file = f"static_oram_baseline_{order}.pkl"
                            build_cmd = [python_cmd, os.path.join(root_dir, "scripts", "prebuild_server.py"),
                                "--protocol", proto,
                                "--num-data", str(NUM_DATA),
                                "--value-size", str(VALUE_SIZE),
                                "--key-size", str(KEY_SIZE),
                                "--output", full_file,
                                "--order", str(order),
                                "--target", "full"]
                            run_command(build_cmd, f"Prebuild static ORAM for Baseline (order={order})")
                            full_state_files[static_key] = full_file
                        ds_file = full_state_files[static_key]
                    else:
                        base_key = (proto, order)
                        if base_key not in base_ds_files:
                            base_file = (
                                f"static_oram_bottom_up_{order}_ds.pkl"
                                if proto == "bottom_up"
                                else f"static_serverstate_top_down_{order}_ds.pkl"
                            )
                            build_cmd = [python_cmd, os.path.join(root_dir, "scripts", "prebuild_server.py"),
                                "--protocol", proto,
                                "--num-data", str(NUM_DATA),
                                "--value-size", str(VALUE_SIZE),
                                "--key-size", str(KEY_SIZE),
                                "--output", base_file,
                                "--order", str(order),
                                "--target", "ds_only",
                                "--cache-size", str(max(CACHE_SIZES))]
                            if proto == "top_down":
                                run_command(build_cmd, f"Prebuild base DS for Top-Down (order={order})")
                            else:
                                run_command(build_cmd, f"Prebuild base DS for Bottom-Up (order={order})")
                            base_ds_files[base_key] = base_file

                        static_key = (proto, order, cache_size)
                        if static_key not in full_state_files:
                            full_file = (
                                f"static_oram_bottom_up_{order}_cache{cache_size}.pkl"
                                if proto == "bottom_up"
                                else f"static_serverstate_top_down_{order}_cache{cache_size}.pkl"
                            )
                            build_cmd = [python_cmd, os.path.join(root_dir, "scripts", "prebuild_server.py"),
                                "--protocol", proto,
                                "--num-data", str(NUM_DATA),
                                "--value-size", str(VALUE_SIZE),
                                "--key-size", str(KEY_SIZE),
                                "--output", full_file,
                                "--order", str(order),
                                "--target", "full",
                                "--cache-size", str(cache_size),
                                "--reuse-ds", base_ds_files[base_key]]
                            if proto == "top_down":
                                run_command(build_cmd, f"Prebuild full Top-Down (order={order}, cache={cache_size})")
                            else:
                                run_command(build_cmd, f"Prebuild full Bottom-Up (order={order}, cache={cache_size})")
                            full_state_files[static_key] = full_file
                        ds_file = full_state_files[static_key]
                    # 2. 启动 server
                    port = 31000 + (case_idx % 1000)
                    server_cmd = [python_cmd, os.path.join(root_dir, "demo", "server.py"), "--port", str(port)]
                    server_proc = subprocess.Popen(server_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    try:
                        time.sleep(2)
                        client_cmd = [python_cmd, os.path.join(root_dir, "scripts", "benchmark_remote.py"),
                            "--protocol", proto,
                            "--server-ip", "localhost",
                            "--port", str(port),
                            "--load-storage", ds_file,
                            "--num-data", str(NUM_DATA),
                            "--key-size", str(KEY_SIZE),
                            "--value-size", str(VALUE_SIZE),
                            "--ops", str(OPS),
                            "--order", str(order),
                            "--mode", "mix",
                            "--latency", str(latency)]
                        if proto != "baseline":
                            client_cmd += ["--cache-size", str(cache_size)]
                        out = run_command(client_cmd, f"Benchmark {proto_name}")
                        metrics = parse_benchmark_output(out)
                    finally:
                        os.kill(server_proc.pid, signal.SIGTERM)
                        server_proc.wait()
                    # 3. 结果写入
                    with open(RESULT_FILE, "a") as f:
                        f.write(f"{case_tag}\n")
                        f.write(json.dumps({
                            "cache": cache_size,
                            "order": order,
                            "latency": latency,
                            "protocol": proto,
                            "metrics": metrics
                        }, ensure_ascii=False) + "\n")
                        f.flush()
                    print(f"    [Result saved] {metrics}")
                    case_idx += 1
                    # 4. static ORAM 不删除，便于复用
                    # 5. 进度提示
                    if total_cases:
                        print(f"    [Progress] {case_idx}/{total_cases} cases finished.\n", flush=True)
                    else:
                        print(f"    [Progress] {case_idx} cases finished.\n", flush=True)
    return case_idx

# 先 sweep latency=50ms 且 order=8
print("\n[Phase 1] Sweep latency=50ms, order=8 全部 case...")
phase1_cases = len(CACHE_SIZES)*1*1*3
idx = sweep_cases(lambda c,o,l,p: o==8 and l==50, base_ds_files, full_state_files, completed, start_idx=0, total_cases=phase1_cases)

# 再 sweep 其它 case
print("\n[Phase 2] Sweep 其它 case...")
total_cases = len(CACHE_SIZES)*len(ORDERS)*len(LATENCIES)*3
sweep_cases(lambda c,o,l,p: not (o==8 and l==50), base_ds_files, full_state_files, completed, start_idx=idx, total_cases=total_cases)

print("\nAll cases finished! Results in sweep_results_real.txt")
