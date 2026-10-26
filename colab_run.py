"""One-shot Colab runner: GPU install -> tests -> benchmarks -> training ->
renders -> report -> README update -> git push.

In a fresh Colab notebook (Runtime -> Change runtime type -> T4 GPU):

    !git clone -q https://github.com/shiroonigami23-ui/jax-swarm-es /content/jax-swarm-es
    %run /content/jax-swarm-es/colab_run.py            # full run
    %run /content/jax-swarm-es/colab_run.py --sweep    # hyperparameter sweep only

The push needs a GitHub token stored in Colab Secrets (key icon) as
GITHUB_TOKEN or GitHub, with "Notebook access" switched on. A fine-grained
token scoped to this repo with Contents: read and write is enough.

Every heavy step runs as its own `python -m ...` subprocess, so the CUDA
build of JAX installed here is picked up without restarting the runtime.
"""
import csv
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = "shiroonigami23-ui/jax-swarm-es"
WORKDIR = Path("/content/jax-swarm-es")
OUT = Path("results/gpu")      # relative to the repo root

DRONE_ARGS = ["--env", "drone", "--pop", "2048", "--n-eval", "5", "--gens", "500"]
PENDULUM_ARGS = ["--env", "pendulum", "--pop", "2048", "--n-eval", "5",
                 "--gens", "400", "--sigma", "0.1"]
BENCH_SIZES = ["1", "64", "1024", "8192", "32768", "65536"]
RENDER_EPISODES = "3"
SWEEP_ARGS = ["--env", "pendulum", "--pop", "1024", "--n-eval", "5", "--gens", "300",
              "--lrs", "0.005", "0.01", "0.02", "0.05", "--sigmas", "0.03", "0.1",
              "--lr-decays", "1.0", "0.995"]
TUNED_GENS = {"pendulum": "400", "drone": "500"}
ALLOW_CPU = False   # set True to run anyway without a GPU (slow)


# ---------------------------------------------------------------- helpers

def banner(text):
    print(f"\n{'=' * 70}\n{text}\n{'=' * 70}", flush=True)


def run(cmd, cwd=None, check=True):
    """Run a command, streaming its output live. Returns (returncode, seconds)."""
    print("$", " ".join(cmd), flush=True)
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    seconds = time.perf_counter() - t0
    if check and proc.returncode:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")
    return proc.returncode, seconds


def capture(cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True).stdout.strip()


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def get_token():
    try:
        from google.colab import userdata
    except ImportError:
        return os.environ.get("GITHUB_TOKEN")
    for name in ("GITHUB_TOKEN", "GitHub"):
        try:
            value = userdata.get(name)
            if value:
                return value
        except Exception:
            pass
    return None


# ---------------------------------------------------------------- report

def training_summary(history_csv):
    rows = read_csv(history_csv)
    ret = [float(r["mean_policy_return"]) for r in rows]
    best_i = max(range(len(ret)), key=ret.__getitem__)
    summary = {
        "gens": len(rows),
        "start": ret[0],
        "best": ret[best_i],
        "best_gen": int(float(rows[best_i]["gen"])),
        "final": ret[-1],
        "last10_avg": sum(ret[-10:]) / min(10, len(ret)),
    }
    if "mean_policy_hits" in rows[0]:
        hits = [float(r["mean_policy_hits"]) for r in rows]
        summary["final_hits"] = hits[-1]
        summary["last10_hits_avg"] = sum(hits[-10:]) / min(10, len(hits))
    return summary


def bench_table(gpu_csv, cpu_csv):
    gpu = {int(r["envs"]): float(r["steps_per_sec"]) for r in read_csv(gpu_csv)}
    cpu = {}
    if Path(cpu_csv).exists():
        cpu = {int(r["envs"]): float(r["steps_per_sec"]) for r in read_csv(cpu_csv)}
    lines = ["| Parallel envs | GPU env-steps/s | CPU env-steps/s | GPU / CPU |",
             "| ---: | ---: | ---: | ---: |"]
    for n in sorted(gpu):
        c = cpu.get(n)
        lines.append(
            f"| {n:,} | {gpu[n]:,.0f} | {f'{c:,.0f}' if c else '–'} | "
            f"{f'{gpu[n] / c:.1f}x' if c else '–'} |"
        )
    return "\n".join(lines), gpu, cpu


def plot_throughput(gpu, cpu, path, device_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.2))
    xs = sorted(gpu)
    ax.loglog(xs, [gpu[x] for x in xs], "o-", label=f"GPU ({device_name})")
    if cpu:
        xc = sorted(cpu)
        ax.loglog(xc, [cpu[x] for x in xc], "s--", label="CPU (Colab)")
    ax.set_xlabel("parallel envs")
    ax.set_ylabel("env steps / second")
    ax.set_title("Drone env throughput: GPU vs CPU")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def build_section(device_name, info, bench_md, peak, n_episodes):
    d, p = info["drone"], info["pendulum"]
    pend_status = ("solved (≥ −200)" if p["last10_avg"] >= -200
                   else "improved, still not solved (solved ≈ −150)")
    date = dt.date.today().isoformat()
    gifs_d = " | ".join(f"![drone gpu {i}](results/gpu/drone/episode_{i}.gif)"
                        for i in range(n_episodes))
    gifs_p = " | ".join(f"![pendulum gpu {i}](results/gpu/pendulum/episode_{i}.gif)"
                        for i in range(n_episodes))
    header = " | ".join(f"Episode {i}" for i in range(n_episodes))
    align = " | ".join(":---:" for _ in range(n_episodes))
    return f"""### GPU results ({device_name})

_Generated by `colab_run.py` on {date}. Raw data in [`results/gpu/`](results/gpu/)._

| Task | Config | Start | Best (gen) | Final | Last-10 avg | Wall time |
| :--- | :--- | :---: | :---: | :---: | :---: | ---: |
| Drone | {' '.join(DRONE_ARGS[2:])} | {d['start']:.1f} | {d['best']:.1f} ({d['best_gen']}) | {d['final']:.1f}, {d['final_hits']:.1f} targets | {d['last10_avg']:.1f}, {d['last10_hits_avg']:.1f} targets | {info['drone_time'] / 60:.1f} min |
| Pendulum | {' '.join(PENDULUM_ARGS[2:])} | {p['start']:.1f} | {p['best']:.1f} ({p['best_gen']}) | {p['final']:.1f} | {p['last10_avg']:.1f} | {info['pendulum_time'] / 60:.1f} min |

Pendulum status: **{pend_status}**.

**Throughput.** Peak GPU throughput was **{peak[1]:,.0f} env-steps/s at {peak[0]:,} parallel envs**.

{bench_md}

![GPU vs CPU throughput](results/gpu/throughput_gpu_vs_cpu.png)

| Drone learning curve | Pendulum learning curve |
| :---: | :---: |
| ![drone curve](results/gpu/drone/learning_curve.png) | ![pendulum curve](results/gpu/pendulum/learning_curve.png) |

| {header} |
| {align} |
| {gifs_d} |
| {gifs_p} |
"""


def update_readme(readme, section, marker="GPU-RESULTS"):
    text = readme.read_text()
    start, end = f"<!-- {marker}:START -->", f"<!-- {marker}:END -->"
    if start in text and end in text:
        before = text.split(start)[0]
        after = text.split(end, 1)[1]
        text = f"{before}{start}\n{section}{end}{after}"
    else:
        text += f"\n\n{start}\n{section}{end}\n"
    readme.write_text(text)


def build_sweep_section(device_name, sweep_rows, best, tuned, baseline, n_episodes):
    date = dt.date.today().isoformat()
    head = "| Rank | lr | sigma | lr decay | Score (last-20 avg) | Best | Time |"
    sep = "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    table = [head, sep]
    for i, r in enumerate(sweep_rows, 1):
        mark = " **(best)**" if i == 1 else ""
        table.append(
            f"| {i}{mark} | {float(r['lr']):g} | {float(r['sigma']):g} | {float(r['lr_decay']):g} | "
            f"{float(r['score']):.1f} | {float(r['best']):.1f} | {float(r['seconds']):.0f}s |"
        )

    cmp_rows = ["| Task | Baseline last-10 avg | Tuned last-10 avg | Change |",
                "| :--- | ---: | ---: | ---: |"]
    for env in ("pendulum", "drone"):
        if env in tuned and env in baseline:
            b, t = baseline[env], tuned[env]
            extra_b = f", {b['last10_hits_avg']:.1f} targets" if "last10_hits_avg" in b else ""
            extra_t = f", {t['last10_hits_avg']:.1f} targets" if "last10_hits_avg" in t else ""
            cmp_rows.append(f"| {env} | {b['last10_avg']:.1f}{extra_b} | "
                            f"{t['last10_avg']:.1f}{extra_t} | {t['last10_avg'] - b['last10_avg']:+.1f} |")

    p_t, p_b = tuned.get("pendulum"), baseline.get("pendulum")
    gain = p_t["last10_avg"] - p_b["last10_avg"] if (p_t and p_b) else 0.0
    if p_t and p_t["last10_avg"] >= -200:
        verdict = "**Solved:** the tuned pendulum averages better than −200."
    elif gain >= 100:
        verdict = (f"**Big improvement, not solved yet:** tuning gained {gain:+.0f} on the pendulum, "
                   "so lr / sigma were a real bottleneck, but it is still short of ≈ −150.")
    else:
        verdict = (f"**Still stuck:** the best config only moved the pendulum by {gain:+.0f}, "
                   "so the plateau is probably not about lr / sigma.")
    gifs = " | ".join(f"![tuned pendulum {i}](results/gpu/tuned/pendulum/episode_{i}.gif)"
                      for i in range(n_episodes))
    header = " | ".join(f"Episode {i}" for i in range(n_episodes))
    align = " | ".join(":---:" for _ in range(n_episodes))
    return f"""### Hyperparameter sweep ({device_name})

_Generated by `colab_run.py --sweep` on {date}. Raw data in [`results/gpu/sweep/`](results/gpu/sweep/) and [`results/gpu/tuned/`](results/gpu/tuned/)._

Pendulum, {len(sweep_rows)} configs, each trained from the same seed with `{' '.join(SWEEP_ARGS[2:8])}`. Score = mean-policy return averaged over the last 20 generations.

{chr(10).join(table)}

![sweep curves](results/gpu/sweep/curves.png)

**Best config:** lr = {float(best['lr']):g}, sigma = {float(best['sigma']):g}, lr decay = {float(best['lr_decay']):g}. Retrained at full size (pop 2048, more generations) on both tasks:

{chr(10).join(cmp_rows)}

{verdict}

| {header} |
| {align} |
| {gifs} |
"""


# ---------------------------------------------------------------- main

def setup():
    """Clone/pull, install CUDA JAX, refuse to continue without a GPU, run tests."""
    banner("Setup: repo")
    if not (WORKDIR / ".git").exists():
        run(["git", "clone", f"https://github.com/{REPO}", str(WORKDIR)])
    run(["git", "pull", "--ff-only"], cwd=WORKDIR)
    os.chdir(WORKDIR)

    banner("Setup: install JAX (CUDA) + package")
    if shutil.which("nvidia-smi") is not None:
        run([sys.executable, "-m", "pip", "install", "-q", "-U", "jax[cuda12]"])
    run([sys.executable, "-m", "pip", "install", "-q", "-e", ".[dev]"])

    devices = capture([sys.executable, "-c",
                       "import jax; d=jax.devices()[0]; print(d.platform, '|', d.device_kind)"])
    print("JAX device:", devices)
    platform, _, device_name = devices.partition(" | ")
    if platform != "gpu" and not ALLOW_CPU:
        raise SystemExit(
            "JAX does not see a GPU. Set Runtime -> Change runtime type -> T4 GPU, "
            "then run again (or set ALLOW_CPU = True to run on CPU anyway)."
        )

    banner("Setup: tests")
    run([sys.executable, "-m", "pytest", "-q"])
    return platform, device_name


def push(paths, message):
    banner("Commit + push")
    token = get_token()
    if not token:
        print("No GitHub token found in Colab Secrets (GITHUB_TOKEN or GitHub). "
              "Results are saved locally in", WORKDIR, "but were NOT pushed.")
        return
    run(["git", "config", "user.name", "shiroonigami23-ui"])
    run(["git", "config", "user.email", "233169249+shiroonigami23-ui@users.noreply.github.com"])
    run(["git", "add", *map(str, paths)])
    if not capture(["git", "status", "--porcelain"]):
        print("Nothing changed, nothing to push.")
        return
    run(["git", "commit", "-q", "-m", message])

    url = f"https://x-access-token:{token}@github.com/{REPO}.git"
    # Output is captured, never printed, so the token can't leak into the notebook.
    for args in (["pull", "--rebase", "-q", url, "main"], ["push", "-q", url, "HEAD:main"]):
        res = subprocess.run(["git", *args], capture_output=True, text=True)
        if res.returncode:
            err = (res.stderr or res.stdout).replace(token, "***")
            raise SystemExit(f"git {args[0]} failed:\n{err}")
    print(f"Pushed to https://github.com/{REPO}")


def full_run():
    t_start = time.perf_counter()
    platform, device_name = setup()

    banner("Throughput benchmark")
    bench_dir = OUT / "bench"
    run([sys.executable, "-m", "swarm_es.bench", "--env", "drone",
         "--sizes", *BENCH_SIZES, "--out", str(bench_dir)])
    run([sys.executable, "-m", "swarm_es.bench", "--env", "pendulum",
         "--sizes", *BENCH_SIZES, "--out", str(bench_dir)])

    banner("Train drone")
    _, drone_time = run([sys.executable, "-m", "swarm_es.train", *DRONE_ARGS, "--out", str(OUT)])

    banner("Train pendulum")
    _, pend_time = run([sys.executable, "-m", "swarm_es.train", *PENDULUM_ARGS, "--out", str(OUT)])

    banner("Render + report")
    for env in ("drone", "pendulum"):
        run([sys.executable, "-m", "swarm_es.render", "--run", str(OUT / env),
             "--episodes", RENDER_EPISODES])

    info = {
        "drone": training_summary(OUT / "drone" / "history.csv"),
        "pendulum": training_summary(OUT / "pendulum" / "history.csv"),
        "drone_time": drone_time,
        "pendulum_time": pend_time,
    }
    gpu_csv = bench_dir / f"drone_{platform}.csv"
    bench_md, gpu, cpu = bench_table(gpu_csv, "results/bench/drone_cpu.csv")
    peak = max(gpu.items(), key=lambda kv: kv[1])
    plot_throughput(gpu, cpu, OUT / "throughput_gpu_vs_cpu.png", device_name)
    info["device"] = device_name
    info["throughput_peak"] = {"envs": peak[0], "steps_per_sec": peak[1]}
    (OUT / "summary.json").write_text(json.dumps(info, indent=2))

    section = build_section(device_name, info, bench_md, peak, int(RENDER_EPISODES))
    update_readme(Path("README.md"), section)
    print(section)

    push(["README.md", OUT], f"Add GPU results ({device_name}): training, benchmarks, renders")
    print(f"Total {(time.perf_counter() - t_start) / 60:.1f} min")


def sweep_run():
    t_start = time.perf_counter()
    _, device_name = setup()
    sweep_dir, tuned_dir = OUT / "sweep", OUT / "tuned"

    banner("Sweep: pendulum lr x sigma x lr-decay")
    run([sys.executable, "-m", "swarm_es.sweep", *SWEEP_ARGS, "--out", str(sweep_dir)])
    best = json.loads((sweep_dir / "best.json").read_text())
    rows = read_csv(sweep_dir / "sweep.csv")
    print("Best:", best)

    hp = ["--lr", str(best["lr"]), "--sigma", str(best["sigma"]), "--lr-decay", str(best["lr_decay"])]
    tuned, baseline = {}, {}
    for env in ("pendulum", "drone"):
        banner(f"Retrain {env} with the best config")
        run([sys.executable, "-m", "swarm_es.train", "--env", env, "--pop", "2048",
             "--n-eval", "5", "--gens", TUNED_GENS[env], *hp, "--out", str(tuned_dir)])
        tuned[env] = training_summary(tuned_dir / env / "history.csv")
        base_csv = OUT / env / "history.csv"
        if base_csv.exists():
            baseline[env] = training_summary(base_csv)
    run([sys.executable, "-m", "swarm_es.render", "--run", str(tuned_dir / "pendulum"),
         "--episodes", RENDER_EPISODES])
    run([sys.executable, "-m", "swarm_es.render", "--run", str(tuned_dir / "drone"),
         "--episodes", RENDER_EPISODES])

    (tuned_dir / "summary.json").write_text(json.dumps(
        {"device": device_name, "best": best, "tuned": tuned, "baseline": baseline}, indent=2))
    section = build_sweep_section(device_name, rows, best, tuned, baseline, int(RENDER_EPISODES))
    update_readme(Path("README.md"), section, marker="SWEEP-RESULTS")
    print(section)

    push(["README.md", sweep_dir, tuned_dir],
         f"Add hyperparameter sweep + tuned runs ({device_name})")
    print(f"Total {(time.perf_counter() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    if "--sweep" in sys.argv[1:]:
        sweep_run()
    else:
        full_run()
