import random
import time

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ---- Default GA Parameters (lab specification; adjustable in the sidebar) ----
DEFAULTS = {
    "pop_size": 300,
    "chrom_len": 80,
    "target_ones": 50,
    "generations": 50,
    "tournament_k": 3,
    "crossover_rate": 0.9,
    "mutation_rate": 0.0125,   # about 1 bit flip per 80-bit chromosome
    "autoplay_speed": 0.20,    # seconds per generation
    "fixed_seed": False,
    "seed_input": 42,
}


# ---- Fitness Function ----
def fitness(individual: np.ndarray, target_ones: int, max_fitness: int) -> float:
    """Max fitness (= chromosome length) is returned when ones == target_ones."""
    ones = int(individual.sum())
    return max_fitness - abs(ones - target_ones)


# ---- GA Operators ----
def init_population(pop_size: int, chrom_len: int) -> np.ndarray:
    return np.random.randint(0, 2, size=(pop_size, chrom_len), dtype=np.int8)


def tournament_selection(pop: np.ndarray, fits: np.ndarray, k: int) -> np.ndarray:
    idxs = np.random.randint(0, len(pop), size=k)
    best_idx = idxs[np.argmax(fits[idxs])]
    return pop[best_idx].copy()


def single_point_crossover(p1: np.ndarray, p2: np.ndarray, crossover_rate: float):
    if np.random.rand() > crossover_rate:
        return p1.copy(), p2.copy()
    point = np.random.randint(1, len(p1))  # cut between [1 .. len-1]
    c1 = np.concatenate([p1[:point], p2[point:]])
    c2 = np.concatenate([p2[:point], p1[point:]])
    return c1, c2


def mutate(individual: np.ndarray, mutation_rate: float) -> np.ndarray:
    mask = np.random.rand(len(individual)) < mutation_rate
    individual[mask] = 1 - individual[mask]
    return individual


def evolve(pop, generations, target_ones, max_fitness,
           tournament_k, crossover_rate, mutation_rate):
    """Run the GA and return one record per generation (used for autoplay)."""
    history = []
    for g in range(1, generations + 1):
        fits = np.array([fitness(ind, target_ones, max_fitness) for ind in pop])
        i = int(np.argmax(fits))
        history.append({
            "generation": g,
            "best": float(fits[i]),
            "mean": float(fits.mean()),
            "worst": float(fits.min()),
            "ones": int(pop[i].sum()),
            "bits": "".join(map(str, pop[i].tolist())),
        })

        new_pop = []
        while len(new_pop) < len(pop):
            p1 = tournament_selection(pop, fits, tournament_k)
            p2 = tournament_selection(pop, fits, tournament_k)
            c1, c2 = single_point_crossover(p1, p2, crossover_rate)
            new_pop.extend([mutate(c1, mutation_rate), mutate(c2, mutation_rate)])
        pop = np.array(new_pop[:len(pop)], dtype=np.int8)
    return history


# ---- Drawing helper: one "frame" of the GA at generation g ----
def draw_generation(res: dict, g: int):
    hist = res["history"][:g]
    cur = hist[-1]
    max_fit = res["max_fitness"]
    G = res["generations"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Generation", f"{g} / {G}")
    c2.metric("Best fitness", f"{cur['best']:.0f} / {max_fit}")
    c3.metric("Average fitness", f"{cur['mean']:.2f}")
    c4.metric("Ones in best", cur["ones"])
    st.code(cur["bits"], language="text")

    st.subheader("📉 Convergence (Best and Average Fitness per Generation)")
    fig, ax = plt.subplots()
    xs = [h["generation"] for h in hist]
    ax.plot(xs, [h["best"] for h in hist], linewidth=2, label="Best")
    ax.plot(xs, [h["mean"] for h in hist], linewidth=2, label="Average")

    # Fix the axes to the full run so the chart does not jump while autoplaying.
    # They use matplotlib's automatic 5% margins, so the final frame looks the same
    # as a normal auto-scaled plot (y-axis starts near the lowest average fitness).
    allv = [h["best"] for h in res["history"]] + [h["mean"] for h in res["history"]]
    ylo, yhi = min(allv), max(allv)
    ypad = 0.05 * max(yhi - ylo, 1e-9)
    xpad = 0.05 * max(G - 1, 1)
    ax.set_xlim(1 - xpad, G + xpad)
    ax.set_ylim(ylo - ypad, yhi + ypad)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Fitness")
    ax.set_title("GA Convergence")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right")
    st.pyplot(fig)
    plt.close(fig)


# ---- Reset callback ----
def reset_defaults():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.session_state.pop("result", None)


# ---- Streamlit UI ----
st.set_page_config(page_title="Genetic Algorithm | Bit Pattern Evolution",
                   page_icon="🧬", layout="centered")

for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
# keep the target slider valid if the chromosome length is lowered
if st.session_state["target_ones"] > st.session_state["chrom_len"]:
    st.session_state["target_ones"] = st.session_state["chrom_len"]

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ GA Parameters")

    st.subheader("Problem setup")
    pop_size = st.slider("Population size", 10, 1000, step=10, key="pop_size")
    chrom_len = st.slider("Chromosome length", 10, 200, key="chrom_len")
    target_ones = st.slider("Target number of ones", 0, chrom_len, key="target_ones")
    generations = st.slider("Number of generations", 1, 200, key="generations")

    st.subheader("Operators")
    tournament_k = st.slider("Tournament size (k)", 2, 10, key="tournament_k")
    crossover_rate = st.slider("Crossover rate", 0.0, 1.0, step=0.05, key="crossover_rate")
    mutation_rate = st.slider("Mutation rate (per bit)", 0.0, 0.2, step=0.0005,
                              format="%.4f", key="mutation_rate")

    st.subheader("Run")
    fixed_seed = st.checkbox("Use fixed seed (same output every run)", key="fixed_seed")
    seed_input = st.number_input("Seed value", min_value=0, step=1,
                                 disabled=not fixed_seed, key="seed_input")
    run_btn = st.button("Run Genetic Algorithm", type="primary", use_container_width=True)

    st.subheader("Autoplay")
    autoplay_speed = st.slider("Autoplay speed (seconds per generation)",
                               0.0, 2.0, step=0.05, key="autoplay_speed")
    st.button("🔄 Reset to default settings", use_container_width=True,
              on_click=reset_defaults)

max_fitness = chrom_len  # fitness is 80 for length 80 at ones == 50

# ---------------- Run the GA ----------------
if run_btn:
    seed = int(seed_input) if fixed_seed else random.randrange(1_000_000)
    random.seed(seed)
    np.random.seed(seed)
    history = evolve(init_population(pop_size, chrom_len), generations, target_ones,
                     max_fitness, tournament_k, crossover_rate, mutation_rate)
    st.session_state["result"] = {
        "history": history, "seed": seed, "generations": generations,
        "target_ones": target_ones, "max_fitness": max_fitness,
    }
    st.session_state["inspect_gen"] = generations  # start the inspector at the end

# ---------------- Main page ----------------
st.title("🧬 Genetic Algorithm | Interactive Bit Pattern Evolution")
st.caption(
    "BSD3513 Introduction to Artificial Intelligence · Loh Xiang YI · Lab Report 2 · "
    "Genetic Algorithm · UMPSA"
)
st.caption(
    f"Evolving a {chrom_len}-bit pattern. Fitness peaks at ones = {target_ones}, "
    f"with max fitness = {max_fitness}. Change the parameters in the sidebar and press **Run**."
)

with st.expander("ℹ️ Current problem setup", expanded=False):
    st.write(
        f"""
- **Population size**: `{pop_size}`  
- **Chromosome length**: `{chrom_len}`  
- **Target number of ones**: `{target_ones}`  
- **Max fitness at optimum**: `{max_fitness}` (when ones = {target_ones})  
- **Generations**: `{generations}`  
- **Selection**: Tournament (k={tournament_k})  
- **Crossover**: Single-point (rate={crossover_rate})  
- **Mutation**: Bit-flip (per-bit rate={mutation_rate:.4f})
"""
    )

res = st.session_state.get("result")
if res is None:
    st.info("👈 Set the parameters in the sidebar, then click **Run Genetic Algorithm**.")
else:
    G = res["generations"]
    best_rec = max(res["history"], key=lambda h: h["best"])
    first_opt = next((h["generation"] for h in res["history"]
                      if h["best"] == res["max_fitness"]), None)

    st.subheader("🏁 Result")
    st.caption(f"Seed used for this run: {res['seed']}")
    m1, m2, m3 = st.columns(3)
    m1.metric("Best fitness", f"{best_rec['best']:.0f} / {res['max_fitness']}")
    m2.metric("Ones in best individual", best_rec["ones"])
    m3.metric("Optimum first reached", f"Gen {first_opt}" if first_opt else "—")

    if first_opt:
        st.success(f"Perfect match achieved: ones = {res['target_ones']} and "
                   f"fitness = {res['max_fitness']} ✅")
    else:
        st.info("GA reached a near-optimal solution; try another seed or more generations.")

    st.subheader("🎞️ Step through the evolution")
    st.slider("GA: inspect generation", 1, G, key="inspect_gen")
    st.caption("💡 Click Autoplay to watch the GA evolve generation by generation.")
    autoplay = st.button("▶️ Autoplay GA", key="autoplay_btn")

    view = st.empty()
    if autoplay:
        for g in range(1, G + 1):
            with view.container():
                draw_generation(res, g)
            time.sleep(autoplay_speed)
    else:
        with view.container():
            draw_generation(res, st.session_state["inspect_gen"])

st.caption("BSD3513 Introduction to Artificial Intelligence · Lab Report 2 · UMPSA")