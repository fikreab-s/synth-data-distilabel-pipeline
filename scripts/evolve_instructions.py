"""Synthetic instruction evolution via Evol-Instruct (offline simulation)."""
import json, random, argparse, numpy as np
from pathlib import Path
random.seed(42); np.random.seed(42)

SEED_INSTRUCTIONS = [
    "What is the ROI of our TV campaign?",
    "Summarize brand performance for Q3.",
    "Compare digital vs print effectiveness.",
    "Which HCP segment has highest engagement?",
    "What budget allocation maximizes market share?",
    "Explain the carryover effect in our MMM.",
    "Generate a competitive landscape report.",
    "What is the optimal promotional mix?",
]

EVOLUTION_OPS = ["add_constraint", "deepen", "concretize", "multi_step"]

def evolve(instruction, op, depth):
    if op == "add_constraint":
        constraints = ["under a $5M budget cap","for the Northeast region only",
            "excluding samples channel","for Tier 1 HCPs only","in the last 2 quarters"]
        return f"{instruction} {random.choice(constraints)}"
    elif op == "deepen":
        deepeners = ["Also explain the statistical significance.",
            "Include confidence intervals.", "Break down by specialty.",
            "Compare year-over-year.", "Account for seasonality."]
        return f"{instruction} {random.choice(deepeners)}"
    elif op == "concretize":
        brands = ["Cardivex","Immunolex","OncoPrime","NeuraStar"]
        return instruction.replace("our", f"{random.choice(brands)}'s").replace(
            "brand", random.choice(brands))
    else:
        return f"Step 1: {instruction} Step 2: Provide actionable recommendations. Step 3: Format as executive summary."

def score_complexity(text):
    return round(len(text.split()) / 10 + len([c for c in text if c in '?.,;:']) * 0.5, 2)

def main():
    p = argparse.ArgumentParser(); p.add_argument("--n_evolutions", type=int, default=3)
    p.add_argument("--output_dir", default="data"); a = p.parse_args()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)

    all_instructions = []
    for seed in SEED_INSTRUCTIONS:
        chain = [{"depth": 0, "instruction": seed, "complexity": score_complexity(seed)}]
        current = seed
        for d in range(1, a.n_evolutions + 1):
            op = random.choice(EVOLUTION_OPS)
            current = evolve(current, op, d)
            chain.append({"depth": d, "instruction": current, "op": op,
                          "complexity": score_complexity(current)})
        all_instructions.extend(chain)

    with open(out / "evolved_instructions.jsonl", "w") as f:
        for inst in all_instructions: f.write(json.dumps(inst) + "\n")

    depths = [i["depth"] for i in all_instructions]
    complexities = [i["complexity"] for i in all_instructions]
    print(f"✅ Evol-Instruct Pipeline")
    print(f"   Seeds: {len(SEED_INSTRUCTIONS)}")
    print(f"   Evolutions: {a.n_evolutions} rounds")
    print(f"   Total: {len(all_instructions)} instructions")
    print(f"   Complexity: {np.mean(complexities):.2f} avg (depth 0: "
          f"{np.mean([c for d,c in zip(depths,complexities) if d==0]):.2f} → depth {a.n_evolutions}: "
          f"{np.mean([c for d,c in zip(depths,complexities) if d==a.n_evolutions]):.2f})")
    print(f"   📁 {out}/evolved_instructions.jsonl")

if __name__ == "__main__": main()
