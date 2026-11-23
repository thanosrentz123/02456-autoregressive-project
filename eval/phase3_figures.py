from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

EVAL = Path("results/phase3/eval")
OUT = Path("results/phase3/figures")
SETS = {
    "cg_AcyP_lambda0": ("CG", 0.0), "cg_AcyP_lambda0.5": ("CG", 0.5),
    "cg_AcyP_lambda1": ("CG", 1.0), "cg_AcyP_lambda2": ("CG", 2.0),
    "cg_AcyP_lambda4": ("CG", 4.0),
    "cfg_null": ("CFG", 0.0), "cfg_AcyP_w0": ("CFG", 1.0), "cfg_AcyP_w0.5": ("CFG", 1.5),
    "cfg_AcyP_w1": ("CFG", 2.0), "cfg_AcyP_w2": ("CFG", 3.0), "cfg_AcyP_w3": ("CFG", 4.0),
}
ARMS = (("CG", "o", "tab:blue"), ("CFG", "s", "tab:orange"))


def load():
    rows = []
    for name in list(SETS) + ["control_real", "control_shuffled"]:
        row = pd.read_csv(EVAL / name / "summary.csv").iloc[0].to_dict()
        row["set"] = name
        row["method"], row["lam"] = SETS.get(name, ("control", float("nan")))
        rows.append(row)
    table = pd.DataFrame(rows).set_index("set")
    for predictor in ("guide", "judge"):
        path = EVAL / f"family_membership_{predictor}_AcyP.csv"
        table[predictor] = pd.read_csv(path).set_index("set")["frac_target"]
    return table


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t = load()
    t.to_csv(OUT / "phase3_table.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
    for method, marker, color in ARMS:
        d = t[t.method == method].sort_values("lam")
        a.plot(d.lam, d.judge, marker=marker, color=color, label=method)
        b.plot(d.lam, d.mean_plddt, marker=marker, color=color, label=method)

    a.axhline(1 / 3, color="grey", ls=":", label="chance")
    a.set_xlabel("guidance strength λ")
    a.set_ylabel("predicted target-family membership")
    a.set_title("Guidance increases family membership")
    a.legend()

    b.axhline(t.loc["control_real", "mean_plddt"], color="grey", ls="-", label="real proteins")
    b.axhline(t.loc["control_shuffled", "mean_plddt"], color="grey", ls="--", label="shuffled")
    b.set_xlabel("guidance strength λ")
    b.set_ylabel("mean pLDDT")
    b.set_title("Structural confidence dips, then recovers")
    b.legend()

    fig.tight_layout()
    fig.savefig(OUT / "guidance.png", dpi=200)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for method, marker, color in ARMS:
        d = t[t.method == method].sort_values("lam")
        ax.plot(d.lam, d.judge, marker=marker, color=color,
                label=f"{method}, independent judge")
        ax.plot(d.lam, d.guide, marker=marker, color=color, ls="--",
                label=f"{method}, guiding classifier")
    ax.set_xlabel("guidance strength λ")
    ax.set_ylabel("predicted target-family membership")
    ax.set_title("Scored by the guiding classifier vs an independent one")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "circularity.png", dpi=200)
    plt.close(fig)

    print(t[["method", "lam", "guide", "judge", "mean_plddt",
             "nearest_identity_median"]].round(3).to_string())
    print("\nFigures in", OUT)


if __name__ == "__main__":
    main()