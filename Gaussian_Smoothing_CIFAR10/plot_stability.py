from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
# Dossier des résultats
output_dir = Path("results/phase6_final/stability_safe")
# Charger les résultats calculés par seed
csv_path = output_dir / "stability_per_seed.csv"
df = pd.read_csv(csv_path)
# Vérifier les données
required = {"seed", "method", "sigma_relative", "mean_distance_l2"}
if not required.issubset(df.columns):
    raise ValueError(f"Colonnes manquantes : {required - set(df.columns)}")
# Agréger les 3 seeds : moyenne et écart-type empirique
summary = (
    df.groupby(["method", "sigma_relative"])["mean_distance_l2"]
      .agg(["mean", "std", "count"])
      .reset_index()
)
if not (summary["count"] == 3).all():
    raise ValueError("Chaque combinaison doit contenir exactement 3 seeds.")
# Sauvegarder le tableau agrégé
summary.to_csv(output_dir / "stability_aggregated.csv", index=False)
# Tracer les courbes
plt.figure(figsize=(10, 6))
for method in ["SimpleGrad", "IntegratedGradients"]:
    data = summary[summary["method"] == method].sort_values("sigma_relative")
    plt.errorbar(
        data["sigma_relative"],
        data["mean"],
        yerr=data["std"],
        marker="o",
        capsize=4,
        linewidth=2,
        label=method
    )
plt.xlabel("Sigma relatif")
plt.ylabel("Distance L2 moyenne entre les modèles A et B")
plt.title("Stabilité des cartes de saillance selon le lissage gaussien")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
# Sauvegarder la figure
figure_path = output_dir / "stability_curves.png"
plt.savefig(figure_path, dpi=300, bbox_inches="tight")
plt.show()
print("\nCourbe sauvegardée :", figure_path)
print("Tableau agrégé sauvegardé :", output_dir / "stability_aggregated.csv")
print("\nRésumé :")
print(summary.to_string(index=False))
