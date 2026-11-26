from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "base_prospect.csv"
REPORT_PATH = BASE_DIR / "distributions_variables.pdf"

CATEGORICAL_COLUMNS = {
    "code_cr",
    "dept",
    "risque",
    "type_com",
    "activite",
    "actionnaire",
    "forme_jur_simpl",
    "chgt_dir",
    "rdv",
}
BINARY_COLUMNS = {"chgt_dir", "rdv"}
DISCRETE_COLUMNS = {"evo_risque"}
RISK_ORDER = ["1-6", "7", "8-10", "11-13", "14", "Manquant"]


def variable_discussion(name, values, missing_count):
    observed = values.dropna()
    missing_note = (
        f"{missing_count:,} valeur(s) manquante(s)"
        if missing_count
        else "aucune valeur manquante"
    )

    if name in CATEGORICAL_COLUMNS:
        counts = observed.value_counts()
        top_value = counts.index[0]
        top_share = counts.iloc[0] / len(observed)

        if name in BINARY_COLUMNS:
            positive_share = (observed.astype(float) == 1).mean()
            law = f"Bernoulli empirique (p={positive_share:.1%} pour la modalité 1)"
            variable_type = "Qualitative binaire"
        elif name == "risque":
            law = "pas de loi continue standard; fréquences catégorielles"
            variable_type = "Qualitative ordinale"
        else:
            law = "pas de loi continue standard; fréquences catégorielles"
            variable_type = "Qualitative nominale"

        return (
            f"{variable_type}; {observed.nunique()} modalité(s); "
            f"mode = {top_value} ({top_share:.1%}). {missing_note}. {law}."
        )

    skewness = observed.skew()
    median = observed.median()
    mean = observed.mean()

    if name in DISCRETE_COLUMNS:
        zero_share = (observed == 0).mean()
        shape = f"discrète, concentrée en 0 ({zero_share:.1%})"
    elif skewness > 1:
        shape = "fortement asymétrique à droite, avec une longue queue"
    elif skewness > 0.4:
        shape = "asymétrique à droite"
    elif skewness < -1:
        shape = "fortement asymétrique à gauche"
    elif skewness < -0.4:
        shape = "asymétrique à gauche"
    else:
        shape = "asymétrie faible"

    if name == "effectif":
        standard = "pas Poisson (variance très supérieure à la moyenne)"
    else:
        standard = "pas de loi normale apparente"

    extra = ""
    if name in {"evo_effectif", "ca_export_FK"}:
        extra = f"; {(observed == 0).mean():.1%} de zéros"

    return (
        f"Quantitative; {shape} (asymétrie={skewness:.2f}); "
        f"médiane={median:.4g}, moyenne={mean:.4g}{extra}. "
        f"{missing_note}. {standard}."
    )


def plot_categorical(axis, name, values):
    display_values = values.astype("object").where(values.notna(), "Manquant")
    counts = display_values.value_counts()

    if name == "dept":
        counts = counts.sort_index()
        axis.bar(counts.index.astype(str), counts.values, color="#4472C4")
        axis.tick_params(axis="x", labelrotation=90, labelsize=7)
        axis.set_xlabel("Département")
    else:
        if name == "risque":
            counts = counts.reindex(RISK_ORDER, fill_value=0)
        else:
            counts = counts.sort_values()
        labels = [str(value) for value in counts.index]
        axis.barh(labels, counts.values, color="#4472C4")
        axis.tick_params(axis="y", labelsize=8)
        axis.set_xlabel("Nombre d’entreprises")

    axis.set_ylabel("")
    axis.set_title(f"Distribution de {name}")


def plot_numeric(axis, name, values):
    observed = values.dropna()

    if name in DISCRETE_COLUMNS:
        counts = observed.value_counts().sort_index()
        axis.bar(counts.index.astype(str), counts.values, color="#70AD47")
        axis.set_xlabel(name)
        axis.set_ylabel("Nombre d’entreprises")
        axis.set_title(f"Distribution de {name} (valeurs discrètes)")
        return

    lower = observed.quantile(0.005)
    upper = observed.quantile(0.995)
    visible = observed[(observed >= lower) & (observed <= upper)]
    bin_count = min(max(int(np.sqrt(len(visible))), 20), 80)
    axis.hist(visible, bins=bin_count, color="#70AD47", edgecolor="white")
    axis.set_xlabel(name)
    axis.set_ylabel("Nombre d’entreprises")
    axis.set_title(f"Distribution de {name}")

    excluded = len(observed) - len(visible)
    if excluded:
        axis.text(
            0.99,
            0.96,
            f"Axe limité aux quantiles 0,5–99,5 %\n"
            f"{excluded:,} valeur(s) extrême(s) hors champ",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )


def main():
    df = pd.read_csv(DATA_PATH, encoding="latin1")
    print(f"Données : {len(df):,} lignes, {len(df.columns)} variables")

    with PdfPages(REPORT_PATH) as pdf:
        for name in df.columns:
            values = df[name]
            figure, axis = plt.subplots(figsize=(11.7, 8.3))

            if name in CATEGORICAL_COLUMNS:
                plot_categorical(axis, name, values)
            else:
                plot_numeric(axis, name, values)

            discussion = variable_discussion(name, values, int(values.isna().sum()))
            figure.text(
                0.08,
                0.035,
                discussion,
                ha="left",
                va="bottom",
                wrap=True,
                fontsize=9,
            )
            figure.tight_layout(rect=(0.06, 0.13, 0.98, 0.95))
            pdf.savefig(figure)
            plt.close(figure)
            print(f"- {name}: {discussion}")

    print(f"Graphiques et commentaires enregistrés dans : {REPORT_PATH}")


if __name__ == "__main__":
    main()
