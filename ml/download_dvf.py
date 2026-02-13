"""
═══════════════════════════════════════════════════════════════════════════════
PROJET: AI API App E3 - Développeur IA (RNCP37827)
FICHIER: ml/download_dvf.py
COMPÉTENCES: C12, C13
═══════════════════════════════════════════════════════════════════════════════

C12 - Programmer les tests automatisés
      → Validation données (schéma, nulls, outliers)
      → Testée par test_data.py
      
C13 - Créer une chaîne de livraison continue
      → Téléchargement auto des données dans CI/CD
      → Appelée par .github/workflows/ci-cd.yml

═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("download_dvf")

# URLs officielles data.gouv.fr par année
DVF_URLS = {
    2023: "https://files.data.gouv.fr/geo-dvf/latest/csv/2023/full.csv.gz",
    2022: "https://files.data.gouv.fr/geo-dvf/latest/csv/2022/full.csv.gz",
}

DVF_COLUMNS_RAW = [
    "valeur_fonciere", "surface_reelle_bati", "nombre_pieces_principales",
    "surface_terrain", "longitude", "latitude", "type_local", 
    "code_departement", "date_mutation", "nature_mutation"
]

OUTPUT_COLUMNS = [
    "surface_reelle_bati", "nombre_pieces_principales", "surface_terrain",
    "longitude", "latitude", "type_local", "code_departement", "valeur_fonciere"
]


def download_dvf(year: int, output_path: str, sample_size: int = None):
    """
    Télécharge, nettoie et sauvegarde les données DVF.
    
    Args:
        year: Année des données (2022-2023)
        output_path: Chemin de sortie du CSV
        sample_size: Nombre de lignes (None = toutes)
    """
    if year not in DVF_URLS:
        raise ValueError(f"Année {year} non disponible")

    url = DVF_URLS[year]
    logger.info(f"Téléchargement DVF {year} depuis {url}")

    try:
        df_raw = pd.read_csv(
            url, sep=",", 
            usecols=lambda c: c in DVF_COLUMNS_RAW,
            dtype={"code_departement": str}, 
            low_memory=False
        )
        logger.info(f"DVF {year} chargé : {len(df_raw):,} lignes brutes")
    except Exception as e:
        logger.error(f"Erreur : {e}. Génération données synthétiques...")
        df_raw = _generate_synthetic_dvf(n=50_000, year=year)

    df = _clean_dvf(df_raw)

    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        logger.info(f"Échantillonnage : {sample_size:,} lignes")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df[OUTPUT_COLUMNS].to_csv(output_path, index=False)
    logger.info(f"Dataset sauvegardé : {output_path} ({len(df):,} lignes)")

    # Données de référence pour le monitoring
    ref_path = output_path.replace(".csv", "_reference.csv")
    df.sample(frac=0.3, random_state=42)[OUTPUT_COLUMNS].to_csv(ref_path, index=False)
    logger.info(f"Données référence : {ref_path}")

    return df


def _clean_dvf(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie le jeu de données DVF."""
    logger.info("Nettoyage des données DVF...")
    
    if "nature_mutation" in df.columns:
        df = df[df["nature_mutation"] == "Vente"]

    required = ["valeur_fonciere", "surface_reelle_bati", "type_local",
                "code_departement", "longitude", "latitude"]
    df = df.dropna(subset=required)

    # Conversions de types
    df["valeur_fonciere"] = pd.to_numeric(df["valeur_fonciere"], errors="coerce")
    df["surface_reelle_bati"] = pd.to_numeric(df["surface_reelle_bati"], errors="coerce")
    df["nombre_pieces_principales"] = pd.to_numeric(
        df["nombre_pieces_principales"], errors="coerce"
    ).fillna(1).astype(int)
    df["surface_terrain"] = pd.to_numeric(df["surface_terrain"], errors="coerce").fillna(0.0)
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["code_departement"] = df["code_departement"].astype(str).str.zfill(2).str[:3]

    df = df.dropna(subset=required)

    # Filtres métier
    df = df[df["valeur_fonciere"].between(10_000, 5_000_000)]
    df = df[df["surface_reelle_bati"].between(5, 2000)]
    df = df[df["type_local"].isin(["Maison", "Appartement", "Dépendance"])]
    df = df[df["longitude"].between(-5.5, 10.0)]
    df = df[df["latitude"].between(41.0, 52.0)]
    df = df[df["nombre_pieces_principales"].between(1, 20)]
    df = df[df["surface_terrain"].between(0, 50_000)]

    # Filtre prix/m²
    df["prix_m2"] = df["valeur_fonciere"] / df["surface_reelle_bati"]
    df = df[df["prix_m2"].between(200, 30_000)]
    df = df.drop(columns=["prix_m2"])
    df = df.drop_duplicates()

    logger.info(f"Après nettoyage : {len(df):,} lignes")
    return df


def _generate_synthetic_dvf(n: int = 50_000, year: int = 2023) -> pd.DataFrame:
    """Génère des données DVF synthétiques (fallback si téléchargement échoue)."""
    np.random.seed(42)
    surfaces = np.clip(np.random.lognormal(mean=4.2, sigma=0.5, size=n), 10, 500)
    pieces = np.clip(np.round(surfaces / 25).astype(int), 1, 10)

    depts = np.random.choice(
        ["75", "69", "13", "33", "44", "06", "59", "67", "31", "92"], n
    )
    
    prix_m2_base = {
        "75": 10000, "92": 7500, "69": 4500, "13": 3500, "06": 5000, 
        "33": 4000, "44": 3800, "59": 2500, "67": 3200, "31": 3600
    }
    base_prices = np.array([prix_m2_base.get(d, 3000) for d in depts])
    prices = (surfaces * base_prices * np.random.uniform(0.7, 1.4, n)).round(0)

    types = np.random.choice(["Appartement", "Maison", "Dépendance"], n, p=[0.65, 0.30, 0.05])
    terrains = np.where(types == "Maison", np.random.uniform(50, 1000, n), np.zeros(n))

    dept_coords = {
        "75": (2.35, 48.86), "69": (4.84, 45.76), "13": (5.37, 43.30),
        "33": (-0.58, 44.84), "44": (-1.55, 47.22), "06": (7.26, 43.71)
    }
    lons = np.array([dept_coords.get(d, (2.35, 48.86))[0] for d in depts])
    lats = np.array([dept_coords.get(d, (2.35, 48.86))[1] for d in depts])
    lons += np.random.normal(0, 0.3, n)
    lats += np.random.normal(0, 0.2, n)

    return pd.DataFrame({
        "valeur_fonciere": prices,
        "surface_reelle_bati": surfaces,
        "nombre_pieces_principales": pieces,
        "surface_terrain": terrains,
        "longitude": lons,
        "latitude": lats,
        "type_local": types,
        "code_departement": depts,
        "nature_mutation": "Vente",
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--output", type=str, default="data/dvf_clean.csv")
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()
    
    download_dvf(year=args.year, output_path=args.output, sample_size=args.sample)