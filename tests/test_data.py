"""
═══════════════════════════════════════════════════════════════════════════════
PROJET: AI API App E3 - Développeur IA (RNCP37827)
FICHIER: tests/test_data.py
COMPÉTENCES: C12
═══════════════════════════════════════════════════════════════════════════════

Validation jeu de données DVF

═══════════════════════════════════════════════════════════════════════════════
"""

"""
test_data.py – Tests de validation du jeu de données DVF (C12)
Valide : schéma, valeurs manquantes, distributions, cohérence métier.
Doit passer AVANT l'entraînement du modèle (gate de qualité des données).
"""

import pytest
import pandas as pd
import numpy as np


# ── Fixtures locales ───────────────────────────────────────────────────────────
REQUIRED_COLUMNS = [
    "surface_reelle_bati", "nombre_pieces_principales", "surface_terrain",
    "longitude", "latitude", "type_local", "code_departement", "valeur_fonciere"
]
NUMERIC_COLS  = ["surface_reelle_bati", "nombre_pieces_principales",
                  "surface_terrain", "longitude", "latitude", "valeur_fonciere"]
VALID_TYPES   = {"Maison", "Appartement", "Dépendance"}


class TestDVFSchema:
    """Vérifie la structure du jeu de données."""

    def test_all_required_columns_present(self, dvf_dataframe):
        for col in REQUIRED_COLUMNS:
            assert col in dvf_dataframe.columns, f"Colonne manquante : {col}"

    def test_dataframe_not_empty(self, dvf_dataframe):
        assert len(dvf_dataframe) > 100, "Jeu de données trop petit (< 100 lignes)"

    def test_numeric_columns_are_numeric(self, dvf_dataframe):
        for col in NUMERIC_COLS:
            assert pd.api.types.is_numeric_dtype(dvf_dataframe[col]), \
                f"{col} devrait être numérique"

    def test_no_duplicate_rows(self, dvf_dataframe):
        n_dupes = dvf_dataframe.duplicated().sum()
        assert n_dupes == 0, f"{n_dupes} lignes dupliquées"


class TestDVFMissingValues:
    """Vérifie les valeurs manquantes."""

    def test_target_has_no_nulls(self, dvf_dataframe):
        null_count = dvf_dataframe["valeur_fonciere"].isna().sum()
        assert null_count == 0, f"valeur_fonciere contient {null_count} valeurs nulles"

    def test_features_null_rate_acceptable(self, dvf_dataframe):
        """Taux de valeurs manquantes < 5 % par colonne."""
        for col in REQUIRED_COLUMNS:
            null_rate = dvf_dataframe[col].isna().mean()
            assert null_rate < 0.05, \
                f"{col} : taux de nulls trop élevé ({null_rate:.1%})"

    def test_surface_not_null(self, dvf_dataframe):
        assert dvf_dataframe["surface_reelle_bati"].isna().sum() == 0


class TestDVFDistributions:
    """Vérifie les distributions (cohérence métier)."""

    def test_prix_minimum_realiste(self, dvf_dataframe):
        """Prix minimum > 10 000 € (seuil de transaction réaliste)."""
        assert dvf_dataframe["valeur_fonciere"].min() > 10_000, \
            "Des transactions ont un prix < 10 000 € (incohérent)"

    def test_prix_maximum_coherent(self, dvf_dataframe):
        """Prix maximum < 5 M€ (après nettoyage des outliers extrêmes)."""
        pct99 = dvf_dataframe["valeur_fonciere"].quantile(0.99)
        assert pct99 < 5_000_000, f"Le 99e percentile est {pct99:,.0f} € (trop élevé)"

    def test_surface_minimale(self, dvf_dataframe):
        assert dvf_dataframe["surface_reelle_bati"].min() > 5, \
            "Surfaces < 5 m² présentes (incohérent)"

    def test_surface_maximale_coherente(self, dvf_dataframe):
        assert dvf_dataframe["surface_reelle_bati"].max() < 2500, \
            "Surfaces > 2 500 m² présentes (probable erreur de saisie)"

    def test_pieces_range(self, dvf_dataframe):
        assert dvf_dataframe["nombre_pieces_principales"].min() >= 1
        assert dvf_dataframe["nombre_pieces_principales"].max() <= 20

    def test_longitude_dans_france(self, dvf_dataframe):
        assert dvf_dataframe["longitude"].between(-5.5, 10.0).all(), \
            "Des longitudes hors de France métropolitaine"

    def test_latitude_dans_france(self, dvf_dataframe):
        assert dvf_dataframe["latitude"].between(41.0, 52.0).all(), \
            "Des latitudes hors de France métropolitaine"

    def test_type_local_valides(self, dvf_dataframe):
        invalides = set(dvf_dataframe["type_local"].unique()) - VALID_TYPES
        assert len(invalides) == 0, f"Types invalides : {invalides}"

    def test_prix_m2_coherent(self, dvf_dataframe):
        """Prix au m² entre 200 et 30 000 € (fourchette France entière)."""
        prix_m2 = dvf_dataframe["valeur_fonciere"] / dvf_dataframe["surface_reelle_bati"]
        assert prix_m2.min() >= 200,    f"Prix/m² min trop bas : {prix_m2.min():.0f} €"
        assert prix_m2.max() <= 30_000, f"Prix/m² max trop élevé : {prix_m2.max():.0f} €"


class TestDVFEquilibre:
    """Vérifie l'équilibre du jeu de données."""

    def test_types_suffisamment_representes(self, dvf_dataframe):
        """Chaque type de bien doit représenter au moins 1 % du dataset."""
        counts = dvf_dataframe["type_local"].value_counts(normalize=True)
        for t in ["Maison", "Appartement"]:
            if t in counts.index:
                assert counts[t] >= 0.01, f"{t} sous-représenté ({counts[t]:.1%})"

    def test_au_moins_3_departements(self, dvf_dataframe):
        n_depts = dvf_dataframe["code_departement"].nunique()
        assert n_depts >= 3, f"Trop peu de départements ({n_depts}), risque de biais géographique"

    def test_split_train_test_suffisant(self, dvf_dataframe):
        """80/20 split : minimum 100 lignes en test."""
        n_test = int(len(dvf_dataframe) * 0.2)
        assert n_test >= 100, f"Jeu de test trop petit : {n_test} lignes"