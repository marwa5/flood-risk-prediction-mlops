# Prédiction du Risque d'Inondation — Nord Tunisie

## PFA Machine Learning — Université iTeam — 2025/2026

---

## Introduction

Ce projet vise à développer une solution de **Machine Learning supervisé pour l'estimation du risque d'inondation dans le nord de la Tunisie**.

Le dataset `InondatNorthTun.csv` contient différentes variables géographiques, topographiques et climatiques décrivant les zones étudiées, notamment l'altitude, la pente, les précipitations, les distances aux cours d'eau, à la mer et aux routes, ainsi que le type d'occupation du sol.

L'objectif est de concevoir un pipeline de Machine Learning permettant de :

* préparer et analyser les données ;
* développer des variables pertinentes à travers le **feature engineering** ;
* sélectionner les variables les plus discriminantes ;
* entraîner et optimiser plusieurs modèles de classification ;
* comparer leurs performances à l'aide de métriques adaptées ;
* analyser les erreurs de classification et les probabilités prédites ;
* préparer le modèle pour un déploiement sous forme de service de prédiction.

### Pipeline du projet

1. **Exploration et préparation des données**

   * analyse statistique et visualisation ;
   * traitement des valeurs manquantes ;
   * analyse des distributions et des classes ;
   * préparation des variables numériques et catégorielles.

2. **Feature Engineering**

   * création de variables liées à la proximité de l'eau ;
   * transformations logarithmiques des variables asymétriques ;
   * création de variables combinant les caractéristiques topographiques et climatiques.

3. **Sélection des variables**

   * analyse statistique des variables ;
   * sélection des caractéristiques pertinentes avec `SelectKBest` et le test ANOVA F-score ;
   * intégration de la sélection dans le pipeline d'entraînement afin d'éviter les fuites de données.

4. **Modélisation**

   * Logistic Regression ;
   * Random Forest ;
   * Support Vector Machine (SVM) ;
   * XGBoost ;
   * optimisation des hyperparamètres par validation croisée stratifiée.

5. **Évaluation**

   * ROC-AUC ;
   * Precision ;
   * Recall ;
   * F1-score ;
   * Accuracy ;
   * Specificity ;
   * matrices de confusion ;
   * courbes ROC et analyse des seuils de décision.

6. **Déploiement**

   * sauvegarde du modèle et de ses artefacts ;
   * exposition du modèle à travers une API ;
   * conteneurisation avec Docker ;
   * interface interactive pour tester les prédictions.

### Architecture cible

À terme, le projet évoluera vers une architecture **MLOps** intégrant le suivi des expériences, la gestion des versions du modèle, le déploiement et la surveillance du service de prédiction.
