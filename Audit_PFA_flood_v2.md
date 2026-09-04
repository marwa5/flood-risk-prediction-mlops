# Audit complet — PFA_iTeam_Corrige_v2_anti_leakage.ipynb

**Légende :** 🟢 correct, rien à changer · 🟠 acceptable mais à améliorer · 🔴 à corriger avant soutenance

---

## 1. Audit cellule par cellule

| Cellule | Contenu | Verdict | Problème identifié | Correction proposée |
|---|---|---|---|---|
| 0 | Introduction | 🟢 | — | — |
| 2 | Étape 0 — Imports | 🟢 | Imports standards, tous utilisés plus loin | — |
| 5 | 1.1 — Chargement/exploration | 🟢 | `df_raw` chargé une seule fois, aucune transformation | — |
| 8 | 1.2 — Distributions / outliers | 🟢 | EDA pure sur `df_raw`, pas de fit de modèle | — |
| 11 | 1.3 — Feature Engineering (fonction) | 🟢 | `apply_feature_engineering(df_in, thresholds=None)` : seuils fixes si fournis, sinon calculés sur l'input — utilisé sans `thresholds` uniquement ici pour l'EDA (cellule 1.4), pas pour l'entraînement | — |
| 14 | 1.4 — Matrice de corrélation | 🟠 | Calculée sur `X` (tout le dataset, seuils EDA) — acceptable car **purement exploratoire**, mais à documenter clairement pour la soutenance : « ce graphe n'a jamais influencé le choix d'un modèle » | Ajouter une phrase explicite dans l'interprétation (déjà présente en partie) |
| 17 | 1.5 — Split + pipelines | 🟢 | `train_test_split(..., stratify=y)` puis `FE_THRESHOLDS` recalculés et réappliqués sur `X_train`/`X_test` séparément — plus aucune fuite sur les seuils | — |
| 20 | 1.6 — Avant/après prétraitement | 🟢 | `preprocessor.fit_transform(X_train, y_train)` — fit sur train uniquement | — |
| 23 | 2.1 — SelectKBest (exploratoire) | 🟢 | Ajusté sur `X_train, y_train`. **Non branché** dans les pipelines de modélisation (chaque pipeline de l'Étape 3 utilise toutes les features) | — |
| 27 | 3.1 — Fonctions d'évaluation | 🔴 | `evaluate_model()` calcule le seuil « Optimal » via le critère de Youden **sur `y_true`/`y_proba` passés en argument** — or à l'Étape 4 cette fonction est appelée avec `y_test`/`yp_best`. Le seuil de décision « Optimal » est donc choisi **en utilisant les labels du test**, ce qui est exactement le type de fuite que le brief initial demandait d'éliminer (« Threshold selection — ne jamais choisir le seuil sur le test ») | Choisir le seuil de Youden par validation croisée sur `X_train`/`y_train` (ex. via `cross_val_predict` avec `method='predict_proba'`), le figer, puis l'appliquer **une seule fois** à `X_test`. Voir section 3 ci-dessous pour le correctif prêt à coller |
| 30 | 3.2 — Modèles + grilles d'hyperparamètres | 🟢 | Les 4 modèles reçoivent le même traitement (2 scalers × RandomizedSearchCV × même `cv`) — comparaison équitable. `scale_pos_weight` de XGBoost calculé sur `y_train` uniquement | — |
| 33 | 3.3 — Entraînement + sélection | 🟢 | Sélection du scaler et de `best_global_name` sur `search.best_score_` (AUC CV), jamais sur `auc_test`. `X_test` n'est utilisé qu'en lecture, pour le reporting | Corrigé — reste dépendant du seuil « Optimal » hérité de la cellule 27 (voir 🔴 ci-dessus) |
| 36 | 3.4 — Comparaison des scalers (graphique) | 🟢 | Visualisation de `scaler_comparison`, aucun impact sur la sélection | — |
| 39 | 4.1 — Tableau comparatif | 🟠 | Classement final par `ROC-AUC` (métrique indépendante du seuil, donc saine), mais les colonnes Accuracy/Recall/Precision/F1/Specificity du tableau utilisent le seuil « Optimal » hérité du 🔴 ci-dessus | Se corrige automatiquement une fois la cellule 27 corrigée |
| 42 | 4.2 — Matrices de confusion | 🟠 | Même dépendance au seuil « Optimal » | Idem |
| 45 | 4.3 — Courbes ROC + analyse des seuils | 🟢/🟠 | La courbe ROC elle-même est saine (ROC-AUC ne dépend pas d'un seuil) ; le tableau des 3 seuils hérite du même problème pour la colonne « Optimal » | Idem |
| 48 | 4.4 — Importance des features | 🟢 | Lit directement `.feature_importances_` des modèles déjà entraînés sur `X_train` — aucune fuite | — |
| 51 | 5.0 — Prédiction nouvelles données | 🟢 | `apply_feature_engineering(..., thresholds=FE_THRESHOLDS)` — mêmes seuils qu'à l'entraînement, corrige aussi l'ancien bug (seuils toujours à 0 pour un point unique) | — |
| 54 | 6.1 — Chargement modèle pour Gradio | 🟢 | Recharge le pipeline complet (`preprocessor` + `model`) depuis `MODEL_PATH`, identique à celui évalué | — |
| 55 | 6.2 — Interface Gradio | 🟢 | `apply_feature_engineering(row, thresholds=FE_THRESHOLDS)` — cohérent avec l'entraînement. Le tableau de métriques affiché vient de `all_results` (calculé une fois, pas recalculé à chaque clic) | Hérite du 🔴 seuil « Optimal » pour l'affichage des métriques dans l'onglet « Métriques » |
| 57 | Étape 7 — Discussion/Conclusion | 🟢 | Discussion qualitative correcte ; mentionne désormais la validation spatiale comme piste future | — |

---

## 2. Les 5 points critiques demandés

### 🔴 1. Data Leakage — bilan
- **Feature engineering (seuils)** → corrigé (`FE_THRESHOLDS` appris sur X_train).
- **Imputation / Scaling** → corrigé (dans le `Pipeline`, fit uniquement via `search.fit(X_train, y_train)`).
- **SelectKBest** → corrigé (fit sur X_train), et de toute façon non branché dans les pipelines de modélisation.
- **Sélection de modèle/scaler** → corrigé (AUC de validation croisée, pas AUC test).
- **Sélection du seuil de décision (« Optimal »/Youden)** → **pas encore corrigé** : c'est le dernier point du brief initial qui restait ouvert. Le choix du seuil actuel « regarde » `y_test`.
- **Usage de X_test** → utilisé une seule fois pour le score final, sauf pour le calcul du seuil optimal (cf. ci-dessus).

### 🗺️ 2. Fuite géographique (spatial leakage)
Je ne peux pas trancher définitivement sans exécuter le notebook sur les vraies données : les colonnes visibles dans le code sont `elev_m, slope_d, precipmm, STREAM_DIS, SEA_DISTAN, ROAD_DISTA, CITIES_SET, lc_code, id, flood` — **aucune colonne de coordonnées (latitude/longitude) n'apparaît explicitement** dans le pipeline. Deux cas possibles :
- Si le dataset ne contient vraiment aucune coordonnée exploitable, on ne peut pas faire de découpage spatial à blocs géographiques sans donnée supplémentaire (il faudrait au minimum X/Y ou une grille).
- S'il existe une colonne de coordonnées dans le CSV mais non utilisée comme feature (ce qui serait logique, pour éviter qu'un modèle apprenne juste « ce point précis »), elle pourrait servir à construire les folds spatiaux sans jamais entrer dans `X`.

**Recommandation :** vérifier dans `df_raw.columns` s'il existe une paire de coordonnées ou un identifiant de grille/tuile. Si oui, c'est ce qu'il faut utiliser pour un `GroupKFold` ou un découpage par blocs spatiaux. Sinon, il faut le signaler explicitement en soutenance comme une limite connue du projet plutôt que de la passer sous silence.

### 🤖 3. Équité entre les 4 modèles
Comparaison équitable confirmée : même `cv` (StratifiedKFold k=5), mêmes deux scalers testés pour chacun, `n_iter` différent par modèle (20/30/15/30) — c'est un choix raisonnable lié au coût de calcul de SVM, pas un biais en faveur d'un modèle en particulier tant que c'est documenté (à mentionner en soutenance : « SVM a moins d'itérations car son coût de recherche est plus élevé, pas par choix arbitraire »).

### 📊 4. Metrics — quelle doit être la métrique reine ?
Pour un système d'alerte inondation, l'ordre de priorité recommandé est :
1. **Recall** (minimiser les faux négatifs — une inondation manquée coûte des vies) — déjà mis en avant dans le notebook via le seuil « High Recall = 0.3 », bon réflexe.
2. **ROC-AUC** comme métrique de sélection de modèle (insensible au seuil et au déséquilibre de classes) — déjà utilisé, correct.
3. **Precision/F1** en soutien, pour ne pas déclencher une alerte à chaque prédiction.
4. **Accuracy** est la moins pertinente ici si les classes sont déséquilibrées (déjà signalé dans le notebook).

### 🚀 5. Déploiement / cohérence inférence-entraînement
- Le modèle sauvegardé (`joblib.dump`) est bien le pipeline complet (preprocessing + modèle), rechargé identique pour Gradio et pour l'Étape 5. 🟢
- Le feature engineering utilise maintenant les mêmes seuils qu'à l'entraînement (`FE_THRESHOLDS`) partout — plus de divergence entraînement/inférence. 🟢
- Le seul point encore non aligné est le seuil de décision « Optimal » affiché dans Gradio, hérité du 🔴 ci-dessus.

---

## 3. Correctif prêt à appliquer (le dernier 🔴)

Pour éliminer complètement la fuite de seuil, la logique doit devenir :

```
X_train → Cross-Validation (out-of-fold proba) → seuil de Youden calculé UNE FOIS sur ces proba
                                                            ↓
                                                    seuil figé (THRESHOLD_OPTIMAL)
                                                            ↓
                                        appliqué à X_test pour le rapport final
```

Concrètement, dans `evaluate_model()` ou juste avant son appel : remplacer le calcul de `opt_thr` (actuellement basé sur `roc_curve(y_test, y_proba)`) par un calcul basé sur `cross_val_predict(best_pipe_final, X_train, y_train, cv=cv, method='predict_proba')[:, 1]`, puis appliquer ce seuil figé à `X_test` sans jamais le recalculer dessus.

Dis-moi si tu veux que je fasse ce correctif directement dans le notebook — c'est un changement ciblé (cellules 27 et 33).

---

## 4. Note finale

| Axe | Note /20 | Commentaire |
|---|---|---|
| **Qualité des données / EDA** | 18/20 | Exploration complète, gestion des NaN et outliers justifiée ; il manque juste une vérification explicite des coordonnées géographiques |
| **Méthodologie (anti-leakage)** | 16/20 | 5 des 6 sources de fuite corrigées ; le seuil de décision reste calculé sur le test |
| **Modélisation (ML)** | 18/20 | 4 modèles pertinents, recherche d'hyperparamètres sérieuse, comparaison équitable |
| **Évaluation** | 15/20 | Bonnes métriques et bon choix de métrique prioritaire (Recall), pénalisé par le seuil « Optimal » biaisé |
| **MLOps / Déploiement** | 15/20 | Pipeline unique sauvegardé et rechargé correctement ; pas encore de tracking d'expériences (MLflow), pas de containerisation, pas de validation spatiale |

**Moyenne globale : ≈ 16.4/20** — nettement au-dessus d'un notebook académique standard, un cran en dessous d'un projet « production-ready ».

---

## 5. Questions probables du jury + éléments de réponse

**Q1 — « Comment garantissez-vous qu'il n'y a pas de fuite de données ? »**
→ Décrire les 5 points corrigés (seuils de feature engineering, imputation/scaling en pipeline, SelectKBest exploratoire non branché, sélection de modèle/scaler sur AUC de validation croisée) et reconnaître honnêtement le point encore ouvert (seuil de décision), en expliquant la correction prévue. Un jury valorise davantage une limite identifiée et expliquée qu'une affirmation de perfection non vérifiable.

**Q2 — « Pourquoi ne pas avoir utilisé la PCA ? »**
→ Moins de 25 features, interprétabilité géographique prioritaire (justification déjà présente dans le notebook, Étape 2).

**Q3 — « Pourquoi StandardScaler plutôt que MinMaxScaler ? »**
→ Les deux ont été testés systématiquement pour chaque modèle ; StandardScaler l'emporte car les distances géographiques contiennent des valeurs extrêmes réelles (sommets, zones reculées) qui compriment artificiellement les données avec MinMaxScaler (démontré par le graphique de la section 1.5).

**Q4 — « Pourquoi Recall plutôt qu'Accuracy comme métrique prioritaire ? »**
→ Contexte de prévention des risques : un faux négatif (inondation manquée) est bien plus coûteux qu'une fausse alerte. D'où le seuil « High Recall = 0.3 » proposé en complément du seuil optimal.

**Q5 — « Avez-vous vérifié une fuite géographique (points voisins en train et test) ? »**
→ Répondre honnêtement : c'est une limite identifiée, le split actuel est aléatoire stratifié sur la classe, pas sur la position géographique ; une validation croisée spatiale est proposée comme prochaine étape, éventuellement hors du périmètre du PFA actuel selon le temps disponible.

**Q6 — « Le modèle utilisé par l'interface Gradio est-il exactement celui évalué ? »**
→ Oui : le même objet pipeline (preprocessing + modèle) est sauvegardé via `joblib` puis rechargé tel quel, sans retraitement différent entre évaluation et interface.
