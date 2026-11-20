# Gaussian Smoothing in Saliency Maps
## Étude du compromis entre stabilité et conservation des explications
**Projet de Master 2 – Systèmes Pervasifs Intelligents (SPI)**
Ce projet constitue une reproduction adaptée et une étude expérimentale du travail de Ye & Farnia (2025) consacré au lissage gaussien des cartes de saillance et au compromis entre stabilité et conservation des explications.
## 1. Objectifs
Les méthodes d'Explainable AI (XAI) produisent des cartes de saillance destinées à mettre en évidence les régions d'une image qui contribuent à une prédiction du réseau neuronal.
Ces cartes peuvent être sensibles aux perturbations de l'entrée. Le lissage gaussien est étudié comme moyen de réduire cette sensibilité.
L'objectif du projet est d'étudier deux aspects complémentaires :
- **Stabilité :** observer l'évolution des explications lorsque les entrées sont perturbées par du bruit gaussien.
- **Conservation :** mesurer dans quelle mesure le lissage préserve l'explication originale.
Le projet explore également une stratégie adaptative de sélection du paramètre sigma.
## 2. Protocole expérimental
| Élément | Configuration |
|---|---|
| Dataset | CIFAR-10 |
| Architecture | ResNet34 adaptée aux images 32 × 32 |
| Entraînement | Depuis zéro |
| Échantillons d'entraînement | 8 000 |
| Seeds | 10007, 12345, 5678 |
| Modèles | Deux modèles indépendants A et B par seed |
| Nombre total de modèles | 6 |
| Méthodes XAI | SimpleGrad et Integrated Gradients |
| Integrated Gradients | 20 étapes, baseline nulle |
| Lissage | Perturbations gaussiennes des entrées |
Les modèles A et B sont entraînés sur des sous-ensembles distincts selon le protocole expérimental adopté. Les évaluations comparatives utilisent un ensemble de test commun.
## 3. Expériences réalisées
Le projet comprend plusieurs étapes :
1. Préparation et vérification des données.
2. Entraînement des modèles ResNet34.
3. Vérification des checkpoints et évaluation des modèles.
4. Génération des explications originales.
5. Étude de la stabilité des explications sous perturbations gaussiennes.
6. Étude de la conservation des explications après lissage.
7. Comparaison des valeurs de sigma.
8. Exploration d'une stratégie adaptative et analyse de ses résultats.
## 4. Contribution expérimentale
Une stratégie adaptative de sélection de sigma est explorée afin d'étudier la possibilité d'ajuster le niveau de lissage selon les observations expérimentales.
Cette étude comprend notamment :
- l'analyse de la stabilité en fonction de sigma ;
- l'analyse de la conservation de l'explication originale ;
- une comparaison entre stratégie fixe et stratégie adaptative ;
- une analyse exploratoire Leave-One-Image-Out (LOIO) ;
- des expériences d'ablation et une discussion des limites.
Cette contribution est présentée comme une exploration expérimentale. Les résultats disponibles ne permettent pas de conclure à une supériorité générale de la stratégie adaptative.
## 5. Organisation du dépôt
```text
Projet_XAI/
│
├── Gaussian_Smoothing_CIFAR10/
│   ├── results/
│   │   ├── official_protocol_test/
│   │   ├── phase5_evaluation/
│   │   ├── phase6_final/
│   │   │   └── contribution/
│   │   └── plots/
│   │
│   ├── train_cifar10.py
│   ├── config.py
│   ├── xai_config.py
│   ├── generate_stability_safe.py
│   ├── generate_fidelity_safe.py
│   └── ...
│
├── Presentation_XAI/
│   ├── 01_Code/
│   ├── 02_Configuration/
│   ├── 03_Modeles/
│   ├── 04_Resultats_Entrainement/
│   ├── 05_Explications_XAI/
│   ├── 06_Stabilite/
│   ├── 07_Graphiques/
│   ├── 08_Documentation/
│   └── README_PRESENTATION.txt
│
├── check_environment.py
├── inspect_checkpoints.py
└── README.md
```
## 6. Résultats et livrables
Les résultats expérimentaux sont organisés dans le dossier `Gaussian_Smoothing_CIFAR10/results/`.
On y trouve notamment :
- les résultats d'évaluation des modèles ;
- les fichiers CSV des expériences de stabilité ;
- les fichiers CSV des expériences de conservation ;
- les graphiques comparatifs ;
- les analyses de la contribution adaptative ;
- les configurations expérimentales et les bilans scientifiques.
Les fichiers CSV permettent de retrouver les données numériques utilisées pour produire les graphiques.
## 7. Installation et exécution
Le projet a été développé avec Python et PyTorch.
Créer un environnement virtuel :
```bash
python -m venv .venv
```
Activer l'environnement sous Windows :
```powershell
.venv\Scripts\Activate.ps1
```
Installer les dépendances nécessaires à partir de l'environnement documenté pour le projet, puis adapter les chemins des données et des sorties dans les fichiers de configuration avant toute exécution.
Les scripts doivent être exécutés depuis le dossier `Gaussian_Smoothing_CIFAR10`.
## 8. Limites expérimentales
Cette étude est une adaptation expérimentale du protocole de référence et ne constitue pas une reproduction strictement identique de tous ses paramètres.
Les résultats dépendent notamment :
- du sous-ensemble de données utilisé ;
- des seeds et des modèles entraînés ;
- des méthodes d'explication retenues ;
- du nombre d'images utilisé pour certaines analyses ;
- des paramètres de perturbation et de lissage.
L'analyse LOIO repose sur un échantillon limité. Les conclusions doivent donc être interprétées avec prudence et ne doivent pas être généralisées sans validation supplémentaire.
## 9. Référence scientifique
Ye & Farnia (2025).
*Gaussian Smoothing in Saliency Maps: The Stability-Fidelity Trade-Off in Neural Network Interpretability.*
Les informations bibliographiques complètes et le lien vers la publication seront ajoutés après vérification de la référence originale.
---
**Projet académique – Master 2 SPI – 2026/2027**
