PROJET XAI – MASTER 2 SPI
========================
TITRE DU PROJET
---------------
Gaussian Smoothing in Saliency Maps:
The Stability-Fidelity Trade-Off in Neural Network Interpretability.
Ce projet constitue une reproduction adaptee et une etude experimentale
du travail de Ye & Farnia (2025), portant sur le lissage gaussien
des cartes de saillance et le compromis entre stabilite et fidelite
des explications.
1. CONFIGURATION EXPERIMENTALE
------------------------------
- Dataset : CIFAR-10
- Architecture : ResNet34 adaptee aux images 32 x 32
- Entrainement : depuis zero
- Nombre d'echantillons d'entrainement : 8 000
- Seeds : 10007, 12345 et 5678
- Modeles : deux modeles independants A et B par seed
- Nombre total de modeles : 6
- Methodes XAI : SimpleGrad et Integrated Gradients
- Integrated Gradients : 20 etapes, baseline nulle
- Lissage : perturbations gaussiennes des entrees
Les evaluations comparatives utilisent un ensemble de test commun.
2. TRAVAUX ET EXPERIMENTATIONS
------------------------------
Le projet comprend les travaux suivants :
- Preparation et audit des donnees.
- Creation des sous-ensembles experimentaux.
- Entrainement des modeles ResNet34.
- Verification des checkpoints.
- Evaluation des modeles.
- Generation des explications originales.
- Analyse de la stabilite des explications sous perturbations gaussiennes.
- Analyse de la conservation des explications apres lissage.
- Comparaison de differentes valeurs de sigma.
- Exploration d'une strategie adaptative de selection de sigma.
- Etudes comparatives, ablations et analyse des limites.
3. CONTRIBUTION EXPERIMENTALE
-----------------------------
Une strategie adaptative de selection de sigma est exploree afin
d'etudier la possibilite de mieux equilibrer la stabilite et la
conservation des explications.
Cette strategie est examinee a travers des comparaisons avec des
configurations de lissage fixes, ainsi que des analyses d'ablation
et une evaluation LOIO.
Les resultats sont interpretes avec prudence. Ils ne permettent pas
de conclure a une superiorite generale de la strategie adaptative.
Les observations dependent notamment du protocole, des modeles,
des methodes d'explication et des echantillons analyses.
4. ORGANISATION DU DOSSIER
--------------------------
01_Code/
    Scripts et outils utilises pour les experimentations.
02_Configuration/
    Fichiers de configuration du projet.
03_Modeles/
    Organisation des modeles et checkpoints locaux.
04_Resultats_Entrainement/
    Resultats et fichiers associes a l'entrainement.
05_Explications_XAI/
    Explications generees par les methodes XAI.
06_Stabilite/
    Resultats et analyses de stabilite.
07_Graphiques/
    Figures et visualisations experimentales.
08_Documentation/
    Documents et rapports relatifs au projet.
5. REMARQUES SCIENTIFIQUES
--------------------------
Cette implementation est une adaptation experimentale du protocole
etudie et non une reproduction strictement identique de tous les
parametres de l'article original.
Les resultats doivent etre interpretes dans les limites du protocole
experimental adopte. Les conclusions ne doivent pas etre generalisees
au-dela des experiences effectivement realisees.
Projet academique – Master 2
Systemes Pervasifs Intelligents (SPI)
Annee universitaire 2026-2027
