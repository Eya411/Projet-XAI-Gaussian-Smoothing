# Phase 8.6 — Bilan scientifique de la contribution
## 1. Contexte et objectif
Ce travail reproduit et adapte une partie de l'étude de Ye et Farnia (2025),
consacrée au compromis entre stabilité et conservation des cartes de saillance
lorsqu'un lissage gaussien de type SmoothGrad est appliqué.
L'expérience est adaptée au jeu CIFAR-10 et é l'architecture ResNet34.
Deux modèles sont entraînés sur deux sous-ensembles disjoints A et B,
pour chacune des trois graines aléatoires : 10007, 5678 et 12345.
L'objectif de la contribution est d'étudier une stratégie de sélection
adaptative du paramètre Sigma, sous une contrainte de conservation des
explications originales.
La contribution est présentée comme une stratégie expérimentale de sélection
adaptative. Elle ne prétend pas établir une nouvelle théorie générale de
la stabilité des explications.
## 2. Problématique
Le lissage par perturbations gaussiennes peut rendre les cartes de saillance
plus stables vis-à-vis des différences entre modèles entraînés sur des
échantillons distincts.
Cependant, l'augmentation de Sigma modifie progressivement les cartes
originales. Un Sigma élevé peut donc améliorer la stabilité mesurée tout en
déformant davantage les explications.
Le choix d'un Sigma fixe présente une limite : un même niveau de lissage
est appliqué é toutes les images, alors que leur sensibilité au bruit et
leur tolérance é la déformation peuvent varier.
La problématique étudiée est donc la suivante :
Comment sélectionner, pour chaque image, un niveau de lissage Sigma qui
respecte une contrainte de conservation de l'explication originale,
tout en recherchant une meilleure stabilité entre les explications
produites par deux modèles entraînés sur des sous-ensembles distincts ?
## 3. Protocole expérimental
### 3.1 Données et modèles
- Jeu de données : CIFAR-10.
- Architecture : ResNet34 adaptée aux images 32 x 32.
- Trois graines : 10007, 5678 et 12345.
- Deux sous-ensembles d'entraînement disjoints A et B par graine.
- Six modèles au total.
- Huit images de test sélectionnées pour les analyses détaillées :
  3535, 4073, 6156, 6937, 360, 2770, 1225 et 6481.
### 3.2 Méthodes d'explication
Deux méthodes sont étudiées :
- SimpleGrad.
- Integrated Gradients, avec 20 étapes et une référence nulle.
### 3.3 Lissage
Les valeurs de Sigma relatif étudiées sont :
0 ; 0.05 ; 0.10 ; 0.15 ; 0.20 ; 0.25 ; 0.30 ; 0.35 ; 0.40.
Le bruit gaussien est ajouté aux images d'entrée normalisées.
Pour chaque Sigma, les explications obtenues sur 20 perturbations sont
moyennées.
Le paramètre utilisé pour l'ajout du bruit est relié au Sigma relatif par :
Sigma_input = Sigma_relative x 2
car les entrées normalisées se situent dans l'intervalle [-1, 1].
Il s'agit donc d'une moyenne d'explications sur des entrées bruitées,
et non d'un filtre spatial appliqué directement aux cartes de saillance.
### 3.4 Mesure de stabilité
Pour une image i, une méthode m, une graine s et un Sigma donné :
S(i,m,s,Sigma) = || M_A(i,m,s,Sigma) - M_B(i,m,s,Sigma) ||_2
où M_A et M_B sont les cartes produites par les deux modèles.
Une distance plus faible indique une plus grande similarité entre les
explications A et B.
### 3.5 Mesure de conservation
Pour chaque modéle k :
P_k(i,m,s,Sigma) =
|| M_k(i,m,s,Sigma) - M_k(i,m,s,0) ||_2
La valeur utilisée pour comparer les deux modèles est :
P(i,m,s,Sigma) = (P_A + P_B) / 2
Une distance plus faible indique que la carte lissée reste plus proche
de la carte originale du même modéle.
Attention : cette mesure représente la conservation par rapport é la
carte originale. Elle ne constitue pas une mesure de fidélité causale
au comportement du modéle, telle qu'une évaluation par suppression
progressive des pixels.
## 4. Résultats de la reproduction
### 4.1 Stabilité
Figure 1 : figure_1_stabilite_sigma.png
Les résultats montrent une diminution globale de la distance de stabilité
lorsque Sigma augmente.
Pour Integrated Gradients, la moyenne passe de 5.0710 é Sigma = 0
é 2.0982 é Sigma = 0.40.
Pour SimpleGrad, elle passe de 12.4683 é Sigma = 0
é 2.0742 é Sigma = 0.40.
Ces observations indiquent que le lissage réduit la différence entre
les explications des modèles A et B dans le protocole étudié.
### 4.2 Conservation
Figure 2 : figure_2_conservation_sigma.png
La distance de conservation augmente avec Sigma.
Pour Integrated Gradients, elle passe approximativement de 0 é
3.6876 entre Sigma = 0 et Sigma = 0.40.
Pour SimpleGrad, elle passe approximativement de 0 é
9.1398 sur le même intervalle.
Ainsi, une amélioration de la stabilité mesurée s'accompagne d'une
déformation croissante des cartes originales.
Ces résultats mettent en évidence le compromis stabilité-conservation.
### 4.3 Synthése des références fixes
| Méthode | Sigma | Stabilité moyenne | Conservation moyenne |
|---|---:|---:|---:|
| Integrated Gradients | 0.00 | 5.0710 | 0.0000 |
| Integrated Gradients | 0.10 | 3.6724 | 2.2503 |
| Integrated Gradients | 0.40 | 2.0982 | 3.6876 |
| SimpleGrad | 0.00 | 12.4683 | 0.0000 |
| SimpleGrad | 0.10 | 7.1983 | 6.8378 |
| SimpleGrad | 0.40 | 2.0742 | 9.1398 |
Les valeurs sont des moyennes sur huit images et trois graines,
soit 24 combinaisons image-graine par méthode.
## 5. Contribution : sélection adaptative de Sigma
### 5.1 Principe
La contribution consiste é sélectionner un Sigma différent selon l'image,
en utilisant une contrainte de conservation.
Pour une image donnée, la règle étudiée est :
Sigma* = max { Sigma appartenant é la grille :
               P(Sigma) <= tau_P }
où tau_P représente le seuil de conservation.
La règle choisit le plus grand Sigma admissible, c'est-à-dire le niveau
de lissage le plus élevé qui respecte le seuil de conservation.
L'intérét recherché est de bénéficier d'une partie de la stabilisation
apportée par le lissage sans dépasser une limite de déformation.
### 5.2 Calibration et comparaison LOIO
Une validation Leave-One-Image-Out (LOIO) est utilisée :
- huit rotations sont réalisées ;
- é chaque rotation, une image est mise de côté ;
- les sept autres images servent é la calibration ;
- trois graines sont prises en compte ;
- quatre quantiles sont étudiés : 0.25, 0.50, 0.75 et 0.90 ;
- deux méthodes d'explication sont évaluées.
Le total est de 192 évaluations :
2 méthodes x 4 quantiles x 8 images x 3 graines.
La comparaison oppose :
1. une référence é Sigma fixe, sélectionnée sur les images de calibration ;
2. la sélection adaptative, qui choisit le plus grand Sigma admissible
   pour l'image évaluée selon la contrainte de conservation.
Le audit de traçabilité n'a pas permis d'identifier avec certitude le
code exact qui construit le seuil quantile. Les seuils et résultats LOIO
enregistrés sont conservés et les contraintes sont respectées dans les
192 évaluations, mais la formule exacte de calibration ne peut pas être
considérée comme entièrement vérifiée. Cette limite est documentée dans
le fichier phase8_3_validation_limite_calibration_LOIO.md.
### 5.3 Résultats LOIO
| Méthode | Quantile | Sigma adaptatif moyen | Sigma fixe moyen | Stabilité adaptative | Stabilité fixe | Gain moyen (%) |
|---|---:|---:|---:|---:|---:|---:|
| Integrated Gradients | 0.25 | 0.1063 | 0.0563 | 4.2332 | 4.4079 | +7.93 |
| Integrated Gradients | 0.50 | 0.1979 | 0.1500 | 3.4063 | 3.1019 | -4.20 |
| Integrated Gradients | 0.75 | 0.2979 | 0.3938 | 2.7221 | 2.1079 | -23.09 |
| Integrated Gradients | 0.90 | 0.3479 | 0.4000 | 2.4192 | 2.0982 | -10.35 |
| SimpleGrad | 0.25 | 0.1146 | 0.0500 | 9.3617 | 9.9237 | +13.54 |
| SimpleGrad | 0.50 | 0.2063 | 0.1063 | 7.4407 | 7.0238 | +7.50 |
| SimpleGrad | 0.75 | 0.2833 | 0.4000 | 5.3590 | 2.0742 | -170.34 |
| SimpleGrad | 0.90 | 0.3417 | 0.4000 | 3.7156 | 2.0742 | -71.16 |
Un gain positif correspond é une distance de stabilité plus faible
pour la sélection adaptative. Un gain négatif correspond é une
stabilité moins bonne que celle de la référence fixe.
### 5.4 Interprétation
Les résultats montrent que la sélection adaptative n'est pas
systématiquement supérieure é la référence fixe.
Pour Integrated Gradients, le quantile 0.25 donne un gain moyen positif
de 7.93 %. Les trois autres quantiles donnent des gains négatifs.
Pour SimpleGrad, les quantiles 0.25 et 0.50 donnent des gains positifs
de 13.54 % et 7.50 %. Les quantiles 0.75 et 0.90 donnent des résultats
nettement moins favorables.
La forte dispersion des gains, notamment pour SimpleGrad, montre que
les performances varient selon les images et les graines.
Le choix du plus grand Sigma admissible peut conduire é un lissage
trop important pour la stabilité recherchée. Le respect d'une contrainte
de conservation ne garantit donc pas, é lui seul, une meilleure stabilité.
### 5.5 Figure de comparaison
Figure 3 : figure_3_gain_adaptatif_LOIO.png
Cette figure présente les gains de stabilité moyens et leur dispersion
selon le quantile, pour les deux méthodes.
Elle permet de constater que les résultats positifs se concentrent
principalement sur les quantiles faibles et intermédiaires.
## 6. Analyse des ablations descriptives
Les configurations comparées sont :
- Sigma = 0 : absence de lissage ;
- Sigma = 0.10 : lissage fixe modéré ;
- Sigma = 0.40 : lissage fixe élevé ;
- Sigma fixe LOIO ;
- Sigma adaptatif LOIO.
Les références fixes illustrent le compromis entre stabilité et
conservation. Les configurations LOIO permettent d'examiner si une
sélection par image peut améliorer les résultats par rapport é une
valeur fixe calibrée.
Cette analyse est descriptive. Elle ne constitue pas une ablation
complète de tous les composants de la méthode.
Aucune variante fondée uniquement sur la stabilité ni aucune règle
supplémentaire de type acceptation, remplacement ou rejet n'a été
implémentée dans cette premiére expérience.
## 7. Comparaison avec l'état de l'art
### 7.1 Ye et Farnia (2025)
Ye et Farnia étudient l'effet du lissage gaussien de type SmoothGrad
sur les cartes SimpleGrad et Integrated Gradients.
Leur travail analyse théoriquement et expérimentalement le compromis
entre stabilité vis-à-vis de la variation des données d'entraînement
et fidélité des cartes lissées aux cartes originales.
Notre reproduction adaptée retrouve qualitativement ce compromis :
l'augmentation de Sigma réduit la distance entre les cartes A et B,
mais augmente la distance entre les cartes lissées et originales.
Notre contribution exploratoire se distingue par l'étude d'une règle
de sélection adaptative de Sigma sous contrainte de conservation,
évaluée en Leave-One-Image-Out.
Les résultats numériques ne sont pas directement comparables é ceux
de l'article, car les jeux de données, les modèles, les tailles
d'échantillons, les graines et les paramètres expérimentaux différent.
### 7.2 évaluations comparatives récentes
Des travaux comparatifs récents sur les explications d'images montrent
qu'une méthode XAI peut obtenir de bons résultats sur certains critères
sans étre la meilleure sur tous les autres.
Cela soutient l'intérét d'une évaluation multidimensionnelle et invite
é ne pas réduire la qualité d'une explication é une seule mesure.
Notre expérience reste plus ciblée : elle étudie deux distances
particulières liées à la stabilité inter-modèles et à la conservation
par rapport aux cartes originales.
## 8. Limites de l'étude
1. Le nombre d'images utilisées dans l'analyse détaillée est limité é huit.
2. Les huit images sont réutilisées dans les rotations LOIO.
3. Les 192 évaluations ne constituent donc pas 192 observations indépendantes.
4. Les trois graines ne correspondent pas é trois nouveaux jeux d'images
   indépendants.
5. La dispersion des gains adaptatifs est parfois élevée.
6. La règle choisit Sigma é partir de la conservation, sans optimiser
   directement la stabilité sur l'image évaluée.
7. La référence fixe et la règle adaptative n'utilisent pas exactement
   le même objectif de sélection : la premiére minimise une stabilité
   moyenne sous une contrainte de calibration, tandis que la seconde
   choisit le plus grand Sigma admissible par image.
8. Le nombre de perturbations utilisé dans notre adaptation est de 20,
   et non celui du protocole de référence lorsque celui-ci utilise
   un nombre différent.
9. La mesure de conservation est une distance é la carte originale,
   pas une mesure de fidélité causale au modéle.
10. Aucune évaluation humaine ou applicationnelle n'a été réalisée.
## 9. Perspectives
Plusieurs améliorations pourraient étre étudiées :
- augmenter le nombre d'images de test ;
- répéter l'expérience avec davantage de graines ;
- étudier d'autres architectures et jeux de données ;
- comparer différentes règles de sélection adaptative ;
- ajouter une mesure indépendante de fidélité fonctionnelle,
  par exemple une évaluation par suppression ou insertion ;
- étudier la sensibilité aux seuils de conservation ;
- examiner le coût de calcul de la sélection adaptative ;
- évaluer la qualité perceptuelle des explications auprès d'utilisateurs.
Ces perspectives ne sont pas présentées comme des expériences déjà réalisées.
## 10. Conclusion générale
Cette expérience adaptée confirme le compromis entre stabilité et
conservation des cartes de saillance dans le cadre étudié.
L'augmentation de Sigma rend généralement les explications des modèles
A et B plus similaires, mais éloigne les cartes lissées de leurs
références originales.
La sélection adaptative sous contrainte de conservation constitue une
piste pour éviter l'utilisation systématique d'un Sigma unique.
Les résultats LOIO sont encourageants pour certains quantiles,
notamment Q25, mais ils montrent aussi des dégradations importantes
dans d'autres configurations.
La contribution doit donc étre interprétée comme une premiére stratégie
expérimentale de sélection adaptative, dont l'efficacité dépend de la
méthode d'explication et du seuil retenu.
Elle ne démontre pas une supériorité générale sur les méthodes é Sigma
fixe, mais fournit une analyse mesurée du compromis et de ses limites.
La formule exacte de calibration du seuil LOIO n'ayant pas pu être
retracée, la reproductibilité complète de cette étape adaptative reste
une limite de l'étude.
## 11. Références bibliographiques
[1] Z. Ye and F. Farnia, "Gaussian Smoothing in Saliency Maps:
The Stability-Fidelity Trade-Off in Neural Network Interpretability",
Proceedings of the 28th International Conference on Artificial
Intelligence and Statistics, Proceedings of Machine Learning Research,
vol. 258, pp. 2125-2133, 2025.
https://proceedings.mlr.press/v258/ye25a.html
[2] M. Skliarov et al., "A Comparative Evaluation of Explainability
Techniques for Image Data", Scientific Reports, vol. 15, article 41898,
2025.
https://www.nature.com/articles/s41598-025-25839-y
## 12. Fichiers expérimentaux associés
- Figure 1 : figures/figure_1_stabilite_sigma.png
- Figure 2 : figures/figure_2_conservation_sigma.png
- Figure 3 : figures/figure_3_gain_adaptatif_LOIO.png
- Comparaison consolidée : phase8_4_comparaison_consolidee.csv
- Bilan des ablations descriptives : phase8_5_ablations.csv
- Problématique et état de l'art : phase8_1_problematique_etat_art.md
- Pseudocode : phase8_2_pseudocode.md
- Validation LOIO et limite de traçabilité : phase8_3_validation_limite_calibration_LOIO.md
