# Phase 8.3 — Validation de la calibration LOIO et limites de traçabilité
## 1. Objectif
Cette phase examine la traçabilité du seuil de conservation utilisé par la sélection adaptative dans l'expérience Leave-One-Image-Out (LOIO).
L'objectif est de vérifier si la formule exacte de calcul du seuil quantile peut être reconstruite à partir des scripts et des fichiers de résultats disponibles.
## 2. Éléments expérimentaux conservés
Les fichiers de résultats LOIO contiennent :
- 2 méthodes : IntegratedGradients et SimpleGrad ;
- 4 quantiles : Q25, Q50, Q75 et Q90 ;
- 8 rotations, chacune mettant une image de côté ;
- 3 graines : 5678, 10007 et 12345 ;
- 192 évaluations au total.
Le fichier `leave_one_image_out_per_image.csv` contient 192 lignes et enregistre notamment le seuil, le Sigma adaptatif, la stabilité et la conservation.
Les résultats existants ont été conservés sans modification.
## 3. Vérifications effectuées
La recherche du code source a porté sur les fichiers de code et de documentation disponibles dans le projet. Aucun script générateur LOIO permettant de retrouver directement la formule d'agrégation du seuil n'a été identifié.
Plusieurs calculs candidats ont ensuite été comparés aux seuils enregistrés, notamment :
- le quantile des moyennes de conservation par image ;
- la médiane des moyennes par image ;
- la moyenne des quantiles calculés séparément pour chaque Sigma ;
- le quantile des maxima et des médianes par image ;
- des calculs utilisant séparément les mesures des modèles A et B ;
- le quantile global des mesures A/B.
Ces vérifications n'ont pas permis de retrouver exactement la formule d'origine.
Pour IntegratedGradients, Q25 et l'image 360 tenue à l'écart, le seuil enregistré est 2.045515. La moyenne des Q25 calculés séparément pour chaque Sigma à partir des moyennes A/B vaut 2.073139. La variante utilisant les mesures A/B séparément donne 2.085943. Ces valeurs sont proches, mais ne correspondent pas exactement au seuil enregistré.
La comparaison de la moyenne des Q25 par Sigma sur les huit rotations donne également des écarts non nuls. Cette formule candidate ne permet donc pas d'expliquer les seuils enregistrés.
## 4. Conclusion sur la traçabilité
La formule exacte d'agrégation utilisée pour construire les seuils de calibration LOIO demeure non vérifiée.
Cette limite ne démontre pas que les résultats enregistrés sont erronés. Elle signifie que leur procédure exacte de génération ne peut pas être reproduite de manière certaine à partir des éléments actuellement disponibles.
Il serait scientifiquement incorrect de présenter l'une des formules candidates comme la formule réellement utilisée sans retrouver le code source correspondant.
## 5. Décision méthodologique
Les décisions suivantes sont retenues :
1. Conserver les 192 évaluations LOIO et les fichiers de résultats existants.
2. Ne pas modifier les seuils enregistrés.
3. Ne pas prétendre que la formule exacte du quantile a été vérifiée.
4. Présenter la comparaison adaptative/fixe comme une analyse exploratoire soumise à cette limite de traçabilité.
5. Mentionner explicitement cette limite dans le rapport final.
6. Ne pas relancer les entraînements ni les générations d'explications pour cette seule raison.
## 6. Conséquence pour l'interprétation
Les résultats LOIO restent descriptivement exploitables pour examiner les tendances observées dans les fichiers enregistrés. Toutefois, la reproductibilité exacte de la calibration adaptative n'est pas établie.
La contribution doit donc être présentée comme une étude exploratoire d'une sélection de Sigma sous contrainte de conservation, et non comme une méthode entièrement reproductible dont tous les détails d'implémentation auraient été vérifiés.
