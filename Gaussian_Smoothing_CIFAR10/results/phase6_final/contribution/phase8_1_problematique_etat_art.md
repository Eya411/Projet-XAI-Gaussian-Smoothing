# Phase 8.1 ? Probl?matique et ?tat de l'art
## 1. ?tat de l'art
### Ye et Farnia (2025)
Zhuorui Ye et Farzan Farnia, *Gaussian Smoothing in Saliency Maps: The Stability-Fidelity Trade-Off in Neural Network Interpretability*, Proceedings of the 28th International Conference on Artificial Intelligence and Statistics, PMLR 258, 2025, p. 2125?2133.
Ce travail ?tudie l'effet du lissage gaussien sur la stabilit? des cartes SimpleGrad et Integrated Gradients. Il met en ?vidence un compromis : le lissage peut am?liorer la stabilit?, mais un lissage plus intense peut r?duire la fid?lit? des explications originales.
Lien : https://proceedings.mlr.press/v258/ye25a.html
### A Comparative Evaluation of Explainability Techniques for Image Data (2025)
Cette ?tude compare plusieurs m?thodes d'explication sur plusieurs jeux de donn?es et architectures, ? l'aide de crit?res tels que la fid?lit?, la stabilit? et le co?t de calcul. Ses r?sultats soulignent qu'aucune m?thode ne domine syst?matiquement tous les crit?res.
Lien : https://www.nature.com/articles/s41598-025-25839-y
### Evaluating Perceptual and Semantic Interpretability of Saliency Methods (2022)
Cette ?tude examine plusieurs dimensions de la qualit? des cartes de saillance et pr?sente une s?lection adaptative de m?thodes d'explication selon plusieurs crit?res. Elle montre que la s?lection adaptative de cartes d'explication existe d?j? dans la litt?rature. Notre travail se distingue par son objectif : s?lectionner une valeur de Sigma pour une m?thode d'explication donn?e, sous une contrainte de conservation.
Lien : https://doi.org/10.1002/ail2.77
## 2. Limite ?tudi?e
Le choix d'une valeur fixe de Sigma ne tient pas n?cessairement compte des diff?rences entre les images. Une valeur de Sigma peut produire un compromis satisfaisant pour certaines images, mais entra?ner une d?formation excessive des explications pour d'autres.
## 3. Probl?matique
Les m?thodes d'explication fond?es sur les gradients peuvent produire des cartes sensibles aux variations des donn?es d'entra?nement et aux perturbations des entr?es. Le lissage gaussien permet d'am?liorer leur stabilit?, mais une augmentation de son intensit? peut ?galement d?former les cartes d'explication originales. Le choix d'une valeur fixe de Sigma ne tient pas n?cessairement compte des diff?rences entre les images et peut donc conduire ? un compromis inadapt? selon l'exemple consid?r?.
Dans ce travail, nous ?tudions une strat?gie de s?lection adaptative de Sigma, fond?e sur une contrainte de conservation des explications originales. L'objectif est de choisir, pour chaque image, la plus grande valeur de Sigma respectant un seuil de conservation calibr? sur des images distinctes de l'image ?valu?e. Nous analysons ensuite l'effet de cette s?lection sur la stabilit? des explications produites par deux mod?les entra?n?s sur des sous-ensembles distincts.
Cette approche vise ? ?tudier si une s?lection individualis?e du niveau de lissage permet de mieux contr?ler le compromis entre stabilit? et conservation, sans supposer qu'elle am?liore syst?matiquement les deux crit?res.
## 4. Positionnement de la contribution
La contribution ne consiste pas ? introduire le lissage gaussien ni le principe g?n?ral de s?lection adaptative des explications. Elle consiste ? ?tudier et ?valuer une r?gle de s?lection de Sigma, image par image, qui choisit la plus grande valeur admissible sous une contrainte de conservation calibr?e sur les autres images.
Les r?sultats sont ?valu?s avec une proc?dure Leave-One-Image-Out (LOIO). Cette validation reste exploratoire en raison du nombre limit? de huit images et du chevauchement entre les ensembles de calibration des diff?rentes rotations.
