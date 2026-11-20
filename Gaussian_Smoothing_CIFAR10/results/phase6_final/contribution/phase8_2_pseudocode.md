# Phase 8.2 ? Pseudocode d?finitif de la s?lection adaptative
## 1. Nom de la m?thode
S?lection adaptative de Sigma sous contrainte de conservation.
## 2. Entr?es
- Deux mod?les entra?n?s sur des sous-ensembles distincts : A et B.
- Une image ? expliquer : x_i.
- Une m?thode d'explication : SimpleGrad ou Integrated Gradients.
- Une grille de valeurs :
  Sigma = {0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40}.
- Un ensemble de calibration contenant les images autres que l'image tenue ? l'?cart.
- Un quantile q parmi {0.25, 0.50, 0.75, 0.90}.
## 3. Crit?re de conservation
Pour chaque image i, m?thode m, seed s et valeur sigma, la conservation est la moyenne des distances L2 entre les cartes liss?es et les cartes originales des deux mod?les :
P(i,m,s,sigma) =
  [P_A(i,m,s,sigma) + P_B(i,m,s,sigma)] / 2
avec :
P_A = ||M_A(sigma) - M_A(0)||_2
P_B = ||M_B(sigma) - M_B(0)||_2
Une faible distance indique une meilleure conservation de la carte originale.
## 4. Crit?re de stabilit?
La stabilit? est mesur?e par la distance L2 entre les cartes des deux mod?les :
S(i,m,s,sigma) =
  ||M_A(sigma) - M_B(sigma)||_2
Une faible distance indique une meilleure stabilit? entre les explications des deux mod?les.
## 5. Algorithme de calibration et de s?lection
POUR chaque m?thode m :
    POUR chaque quantile q :
        POUR chaque image i tenue ? l'?cart :
            1. D?finir l'ensemble de calibration :
               les sept autres images.
            2. Calculer le seuil de conservation tau :
               quantile q des valeurs de conservation
               obtenues sur l'ensemble de calibration,
               selon le protocole exp?rimental utilis?.
            3. POUR chaque seed s :
                a. Examiner les valeurs de Sigma de la grille.
                b. Pour chaque Sigma, utiliser la conservation
                   de l'image tenue ? l'?cart.
                c. Construire l'ensemble admissible :
                   Sigma_admissible =
                   {sigma : P(i,m,s,sigma) <= tau}
                d. Si l'ensemble est non vide :
                   s?lectionner le plus grand Sigma admissible.
                   sigma_adaptatif =
                   max(Sigma_admissible)
                e. Sinon :
                   signaler qu'aucune valeur n'est admissible.
                   Ne pas inventer une valeur de remplacement.
                f. Apr?s s?lection, mesurer la stabilit?
                   et la conservation au Sigma s?lectionn?.
            4. Enregistrer le seuil, le Sigma s?lectionn?,
               la stabilit? et la conservation.
## 6. R?gle math?matique de s?lection
sigma_i* = max { sigma dans Sigma :
                 P(i,m,s,sigma) <= tau_P }
Cette r?gle choisit le plus grand niveau de lissage qui respecte
la contrainte de conservation.
## 7. Validation Leave-One-Image-Out
R?p?ter la proc?dure pour chacune des huit images :
- 7 images pour la calibration ;
- 1 image tenue ? l'?cart pour l'?valuation ;
- 8 rotations au total.
R?p?ter les ?valuations avec les trois seeds :
5678, 10007 et 12345.
Les r?sultats comprennent :
2 m?thodes x 4 quantiles x 8 images x 3 seeds = 192 ?valuations.
## 8. Comparaison avec la r?f?rence fixe
La s?lection adaptative est compar?e ? une configuration fixe
calibr?e sur les sept images d'apprentissage.
La r?f?rence fixe utilis?e dans l'exp?rience LOIO est choisie
en minimisant la stabilit? moyenne parmi les valeurs fixes
respectant le seuil moyen de conservation sur les images
de calibration.
La s?lection adaptative et la r?f?rence fixe sont ensuite
?valu?es sur la m?me image tenue ? l'?cart.
## 9. Pr?cautions d'interpr?tation
- La stabilit? n'intervient pas dans la r?gle de s?lection adaptative.
- Le seuil de conservation est calibr? sans utiliser l'image tenue ? l'?cart.
- Les huit rotations r?utilisent les m?mes images et leurs ensembles
  de calibration se chevauchent.
- Les trois seeds ne correspondent pas ? trois jeux d'images ind?pendants.
- Les 192 ?valuations ne doivent pas ?tre consid?r?es comme
  192 observations ind?pendantes.
- La m?thode garantit uniquement le respect de la contrainte
  de conservation dans les ?valuations o? une valeur admissible existe.
- Elle ne garantit pas une am?lioration de la stabilit?.
- Le quantile et sa m?thode exacte d'agr?gation doivent rester
  conformes ? l'impl?mentation exp?rimentale effectivement utilis?e.
## 10. Port?e de la contribution
Cette m?thode constitue une strat?gie de s?lection adaptative
du niveau de lissage sous contrainte de conservation.
Elle est ?valu?e exp?rimentalement sur CIFAR-10, avec ResNet34,
SimpleGrad et Integrated Gradients.
Les r?sultats sont exploratoires et limit?s par le nombre d'images.
