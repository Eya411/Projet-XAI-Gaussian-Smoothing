# Phase 8.5 ? Bilan des ablations

## Objectif

?tudier l'effet du niveau de lissage et l'int?r?t de la s?lection adaptative,
en r?utilisant les exp?riences de stabilit?, de conservation et LOIO d?j? r?alis?es.
Aucun nouvel entra?nement ni calcul d'explications n'est effectu?.

## Configurations compar?es

- Sigma = 0 : absence de lissage, r?f?rence initiale.
- Sigma = 0.1 : lissage fixe mod?r?.
- Sigma = 0.4 : lissage fixe ?lev?.
- Sigma fixe LOIO : r?f?rence fixe s?lectionn?e sur les images de calibration.
- Sigma adaptatif LOIO : s?lection du plus grand Sigma respectant le seuil de conservation.

## R?sultats

             method        configuration  quantile  mean_sigma  mean_stability  std_stability  mean_conservation  std_conservation  mean_gain_percent
IntegratedGradients      Sigma fixe 0.00       NaN      0.0000          5.0710         1.5072             0.0000            0.0000                NaN
IntegratedGradients      Sigma fixe 0.10       NaN      0.1000          3.6724         0.9596             2.2503            0.7983                NaN
IntegratedGradients      Sigma fixe 0.40       NaN      0.4000          2.0982         0.7036             3.6876            1.1294                NaN
IntegratedGradients Sigma adaptatif LOIO      0.25      0.1062          4.2332         1.8936             1.4966            0.5400             7.9314
IntegratedGradients      Sigma fixe LOIO      0.25      0.0562          4.4079         1.2847             1.3729            0.4469                NaN
IntegratedGradients Sigma adaptatif LOIO      0.50      0.1979          3.4063         1.8853             2.3740            0.6537            -4.2043
IntegratedGradients      Sigma fixe LOIO      0.50      0.1500          3.1019         0.8001             2.8218            0.9273                NaN
IntegratedGradients Sigma adaptatif LOIO      0.75      0.2979          2.7221         1.6055             3.1169            0.5934           -23.0894
IntegratedGradients      Sigma fixe LOIO      0.75      0.3938          2.1079         0.6945             3.6809            1.1407                NaN
IntegratedGradients Sigma adaptatif LOIO      0.90      0.3479          2.4192         1.4368             3.3918            0.7313           -10.3540
IntegratedGradients      Sigma fixe LOIO      0.90      0.4000          2.0982         0.7036             3.6876            1.1294                NaN
         SimpleGrad      Sigma fixe 0.00       NaN      0.0000         12.4683         3.9103             0.0000            0.0000                NaN
         SimpleGrad      Sigma fixe 0.10       NaN      0.1000          7.1983         1.7467             6.8378            2.4474                NaN
         SimpleGrad      Sigma fixe 0.40       NaN      0.4000          2.0742         0.5540             9.1398            2.9645                NaN
         SimpleGrad Sigma adaptatif LOIO      0.25      0.1146          9.3617         5.7815             3.7687            2.3086            13.5413
         SimpleGrad      Sigma fixe LOIO      0.25      0.0500          9.9237         2.8510             4.5311            1.7361                NaN
         SimpleGrad Sigma adaptatif LOIO      0.50      0.2062          7.4407         6.3581             5.5785            2.3400             7.4955
         SimpleGrad      Sigma fixe LOIO      0.50      0.1062          7.0238         1.8490             6.9788            2.3519                NaN
         SimpleGrad Sigma adaptatif LOIO      0.75      0.2833          5.3590         5.5950             7.3143            2.0895          -170.3381
         SimpleGrad      Sigma fixe LOIO      0.75      0.4000          2.0742         0.5540             9.1398            2.9645                NaN
         SimpleGrad Sigma adaptatif LOIO      0.90      0.3417          3.7156         4.1462             8.3400            1.9410           -71.1611
         SimpleGrad      Sigma fixe LOIO      0.90      0.4000          2.0742         0.5540             9.1398            2.9645                NaN

## Analyse

1. Sans lissage (Sigma = 0), la conservation est maximale au sens de la distance utilis?e : la carte reste identique ? sa r?f?rence. En revanche, la stabilit? entre A et B est moins bonne.
2. Lorsque Sigma augmente, la distance de stabilit? diminue g?n?ralement, mais la distance de conservation augmente : le lissage stabilise les cartes au prix d'une d?formation.
3. Sigma = 0.4 illustre clairement ce compromis : la stabilit? moyenne est faible, mais la d?formation des cartes est plus importante.
4. La s?lection adaptative am?liore parfois la stabilit? par rapport ? la r?f?rence fixe LOIO, mais pas pour tous les quantiles.
5. Les quantiles ?lev?s ne donnent pas de r?sultat syst?matiquement favorable. Le plus grand Sigma admissible peut entra?ner une d?gradation de la stabilit?.

## Ablations non r?alis?es

- Une variante s?lectionnant Sigma uniquement selon la stabilit? n'a pas ?t? impl?ment?e.
- Une variante combinant une contrainte de stabilit? suppl?mentaire n'a pas ?t? impl?ment?e.
- Ces variantes n?cessiteraient de nouvelles r?gles de s?lection et des exp?riences suppl?mentaires. Elles ne sont donc pas pr?sent?es comme des r?sultats.

## Conclusion

Cette ablation descriptive met en ?vidence l'effet du niveau de lissage et permet de situer la s?lection adaptative par rapport aux r?f?rences fixes.
Elle ne d?montre pas que la m?thode adaptative est syst?matiquement sup?rieure.
La contribution doit ?tre pr?sent?e comme une strat?gie exploratoire de s?lection sous contrainte de conservation, dont les performances d?pendent de la m?thode d'explication et du quantile choisi.

Les distances de conservation mesurent ici la d?formation par rapport ? la carte originale ; elles ne constituent pas une mesure de fid?lit? causale au mod?le.