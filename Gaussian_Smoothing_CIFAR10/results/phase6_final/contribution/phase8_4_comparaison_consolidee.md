# Phase 8.4 ? Comparaison consolid?e

Les r?f?rences Sigma fixes 0, 0.1 et 0.4 sont calcul?es sur les m?mes 8 images et 3 seeds. Les configurations LOIO sont calcul?es sur 24 ?valuations par m?thode et quantile.

             method        configuration  quantile  mean_sigma  mean_stability  std_stability  mean_conservation  std_conservation  mean_gain_percent  std_gain_percent  nb_evaluations
IntegratedGradients      Sigma fixe 0.00       NaN      0.0000          5.0710         1.5072             0.0000            0.0000                NaN               NaN              24
IntegratedGradients      Sigma fixe 0.10       NaN      0.1000          3.6724         0.9596             2.2503            0.7983                NaN               NaN              24
IntegratedGradients      Sigma fixe 0.40       NaN      0.4000          2.0982         0.7036             3.6876            1.1294                NaN               NaN              24
         SimpleGrad      Sigma fixe 0.00       NaN      0.0000         12.4683         3.9103             0.0000            0.0000                NaN               NaN              24
         SimpleGrad      Sigma fixe 0.10       NaN      0.1000          7.1983         1.7467             6.8378            2.4474                NaN               NaN              24
         SimpleGrad      Sigma fixe 0.40       NaN      0.4000          2.0742         0.5540             9.1398            2.9645                NaN               NaN              24
IntegratedGradients      Sigma fixe LOIO      0.25      0.0562          4.4079         1.2847             1.3729            0.4469                NaN               NaN              24
IntegratedGradients Sigma adaptatif LOIO      0.25      0.1062          4.2332         1.8936             1.4966            0.5400             7.9314           17.0772              24
IntegratedGradients      Sigma fixe LOIO      0.50      0.1500          3.1019         0.8001             2.8218            0.9273                NaN               NaN              24
IntegratedGradients Sigma adaptatif LOIO      0.50      0.1979          3.4063         1.8853             2.3740            0.6537            -4.2043           35.4271              24
IntegratedGradients      Sigma fixe LOIO      0.75      0.3938          2.1079         0.6945             3.6809            1.1407                NaN               NaN              24
IntegratedGradients Sigma adaptatif LOIO      0.75      0.2979          2.7221         1.6055             3.1169            0.5934           -23.0894           36.7981              24
IntegratedGradients      Sigma fixe LOIO      0.90      0.4000          2.0982         0.7036             3.6876            1.1294                NaN               NaN              24
IntegratedGradients Sigma adaptatif LOIO      0.90      0.3479          2.4192         1.4368             3.3918            0.7313           -10.3540           26.1984              24
         SimpleGrad      Sigma fixe LOIO      0.25      0.0500          9.9237         2.8510             4.5311            1.7361                NaN               NaN              24
         SimpleGrad Sigma adaptatif LOIO      0.25      0.1146          9.3617         5.7815             3.7687            2.3086            13.5413           34.9397              24
         SimpleGrad      Sigma fixe LOIO      0.50      0.1062          7.0238         1.8490             6.9788            2.3519                NaN               NaN              24
         SimpleGrad Sigma adaptatif LOIO      0.50      0.2062          7.4407         6.3581             5.5785            2.3400             7.4955           58.3707              24
         SimpleGrad      Sigma fixe LOIO      0.75      0.4000          2.0742         0.5540             9.1398            2.9645                NaN               NaN              24
         SimpleGrad Sigma adaptatif LOIO      0.75      0.2833          5.3590         5.5950             7.3143            2.0895          -170.3381          287.6975              24
         SimpleGrad      Sigma fixe LOIO      0.90      0.4000          2.0742         0.5540             9.1398            2.9645                NaN               NaN              24
         SimpleGrad Sigma adaptatif LOIO      0.90      0.3417          3.7156         4.1462             8.3400            1.9410           -71.1611          165.9445              24

## Interpr?tation des mesures

- Stabilit? : distance L2 entre les cartes A et B ; plus faible = meilleure stabilit?.
- Conservation : distance L2 entre les cartes liss?es et originales ; plus faible = moins de d?formation.
- Les ?carts-types LOIO d?crivent la dispersion entre les 24 ?valuations image-seed.
- Les 24 ?valuations ne sont pas ind?pendantes, car les images sont r?utilis?es dans les rotations LOIO.
- Les r?f?rences Sigma fixes 0, 0.1 et 0.4 sont des points de comparaison descriptifs, pas des configurations choisies par LOIO.
- La s?lection adaptative respecte le seuil de conservation observ? dans les ?valuations, mais ne garantit pas une meilleure stabilit?.
