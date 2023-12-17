# Challenge Documentation

Decisions, assumptions, and comments will be presented in this file. 
They are divided by the stages of the solution (parts).

# Part I

### Transcription of the notebook into the api implementation

As the data scientist is responsible for the analysis, its background rationale and completeness are trustworthy.

I found a couple if minor mistakes in the syntaxis related to calling functiions with keywords arguments only in some of the arguments and not all of them.

I found also that the `training_data` defined in 4.a was not used, ergo, it was not used when training the models.


### Model decision

In addition to the DS's analysis, I was interested in the linearity correlation of the `delay` with all the other variables. I used the Cramér's V metric for correlation, it returns a value between 0 and 1which 0 = no association and 1 = complete association. 

The implementation of the script:

```python
import numpy as np

def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

# Linearity calculation: 0 = no association, 1 = complete association.
print(f"Cramér's correlation for 'delay'")
metrics_to_compare = ["Fecha-I","Vlo-I","Ori-I","Des-I","Emp-I","Fecha-O","Vlo-O","Ori-O","Des-O","Emp-O","DIA","MES","AÑO","DIANOM","TIPOVUELO","OPERA","SIGLAORI","SIGLADES", 'high_season', 'min_diff', 'period_day']
for metric in metrics_to_compare:
    confusion_matrix = pd.crosstab(data['delay'], data[metric])
    cramer_v = cramers_v(confusion_matrix)
    print(f"'delay' vs {metric}: {cramer_v}")
```

Results:

#### Cramér's correlation  (Cramér's V) for `delay`

1. `delay` vs `Fecha-I`: 0.13410230750276278
1. `delay` vs `Vlo-I`: 0.276025179565339
1. `delay` vs `Des-I`: 0.16418834011433514
1. `delay` vs `Emp-I`: 0.1697730046212206
1. `delay` vs `Fecha-O`: 0.06104773572098205
1. `delay` vs `Vlo-O`: 0.28343883081755594
1. `delay` vs `Des-O`: 0.16417768723394746
1. `delay` vs `Emp-O`: 0.18121062751092176
1. `delay` vs `DIA`: 0.052624562353062655
1. `delay` vs `MES`: 0.13117919736637917
1. `delay` vs `AÑO`: 0.0
1. `delay` vs `DIANOM`: 0.05604163815441127
1. `delay` vs `TIPOVUELO`: 0.09618122383571627
1. `delay` vs `OPERA`: 0.16185084083276668
1. `delay` vs `SIGLADES`: 0.1639824365749197
1. `delay` vs `high_season`: 0.02063651082400301
1. `delay` vs `min_diff`: 0.998723600392344
1. `delay` vs `period_day`: 0.04590699231013053

We can see that there is no linear correlation between `delay` and the othe variables.
The only exception is `min_diff`.

Considering:
1. No significant differences in performance between both models
1. Non-linearity interactions between the target variable and the rest *and/or* no dependency at all between 
1. The size of the dataset may be large (XGBoost handles it better)

I propose using XGBoost for this problem. 

## Part II

...

## Part III

...

## Part IV

...