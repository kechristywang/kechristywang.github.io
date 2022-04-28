# Predicting fetal states based on Cardiotocography
Cardiotocography is an essential indicator of fetal states. Earlier detection of a pathological fetus allows physicians to take action to prevent mortality. The results of machine learning models can inform obstetricians in determining the fetal states. This project uses the random forest to train a model to predict three fetal states (Normal, Suspect, Pathological).

### CTG output has 4 parts:
* A: Fetal heartbeat;
* B: Indicator showing movements felt by mother (triggered by pressing a button);
* C: Fetal movement;
* D: Uterine contractions

## Dataset sources:
- [the UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Cardiotocography)

### Features:
#### Exam data:
    FileName - of CTG examination
    Date - of the examination
    b - start instant
    e - end instant
#### Measurements:
    LBE - baseline value (medical expert)
    LB - baseline value (SisPorto)
    AC - accelerations (SisPorto)
    FM - fetal movement (SisPorto)
    UC - uterine contractions (SisPorto)
    ASTV - percentage of time with abnormal short term variability  (SisPorto)
    mSTV - mean value of short term variability  (SisPorto)
    ALTV - percentage of time with abnormal long term variability  (SisPorto)
    mLTV - mean value of long term variability  (SisPorto)
    DL - light decelerations
    DS - severe decelerations
    DP - prolongued decelerations
    DR - repetitive decelerations
    Width - histogram width
    Min - low freq. of the histogram
    Max - high freq. of the histogram
    Nmax - number of histogram peaks
    Nzeros - number of histogram zeros
    Mode - histogram mode
    Mean - histogram mean
    Median - histogram median
    Variance - histogram variance
    Tendency - histogram tendency: -1=left assymetric; 0=symmetric; 1=right assymetric
#### Classification:
    A - calm sleep
    B - REM sleep
    C - calm vigilance
    D - active vigilance
    SH - shift pattern (A or Susp with shifts)
    AD - accelerative/decelerative pattern (stress situation)
    DE - decelerative pattern (vagal stimulation)
    LD - largely decelerative pattern
    FS - flat-sinusoidal pattern (pathological state)
    SUSP - suspect pattern
    CLASS - Class code (1 to 10) for classes A to SUSP
    NSP - Normal=1; Suspect=2; Pathologic=3
## Data Cleaning
    # module loading 

    import numpy as np
    import pandas as pd
    import warnings
    import matplotlib.pyplot as plt
    import seaborn as sns
    import sklearn.metrics as metrics
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.cluster import KMeans

    import math
    from collections import Counter
    import scipy.stats as ss
    import sklearn.preprocessing as sp
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score
    
    # Loading raw data
    raw_data = pd.read_table("CTG.csv", sep = ",")
    raw_data
    
    


