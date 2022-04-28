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
    
<img width="1137" alt="image" src="https://user-images.githubusercontent.com/89613437/165841710-cd01fc62-3088-409e-8a03-d3846148d8af.png">

    # remove meaningless rows. There are 2126 fetuses in this dataset.
    df = raw_data.drop([2126,2127,2128])
    # NSP is the target in this dataset. Check the balance of target.
    df.NSP.value_counts()
    # Even if it is unbalanced, I will give priority to using real data to build machine learning models. If the results are not good, I will try oversampling or undersampling
    
<img width="218" alt="image" src="https://user-images.githubusercontent.com/89613437/165842311-6907d76c-ff99-4522-adfa-af7660e605e3.png">

    # Confirm the distribution of the target
    fetal_states = df.NSP.value_counts().to_frame()
    plt.figure(dpi=200)
    pie_fetal_states = plt.pie(fetal_states.NSP, labels=["Normal", "Suspect", "Pathologic"], 
                           colors = ["steelblue", "lightskyblue", "lightblue"], autopct="%1.0f%%")
    plt.title("Fetal States");
    
<img width="560" alt="image" src="https://user-images.githubusercontent.com/89613437/165842442-c79e7b7d-7dde-46e2-857c-6bb20bfd3b44.png">

    # Based on the background information, the exam data is not important. 
    # The CLASS and 10 Morphologic Patterns (A, B, C...) have same information, and they are come from doctors.
    # This project is based on CTG data, so I delet these features.

    # remove FileName, Date, and SegFile
    df = df.drop(['FileName', 'Date', 'SegFile'], axis = 1)
    # remove b, e
    df = df.drop(['b', 'e'], axis = 1)
    # remove A, B, C, D, E, AD, DE, LD, FS, SUSP, and CLASS
    df = df.drop(['A', 'B', 'C', 'D', 'E', 'AD', 'DE', 'LD', 'FS', 'SUSP', 'CLASS'], axis = 1)
    df

<img width="1078" alt="image" src="https://user-images.githubusercontent.com/89613437/165842564-575ef28e-af1b-4e5a-abc2-9c012cd0dcc1.png">

    # Check the characteristics of features. According to the backgroud information, the zero is meaningful.
    df.describe().T










    
    
    
    
    
    

