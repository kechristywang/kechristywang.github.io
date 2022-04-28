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
    
<img width="345" alt="image" src="https://user-images.githubusercontent.com/89613437/165843352-533406b7-3bba-4857-9f8d-50e9cd480093.png">

    df.DS.value_counts()

    # For DS, only 7 samples is different with others. The value in DR are all zero.
    # Delete them because they provide too little information.

<img width="209" alt="image" src="https://user-images.githubusercontent.com/89613437/165843426-7bcb35e5-233f-46ed-a280-e9021ad8ea57.png">

    # remove DS, DR
    df = df.drop(['DS', 'DR'], axis = 1)
    # checking which feature left
    df.columns.unique()
    
<img width="702" alt="image" src="https://user-images.githubusercontent.com/89613437/165843530-22515edb-473f-4d47-98b8-e23c4e4e41ee.png">

    # using histogram to check the distribution

    his = ['LBE', 'LB', 'AC', 'FM', 'UC', 'ASTV', 'MSTV', 'ALTV', 'MLTV', 'DL',
       'DP', 'Width', 'Min', 'Max', 'Nmax', 'Nzeros', 'Mode', 'Mean', 'Median',
       'Variance', 'Tendency', 'NSP']

    fig = plt.figure(figsize=(15,20), dpi=300)
    for i in range(0,22):
        plt.subplots_adjust(wspace=0.5,hspace=0.6)
        ax = fig.add_subplot(6,4,i + 1)
        sns.histplot(x = df[his[i]])
        plt.title(his[i])

<img width="1236" alt="image" src="https://user-images.githubusercontent.com/89613437/165843631-80680212-ac9f-4b58-98bf-cacb4a6a2eff.png">
<img width="1240" alt="image" src="https://user-images.githubusercontent.com/89613437/165843664-4bf6fd37-1916-48a2-b90a-4496689ce1aa.png">
<img width="1233" alt="image" src="https://user-images.githubusercontent.com/89613437/165843714-07bc01e0-414c-4291-ac67-91286ac26e0c.png">

    # FM is really strange, and it is not at important for diagnosis
    # keep other features first

    # remove FM 
    df = df.drop(['FM'], axis = 1)

    # Temporarily remove Tendency for heatmap
    df_heatmap = df.drop(['Tendency'], axis = 1)

    # Using the heatmap to find relevant numerical features.

    warnings.filterwarnings("ignore") 
    df_heatmap.corr()
    plt.figure(dpi=400)
    plt.figure(figsize=(30,15))
    sns.heatmap(df.corr(), annot = True, fmt = '.2f', annot_kws={'size': 15}, 
            mask = np.triu(np.ones_like(df.corr(), dtype = np.bool)))
    plt.xticks(fontsize=15, rotation = 45)
    plt.yticks(fontsize=15, rotation = 45)
    plt.title("Heatmap of All Numerical Features", fontsize = 30);

<img width="889" alt="image" src="https://user-images.githubusercontent.com/89613437/165843873-17560e8f-3f0f-4844-832c-b9b720563879.png">

    # screening 0.8 - 1.0
    # LBE & LB
    # Min & Width
    # Mean & Mode & Median

    # keep LB, Width, and Mode

    # remove LBE, Min, Mean, Median
    df = df.drop(['LBE', 'Min', 'Mean', 'Median'], axis = 1)
    df
    
<img width="903" alt="image" src="https://user-images.githubusercontent.com/89613437/165844003-fba69539-e3ee-4c4e-ab59-307083f267c2.png">

    # Using the Chi square test to figure out the correlation of Tendency and target.

    #  Tendency and target
    DP_and_Tendency = df.loc[:, ['Tendency', 'NSP']]
    DP_and_Tendency
    
<img width="179" alt="image" src="https://user-images.githubusercontent.com/89613437/165844141-7c53647e-c386-4113-9434-639158c76cf4.png">

    # Chi square test code

    def conditional_entropy(x,y):
        # entropy of x given y
        y_counter = Counter(y)
        xy_counter = Counter(list(zip(x,y)))
        total_occurrences = sum(y_counter.values())
        entropy = 0
        for xy in xy_counter.keys():
            p_xy = xy_counter[xy] / total_occurrences
            p_y = y_counter[xy[1]] / total_occurrences
            entropy += p_xy * math.log(p_y/p_xy)
        return entropy

    def theil_u(x,y):
        s_xy = conditional_entropy(x,y)
        x_counter = Counter(x)
        total_occurrences = sum(x_counter.values())
        p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))
        s_x = ss.entropy(p_x)
        if s_x == 0:
            return 1
        else:
            return (s_x - s_xy) / s_x
    # heatmap of Tendency

    theilu = pd.DataFrame(index=['NSP'],columns=DP_and_Tendency.columns)
    columns = DP_and_Tendency.columns
    for j in range(0,len(columns)):
        u = theil_u(DP_and_Tendency['NSP'].tolist(),DP_and_Tendency[columns[j]].tolist())
        theilu.loc[:,columns[j]] = u
    theilu.fillna(value=np.nan,inplace=True)
    plt.figure(figsize=(20,1), dpi=400)
    sns.heatmap(theilu,annot=True,fmt='.2f', annot_kws={'size': 20},)
    plt.title("Correlation of the Categorical  Features", fontsize = 20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20);
    plt.show()

<img width="1173" alt="image" src="https://user-images.githubusercontent.com/89613437/165844290-bff10531-ae26-474b-8bb0-e36e4f043772.png">

    # they are independent, remove Tendency
    df = df.drop(['Tendency'], axis = 1)
    df

<img width="809" alt="image" src="https://user-images.githubusercontent.com/89613437/165844366-a27befd9-a144-4826-96bd-d6353f27eb83.png">

    # Final features (15) and target (1)

    df.columns.unique()
    
<img width="693" alt="image" src="https://user-images.githubusercontent.com/89613437/165844459-b49a863f-0fbe-4888-9eb3-afdf42c36240.png">

## Dataset splitting

    ## feature dataframe
    X = df.drop(['NSP'], axis = 1)
    ## target dataframe
    Y = df.loc[:, ['NSP']]
    y = Y.values.ravel()

    ## Retain 40% samples as the test set
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.4, random_state = 515, stratify=df['NSP'])
    y_train = y_train.values.ravel()
    
    #A quick model selection process
    #pipelines of models
    pipeline_lr=Pipeline([('lr_classifier',LogisticRegression(random_state= 515))])

    pipeline_dt=Pipeline([ ('dt_classifier',DecisionTreeClassifier(random_state= 515))])

    pipeline_rf=Pipeline([('rf_classifier',RandomForestClassifier(random_state= 515))])

    pipeline_svc=Pipeline([('sv_classifier',SVC(random_state= 515))])


    # List of all the pipelines
    pipelines = [pipeline_lr, pipeline_dt, pipeline_rf, pipeline_svc]

    # Dictionary of pipelines and classifier types for ease of reference
    pipe_dict = {0: 'Logistic Regression', 1: 'Decision Tree', 2: 'RandomForest', 3: "SVC"}


    # Fit the pipelines
    for pipe in pipelines:
        pipe.fit(x_train, y_train)

    #cross validation on accuracy 
    cv_results_accuracy = []
    for i, model in enumerate(pipelines):
        cv_score = cross_val_score(model, x_train, y_train, cv=10 )
        cv_results_accuracy.append(cv_score)
        print("%s: %f " % (pipe_dict[i], cv_score.mean()))

    # Randomforest gets the best result

<img width="298" alt="image" src="https://user-images.githubusercontent.com/89613437/165844591-7405cd3c-537b-47d8-8b2f-15d12ac152ae.png">

    # find the best criterion of decision trees

    RF = RandomForestClassifier(random_state = 515)
    score = cross_val_score(RF,X,y,cv=10).mean()
    print('gini score: %.4f'%score)
    RF = RandomForestClassifier(criterion = 'entropy',random_state = 515)
    score = cross_val_score(RF,X,y,cv=10).mean()
    print('entropy score: %.4f'%score)

<img width="198" alt="image" src="https://user-images.githubusercontent.com/89613437/165845569-b62e1ec9-f838-4633-a8d1-6aa6982c2bd6.png">

    # find the best number of trees

    superpa = []
    for i in range(100):
        rfc = RandomForestClassifier(criterion = 'entropy', n_estimators=i+1, random_state = 515, max_features = 'sqrt', bootstrap = True, oob_score = False)
        rfc_s_mod = rfc.fit(x_train, y_train)
        rfc_s = rfc.score(x_test, y_test)
        superpa.append(rfc_s)
    print(max(superpa),superpa.index(max(superpa))+1)
    plt.figure(figsize=[8,5], dpi = 100)
    plt.plot(range(1,101),superpa)
    plt.title("Number of Trees Curve", fontsize = 20)
    plt.show()

<img width="783" alt="image" src="https://user-images.githubusercontent.com/89613437/165845682-d0c3fa3e-c63c-445c-8d2d-8fd12b9db00b.png">

    # find the best number of Tree Depth

    MD_L = []
    for i in range(30):
        MD = RandomForestClassifier(criterion = 'entropy', n_estimators=94, max_depth =i+1, 
                                    random_state = 515, max_features = 'sqrt', bootstrap = True, oob_score = False)
        MD_s_mod = MD.fit(x_train, y_train)
        MD_s = MD.score(x_test, y_test)
        MD_L.append(MD_s)
    print(max(MD_L),MD_L.index(max(MD_L))+1)
    plt.figure(figsize=[8,5], dpi = 100)
    plt.plot(range(1,31),MD_L)
    plt.title("Tree Depth Curve", fontsize = 20)
    plt.show()

<img width="786" alt="image" src="https://user-images.githubusercontent.com/89613437/165845750-90cbb0a3-b859-421d-8353-a2def4e43da2.png">

    # find the best The Minimum Number of Samples Required to Split an Internal Node

    MSS_L = []
    for i in range(2, 10):
        MSS = RandomForestClassifier(criterion = 'entropy', n_estimators = 94, max_depth = 20, min_samples_split = i,
                                     random_state = 515, max_features = 'sqrt', bootstrap = True, oob_score = False)
        MSS_s_mod = MSS.fit(x_train, y_train)
        MSS_s = MSS.score(x_test, y_test)
        MSS_L.append(MSS_s)
    print(max(MSS_L),MSS_L.index(max(MSS_L))+2)
    plt.figure(figsize=[8,5], dpi = 100)
    plt.plot(range(2,10),MSS_L)
    plt.title("The Minimum Number of Samples Required to Split an Internal Node Curve", fontsize = 15)
    plt.show()
    
<img width="895" alt="image" src="https://user-images.githubusercontent.com/89613437/165845862-16a61537-b1a0-4255-bbad-75977bc1111a.png">

    # find the best The Minimum Number of Samples Required to be at a Leaf Node

    MSL_L = []
    for i in range(15):
        MSL = RandomForestClassifier(criterion = 'entropy', n_estimators = 94, max_depth = 20, min_samples_split = 2, min_samples_leaf = i+1,
                                           random_state = 515, max_features = 'sqrt', bootstrap = True, oob_score = False)
        MSL_s_mod = MSL.fit(x_train, y_train)
        MSL_s = MSL.score(x_test, y_test)
        MSL_L.append(MSL_s)
    print(max(MSL_L),MSL_L.index(max(MSL_L))+1)
    plt.figure(figsize=[8,5], dpi = 100)
    plt.plot(range(1,16),MSL_L)
    plt.title("The Minimum Number of Samples Required to be at a Leaf Node Curve", fontsize = 15)
    plt.show()
    
<img width="846" alt="image" src="https://user-images.githubusercontent.com/89613437/165845943-dd282fdd-f49d-474b-96fb-cdeda6701390.png">

   # find the best number of features

    MF_L = []
    for i in range(15):
        MF = RandomForestClassifier(criterion = 'entropy', n_estimators = 94, max_depth = 20, min_samples_split = 2, min_samples_leaf = 1,
                                           random_state = 515, max_features = i+1, bootstrap = True, oob_score = False)
        MF_s_mod = MF.fit(x_train, y_train)
        MF_s = MF.score(x_test, y_test)
        MF_L.append(MF_s)
    print(max(MF_L),MF_L.index(max(MF_L))+1)
    plt.figure(figsize=[8,5], dpi = 100)
    plt.plot(range(1,16),MF_L)
    plt.title("Number of Features Curve", fontsize = 20)
    plt.show() 
    
<img width="800" alt="image" src="https://user-images.githubusercontent.com/89613437/165846055-34fb0f06-42a2-47b5-8930-7aac8c6089b7.png">

    # Final model

    random_forest = RandomForestClassifier(criterion = 'entropy', n_estimators = 94, max_depth = 20, min_samples_split = 2, min_samples_leaf = 1,
                                           random_state = 515, max_features = 3, bootstrap = True, oob_score = False)
    random_forest_mod = random_forest.fit(x_train, y_train)
    
    # accuracy of model

    train_score = random_forest.score(x_train, y_train)
    test_score = random_forest.score(x_test, y_test)

    print('train_score: ' , train_score)
    print('test_score: ' , test_score)
    
<img width="297" alt="image" src="https://user-images.githubusercontent.com/89613437/165846212-2fd8d207-63fa-4296-8705-736107e10fec.png">

    # precision, recall, f1-score, and support

    predictions=random_forest.predict(x_test)
    print(classification_report(y_test, predictions))

    # The discrimination ability of the model for the suspected group is relatively poor.

<img width="488" alt="image" src="https://user-images.githubusercontent.com/89613437/165846314-580dac7f-5a71-4209-866c-e297e390c7a2.png">

    ## confusion matrix
    test_predict = random_forest.predict(x_test)
    confusion_matrix_result = metrics.confusion_matrix(test_predict,y_test)

    plt.figure(figsize=(9, 9), dpi=100)
    sns.heatmap(confusion_matrix_result, annot=True, 
                fmt=".0f", linewidths=.5, square = True, 
                cmap= 'Blues', annot_kws={"size": 20},
                xticklabels = ['Normal', 'Suspect', 'Pathologic'],
                yticklabels = ['Normal', 'Suspect', 'Pathologic'],)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Predicted labels', fontsize = 20);
    plt.ylabel('True labels', fontsize = 20);
    plt.title('Accuracy Score: {}'.format(metrics.accuracy_score(y_test,test_predict)), size = 17);
    plt.show()

<img width="542" alt="image" src="https://user-images.githubusercontent.com/89613437/165846457-5626c4db-cb5a-42fb-83cb-ed90e4b33726.png">

    # Ranking the importance of features

    Features = ['LB', 'AC', 'UC', 'ASTV', 'MSTV', 'ALTV', 'MLTV', 'DL', 'DP', 'Width',
           'Max', 'Nmax', 'Nzeros', 'Mode', 'Variance']


    importances = random_forest_mod.feature_importances_

    indices = np.argsort(importances)[::-1]
    #print(indices)
    names = [Features[i] for i in indices]
    plt.figure(dpi=200)
    plt.title("Feature Importance")
    plt.bar(range(X.shape[1]), importances[indices])
    plt.xticks(range(X.shape[1]), names, rotation= 90)
    plt.show()

<img width="718" alt="image" src="https://user-images.githubusercontent.com/89613437/165846576-252c284e-1bdd-4eaa-bec7-cb49f2541e23.png">

    # Relationship between four important characteristics and groups

    column = ['ASTV', 'MSTV', 'ALTV', 'Mode']

    fig = plt.figure(figsize=(15,9), dpi=300)
    for i in range(0,4):
        plt.subplots_adjust(wspace=0.5,hspace=0.6)
        ax = fig.add_subplot(2,2,i + 1)
        sns.boxplot(y='NSP', x=column[i], saturation=0.5,palette='pastel', data=df, orient = 'h' )
        ax.set_yticklabels(labels = ['Normal', 'Suspect', 'Pathologic'], rotation = 60,fontsize = 10)
        plt.title(column[i])

<img width="705" alt="image" src="https://user-images.githubusercontent.com/89613437/165846698-84b7f748-c93b-4929-94cd-4bbdfaaf521f.png">

## Summary

The accuracy of fetal state prediction model using random forest can reach 0.958. More important features can give doctors some clues. 

There are still aspects to be improved in this model. If more data about suspict group and pathological group can be added, this model can be improved.



