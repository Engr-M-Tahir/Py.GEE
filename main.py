#PY.GEE
# Initial Gui For Project
# Supervisor = Dr Sanay Muhammad Umar Saeed
# Groupmembers = Hafsa Shoaib , Momena Khalil , Muhammad Tahir

# importing libraries required
from sklearn.cluster import KMeans
import pandas as pd
from tkinter import *
import tkinter as tk
from tkinter.ttk import *
from tkinter import ttk, filedialog
import numpy as np
from PIL import ImageTk,Image
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn import svm


filename = ' '
def browseFiles():
   global filename
   filename = filedialog.askopenfilename(initialdir = "/",
                                          title = "Select a File",
                                          filetypes = (("csv files",
                                                        ".csv"),
                                                       ("all files", ".")))



def featExtractTime():
    frame = pd.read_csv(filename, usecols={'Delta_TP9',
                           'Delta_AF7',
                           'Delta_AF8',
                           'Delta_TP10',
                           'Theta_TP9',
                           'Theta_AF7',
                           'Theta_AF8',
                           'Theta_TP10',
                           'Alpha_TP9',
                           'Alpha_AF7',
                           'Alpha_AF8',
                           'Alpha_TP10',
                           'Beta_TP9',
                           'Beta_AF7',
                           'Beta_AF8',
                           'Beta_TP10',
                           'Gamma_TP9',
                           'Gamma_AF7',
                           'Gamma_AF8',
                           'Gamma_TP10'}).dropna()

    data = frame.iloc[:, :24]
    max = data.max().tolist()
    min = data.min().tolist()
    mean1 = data.mean()
    mean = data.mean().tolist()
    med = data.median().tolist()
    std = data.std().tolist()
    sk = data.skew().tolist()
    kurt = data.kurtosis().tolist()
    mode = mean1.apply(lambda x: x % 3).tolist()
    com = {"Minimum": min, "Max": max, "Average": mean, "Median": med, "Mod": mode, "Standard Deviation": std,
           "Skewness": sk, "Kurtosis": kurt, }
    d1 = pd.DataFrame(com, index=['Delta_TP9',
                                  'Delta_AF7',
                                  'Delta_AF8',
                                  'Delta_TP10',
                                  'Theta_TP9',
                                  'Theta_AF7',
                                  'Theta_AF8',
                                  'Theta_TP10',
                                  'Alpha_TP9',
                                  'Alpha_AF7',
                                  'Alpha_AF8',
                                  'Alpha_TP10',
                                  'Beta_TP9',
                                  'Beta_AF7',
                                  'Beta_AF8',
                                  'Beta_Tp10',
                                  'Gamma_TP9',
                                  'Gamma_AF7',
                                  'Gamma_AF8',
                                  'Gamma_TP10'
                                  ])
    with pd.ExcelWriter(r'E:\Py.GEE\Results\TimeDomainFeatures.xlsx') as writer:
        d1.to_excel(writer)
    print('Time Domain Features Extracted')

def featExtractFreq():
    frame = pd.read_csv(filename, usecols={'RAW_TP9',
                           'RAW_AF7',
                           'RAW_AF8',
                           'RAW_TP10'}).dropna()

    data = frame.iloc[:, :24]
    max = data.max().tolist()
    min = data.min().tolist()
    mean1 = data.mean()
    mean = data.mean().tolist()
    med = data.median().tolist()
    std = data.std().tolist()
    sk = data.skew().tolist()
    kurt = data.kurtosis().tolist()
    mode = mean1.apply(lambda x: x % 3).tolist()
    com = {"Minimum": min, "Max": max, "Average": mean, "Median": med, "Mod": mode, "Standard Deviation": std,
           "Skewness": sk, "Kurtosis": kurt, }
    d1 = pd.DataFrame(com, index=['RAW_TP9',
                           'RAW_AF7',
                           'RAW_AF8',
                           'RAW_TP10'
                                  ])
    with pd.ExcelWriter(r'E:\Py.GEE\Results\FreQDomainFeatures.xlsx') as writer:
        d1.to_excel(writer)
    print('Frequency Domain Features Extracted')

def featExtractHjorth():
    data = pd.read_csv(filename,
                       usecols={'Delta_TP9',
                                'Delta_AF7',
                                'Delta_AF8',
                                'Delta_TP10',
                                'Theta_TP9',
                                'Theta_AF7',
                                'Theta_AF8',
                                'Theta_TP10',
                                'Alpha_TP9',
                                'Alpha_AF7',
                                'Alpha_AF8',
                                'Alpha_TP10',
                                'Beta_TP9',
                                'Beta_AF7',
                                'Beta_AF8',
                                'Beta_TP10',
                                'Gamma_TP9',
                                'Gamma_AF7',
                                'Gamma_AF8',
                                'Gamma_TP10',
                                'RAW_TP9',
                                'RAW_AF7',
                                'RAW_AF8',
                                'RAW_TP10'}).dropna()

    data_1 = data.iloc[:, :].values

    def first_deriv(data):
        fIrst_deriv = np.diff(data)
        return fIrst_deriv

    def mobility(data):
        mobility = (variance(first_deriv(data)) / variance(data)) ** 1 / 2
        return mobility

    def Complexity(data):
        cOmplexity = mobility(first_deriv(data)) / mobility(data)
        return cOmplexity

    def variance(data, ddof=0):
        n = len(data_1)
        mean = sum(data_1) / n
        vAriance = sum((x - mean) ** 2 for x in data_1) / (n - ddof)
        return vAriance

    activity = variance(data_1)

    com = {'Activity': activity, 'Complexity ': Complexity(data_1), 'Mobility': mobility(data_1)}
    d1 = pd.DataFrame(com, index=['Delta_TP9',
                                  'Delta_AF7',
                                  'Delta_AF8',
                                  'Delta_TP10',
                                  'Theta_TP9',
                                  'Theta_AF7',
                                  'Theta_AF8',
                                  'Theta_TP10',
                                  'Alpha_TP9',
                                  'Alpha_AF7',
                                  'Alpha_AF8',
                                  'Alpha_TP10',
                                  'Beta_TP9',
                                  'Beta_AF7',
                                  'Beta_AF8',
                                  'Beta_Tp10',
                                  'Gamma_TP9',
                                  'Gamma_AF7',
                                  'Gamma_AF8',
                                  'Gamma_TP10',
                                  'RAW_TP9',
                                  'RAW_AF7',
                                  'RAW_AF8',
                                  'RAW_TP10'])
    with pd.ExcelWriter(r'E:\Py.GEE\Results\Hjorthfeatures.xlsx') as writer:
        d1.to_excel(writer)
    print('Hjorth Features Extracted')


#====================================================================================
def iCA():
    df = pd.read_csv(filename).dropna()
    df.isna().sum()

    from sklearn.decomposition import FastICA
    Y = df['Depressed/Not']
    X = df.drop('Depressed/Not', axis=1)

    Y_mod = []
    for i in range(len(Y)):
        if Y[i] == 'D':
            Y_mod.append(1)
        else:
            Y_mod.append(0)

    components = len(X.columns)
    ica = FastICA(n_components=components)
    S_ = ica.fit_transform(X)
    result = pd.DataFrame(S_, columns=X.columns)
    with pd.ExcelWriter(r'E:\Py.GEE\Results\IndependentComponentAnalysis.xlsx') as writer:
        result.to_excel(writer)
    print('done-ica')

def featSelKBEST():
    df = pd.read_csv(filename)
    df.isna().sum()

    from sklearn.decomposition import FastICA
    Y = df['Depressed/Not']
    X = df.drop('Depressed/Not', axis=1)

    Y_mod = []
    for i in range(len(Y)):
        if Y[i] == 'D':
            Y_mod.append(1)
        else:
            Y_mod.append(0)

    components = len(X.columns)
    ica = FastICA(n_components=components)
    S_ = ica.fit_transform(X)
    result = pd.DataFrame(S_, columns=X.columns)
    # k-means clustering
    kmeans = KMeans(n_clusters=2)
    Y = kmeans.fit_predict(result)

    from sklearn.feature_selection import mutual_info_classif

    miC = mutual_info_classif(result, y=Y_mod, n_neighbors=5)  # you can change n_neighbors according to requirement

    args = np.argwhere(miC > 0).reshape(-1).tolist()
    cols = result.columns
    args = cols[args].tolist()
    X_mod = result[args]

    # X_mod['Depressed/Not']=Y_mod
    with pd.ExcelWriter(r'E:\Py.GEE\Results\KBestFeatureSelection.xlsx') as writer:
        X_mod.to_excel(writer)
    print('K-Best Feature Selection Applied')

def naiveBayes():
    df = pd.read_csv('eegDataSetforClassification.csv')

    df.dropna(inplace=True)

    Y = df['Depressed/Not']
    X = df.drop('Depressed/Not', axis=1)
    scaler = MinMaxScaler()
    X_Scaled = scaler.fit_transform(X)
    y = Y.tolist()
    labels = []
    for i in y:
        if i == 'yes':
            labels.append(1)
        else:
            labels.append(0)

    #### NAIVE BAYES

    X = X_Scaled
    Y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0)


    gnb = GaussianNB()
    model = gnb.fit(X_train, y_train)

    y_predicted = gnb.predict(X_test)
    testingaccuracyscore = accuracy_score(y_predicted, y_test)
    print("Testing Accuracy Score for Naaive Bayes : ", testingaccuracyscore)

def kNN():
    df = pd.read_csv('eegDataSetforClassification.csv')

    df.dropna(inplace=True)

    Y = df['Depressed/Not']
    X = df.drop('Depressed/Not', axis=1)
    scaler = MinMaxScaler()
    X_Scaled = scaler.fit_transform(X)
    y = Y.tolist()
    labels = []
    for i in y:
        if i == 'yes':
            labels.append(1)
        else:
            labels.append(0)

    #### NAIVE BAYES

    X = X_Scaled
    Y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0)


    gnb = GaussianNB()
    model = gnb.fit(X_train, y_train)

    y_predicted = gnb.predict(X_test)
    testingaccuracyscore = accuracy_score(y_predicted, y_test)
    neigh = KNeighborsClassifier(n_neighbors=2)
    model = neigh.fit(X_train, y_train)
    yPredicted = model.predict(X_test)
    testing_accuracyscore = accuracy_score(yPredicted, y_test)
    print("Testing Accuracy Score for KNN : ", testing_accuracyscore)
    cm = confusion_matrix(yPredicted, y_test)
    sns.heatmap(cm, annot=True)

def sVM():
    df = pd.read_csv('eegDataSetforClassification.csv')

    df.dropna(inplace=True)

    Y = df['Depressed/Not']
    X = df.drop('Depressed/Not', axis=1)
    scaler = MinMaxScaler()
    X_Scaled = scaler.fit_transform(X)
    y = Y.tolist()
    labels = []
    for i in y:
        if i == 'yes':
            labels.append(1)
        else:
            labels.append(0)

    #### NAIVE BAYES

    X = X_Scaled
    Y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    gnb = GaussianNB()
    model = gnb.fit(X_train, y_train)

    y_predicted = gnb.predict(X_test)
    accuracy_score(y_predicted, y_test)


    cm = confusion_matrix(y_predicted, y_test)
    sns.heatmap(cm, annot=True)
    ###################### KNN

    neigh = KNeighborsClassifier(n_neighbors=2)
    model = neigh.fit(X_train, y_train)
    yPredicted = model.predict(X_test)
    accuracy_score(yPredicted, y_test)

    cm = confusion_matrix(yPredicted, y_test)
    sns.heatmap(cm, annot=True)

    #### SVM

    model = svm.SVC(kernel='linear')
    model.fit(X_train, y_train)
    predictedTest = model.predict(X_test)
    testingAccuracyScore = accuracy_score(predictedTest, y_test)
    print("Testing Accuracy Score for SVM : ", testingAccuracyScore)
    cm = confusion_matrix(predictedTest, y_test)
    sns.heatmap(cm, annot=True)


root = tk.Tk()
apps = []

root.title('PY.GEE')
root.geometry("826x676")
root.resizable(width=False, height=False)
canvas = tk.Canvas(root, height=826, width=676)
file = ImageTk.PhotoImage(Image.open("usedimage.PNG"))
canvas.create_image(0,0, anchor=NW, image= file)

canvas.pack(fill="both", expand = True)




#SELECT_FILE
#_______________________________________________
selectFile = tk.Button(canvas, text="Select File", padx=20, pady=10,fg="WHITE", bg="gray", command=browseFiles)
#selectFile.pack()

#Time Domain
tdf = tk.Button(canvas, text="Time Domain Features", padx=20, pady=10,fg="WHITE", bg="gray", command=featExtractTime)
#Freq Domain
fdf = tk.Button(canvas, text="Frequency Domain Features", padx=20, pady=10,fg="WHITE", bg="gray", command=featExtractFreq)
#Hjorth Parameters

hp = tk.Button(canvas, text="Hjorth Parameters", padx=20, pady=10,fg="WHITE", bg="gray", command=featExtractHjorth)
#_______________________________________________



#Feature_SELECTION
#_______________________________________________
ftSel = tk.Button(canvas, text="Feature Selection", padx=20, pady=10, fg="WHITE", bg="gray", command=featSelKBEST)

#_______________________________________________

#ICA
#_______________________________________________
iCA = tk.Button(canvas, text="ICA", padx=20, pady=10,fg="WHITE", bg="gray", command=iCA)

#_______________________________________________

#CLASSIFIERS
#Naive Bayes
#_______________________________________________
nb = tk.Button(canvas, text="Naive Bayes", padx=20, pady=10,fg="WHITE", bg="gray", command=naiveBayes)

#_______________________________________________

#SVM
#_______________________________________________
sVM1 = tk.Button(canvas, text="  SVM    ", padx=20, pady=10,fg="WHITE", bg="gray", command=sVM)
#svm.pack()
#_______________________________________________

#KNN
#_______________________________________________
knn = tk.Button(canvas, text="  KNN   ", padx=20, pady=10,fg="WHITE", bg="gray", command=kNN)
#knn.pack()
#_______________________________________________
kBest = tk.Button(canvas, text="K Best", padx=20, pady=10,fg="WHITE", bg="gray", command=featSelKBEST)
#GRID
label1 = Label(canvas, font= "Calibri",text="Feature Extraction")
label1.grid(row=1, column=1, sticky=W, pady=10)
tdf.grid(row=2,column=1,sticky=W)
fdf.grid(row=2,column=2,sticky=W)
hp.grid(row=2, column=3,sticky=W)
label2 = Label(canvas, font="Calibri", text="Independent Component Analysis")
label2.grid(row=4, column=1, sticky=W, pady=10, columnspan= 2)
iCA.grid(row=5, column= 1, sticky=W)
label3 = Label(canvas, font="Calibri", text="Feature Selection")
label3.grid(row=6, column=1, sticky=W, pady=10, columnspan= 2)
kBest.grid(row=7,column=1,sticky=W)
label4 = Label(canvas, font="Calibri", text="Classification")
label4.grid(row=8, column=1, sticky=W, pady=5, columnspan= 2)
nb.grid(row=9,column=1,sticky=W)
knn.grid(row=9, column=2, sticky=W)
sVM1.grid(row=9,column=3,sticky=W)
label5 = Label(canvas)
label5.grid(row=10, column=1, sticky=W, pady=5, columnspan= 2)
selectFile.grid(row=11, column= 1, sticky=W)

#selectFile.grid(row= 2, column=2, pady=10)
#ftExtract.grid(row= 2, column=3, pady=10)
root.mainloop()