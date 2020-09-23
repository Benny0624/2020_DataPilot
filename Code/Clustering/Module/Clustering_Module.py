# Import the libraries
import os
import numpy as np
import pandas as pd
import xlsxwriter
import scipy.stats as ss
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.preprocessing import PowerTransformer
from sklearn.cluster import KMeans
from sklearn import metrics
from collections import Counter
from scipy.spatial.distance import cdist
from statsmodels.stats.multicomp import pairwise_tukeyhsd
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False

## Outlier detection , PowerTransform features
def Outlier_sheet(df):
    feature_outliers = []
    writer = pd.ExcelWriter('%s_Outliers_sheets.xlsx'%(NAME_SPLT), engine='xlsxwriter')
    
    # PowerTransform
    df_scaled_features = PT.fit_transform(df.values)
    df_scaled_features_df = pd.DataFrame(df_scaled_features, index=df.index,\
                                             columns=df.columns)
    # Outlier detection
    for feature in df.keys():
        Q1 = np.percentile(df_scaled_features_df[feature], 25)
        Q3 = np.percentile(df_scaled_features_df[feature], 75)
        step = 1.5*(Q3 - Q1)
        Outlier_Index = ~((df_scaled_features_df[feature] >= Q1 - step)&(df_scaled_features_df[feature] <= Q3 + step))
        df.loc[df.loc[Outlier_Index].index,:].to_excel(writer, sheet_name = feature)
        feature_outliers.append(df.loc[df.loc[Outlier_Index].index,:])
    writer.save()

    # Flatten list of outliers
    outliers_flattened = []

    for i, j in enumerate(feature_outliers):
        outliers_flattened.append(feature_outliers[i].index)
    flat_list = [item for sublist in outliers_flattened for item in sublist]

    # Count the number of features for which a given observation is considered an outlier
    outlier_count = Counter(flat_list)
    outliers = [observation for observation in outlier_count.elements() if outlier_count[observation] >= 6]
    
    # Save the outliers sheet
    df.loc[df.loc[outliers].index,:].to_csv('%s_Outliers.csv'%(NAME_SPLT), encoding='utf_8_sig')
    return df_scaled_features_df
    
## Clustering
def Clustering(df, df_scaled, elbow = False):
    if elbow == True:
        distortions = []
        K = range(2,100)
        for k in K:
            kmeanModel = KMeans(n_clusters=k).fit(df_scaled)
            distortions.append(sum(np.min(cdist(df_scaled, kmeanModel.cluster_centers_, 'euclidean'), axis=1))/\
                               df_scaled.shape[0])
        # Plot the elbow
        plt.clf()
        plt.plot(K, distortions, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Distortion')
        plt.title('The Elbow Method showing the optimal k')
        plt.savefig(NAME_SPLT + '_Elbow_Method' + '.png')
        best_size = input('Please input the elbow point')
    else:
        num_clusters = np.arange(2,10)
        kmeans_results = {}
        for size in num_clusters:
            kmeans = KMeans(n_clusters = size).fit(df_scaled)
            preds = kmeans.predict(df_scaled)
            kmeans_results[size] = metrics.silhouette_score(df_scaled, preds)

        best_size = max(kmeans_results, key = kmeans_results.get)

    optimized_kmeans = KMeans(n_clusters = best_size, random_state = 0).fit(df_scaled)
    kmeans_preds = optimized_kmeans.predict(df_scaled)
    kmeans_centers = optimized_kmeans.cluster_centers_
    
    # Inverse transform the box-cox centers
    true_centers = PT.inverse_transform(kmeans_centers)
    #true_centers[np.isnan(true_centers)] = 1 

    # Save the true centers
    writer = pd.ExcelWriter('%s_Data_Recovery_sheets.xlsx'%(NAME_SPLT), engine='xlsxwriter')
    segments = ['Segment {}'.format(i) for i in range(0,len(kmeans_centers))]
    true_centers = pd.DataFrame(np.round(true_centers), columns = df.iloc[:,:FEAT_NUM].keys())
    true_centers.index = segments
    true_centers.to_excel(writer, sheet_name = 'true_centers')
    np.round(df.iloc[:,:FEAT_NUM].mean()).to_excel(writer, sheet_name = 'Population_mean')
    (true_centers - np.round(df.iloc[:,:FEAT_NUM].mean())).to_excel(writer, sheet_name = 'True_centers_Minus_mean')
    
    # Save the label mean and customer list
    df['Label'] = optimized_kmeans.labels_
    round(df.groupby('Label').mean()).to_excel(writer, sheet_name = 'Label_mean')
    df.to_excel(writer, sheet_name = 'Customer_list')
    writer.save()
    return df.drop('Region',axis = 1)
    
# Statistical test
def Stats_test(DF, ANOVA = True):
    if ANOVA == True:
        # ANOVA test (k means)
        Variable = []
        F_val = []
        P_val = []
        for Var in DF.keys():
            samples = [val[1] for val in DF.groupby('Label')[Var]]
            f_val, p_val = ss.f_oneway(*samples)
            Variable.append(Var)
            F_val.append(round(f_val,2))
            P_val.append(round(p_val,2))
        pd.DataFrame({'Variable': (Variable),'F value':(F_val), 'p value':(P_val)}).to_csv('%s_ANOVA_Table.csv'%(NAME_SPLT))
    else:
        # Turkey HSD
        writer = pd.ExcelWriter('%s_Turkey_HSD_sheets.xlsx'%(NAME_SPLT), engine='xlsxwriter')
        for Variable in DF.keys():
            Results = pairwise_tukeyhsd(DF[Variable], DF['Label'])
            pd.DataFrame(data=Results._results_table.data[1:], columns=Results._results_table.data[0])\
            .to_excel(writer, sheet_name = Variable)
        writer.save()
    return None
    
# Setting Parameters
# PATH = 'C:/Users/User/Desktop/2020_DataPilot/Data/PChome_Data'
PATH = input("Please enter data path:")
NAME = input("Please enter data name:")
NAME_SPLT = NAME.split('.')[0]
DF_ORIGIN = pd.read_csv(os.path.join(PATH, NAME), encoding = 'utf-8').set_index('member_id')
DF = pd.read_csv(os.path.join(PATH, NAME), encoding = 'utf-8').set_index('member_id').drop('Region', axis = 1)
FEAT_NUM = DF.shape[1]
PT = PowerTransformer(method='yeo-johnson')

# Set path
os.chdir(PATH)

# Outlier detection
DF_SCALED = Outlier_sheet(DF)

# Clustering & Data recovery
DF_W_LABEL = Clustering(DF_ORIGIN,DF_SCALED)

# Statistical test
## ANOVA
Stats_test(DF_W_LABEL)
## Turkey HSD
Stats_test(DF_W_LABEL, ANOVA = False)
