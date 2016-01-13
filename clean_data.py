# Clean data for predicting

import pandas as pd
import numpy as np
import scipy.sparse as sparse
import pickle
import os
import ipdb
import gc
from sklearn.cluster import MiniBatchKMeans
from joblib import Parallel, delayed

from departments import all_departments, top_upcs

train = test = labels = features = None

gross_categories = {'Food': ['BAKERY', 'CANDY, TOBACCO, COOKIES', 'COMM BREAD', 'COOK AND DINE', 'DAIRY',
                             'DSD GROCERY', 'FROZEN FOODS', 'GROCERY DRY GOODS', 'MEAT - FRESH & FROZEN',
                             'PRE PACKED DELI', 'PRODUCE', 'SEAFOOD', 'SERVICE DELI'],
                    'Clothing': ['ACCESSORIES', 'BOYS WEAR', 'BRAS & SHAPEWEAR', 'GIRLS WEAR, 4-6X  AND 7-14',
                                 'INFANT APPAREL', 'JEWELRY AND SUNGLASSES', 'LADIES SOCKS', 'LADIESWEAR',
                                 'MENS WEAR', 'MENSWEAR', 'PLUS AND MATERNITY', 'SHEER HOSIERY', 'SHOES',
                                 'SLEEPWEAR/FOUNDATIONS', 'SWIMWEAR/OUTERWEAR'],
                    'Electronics': ['1-HR PHOTO', 'CAMERAS AND SUPPLIES', 'ELECTRONICS', 'MEDIA AND GAMING',
                                    'OTHER DEPARTMENTS', 'PLAYERS AND ELECTRONICS', 'WIRELESS'],
                    'Do_It_Yourself': ['AUTOMOTIVE', 'HARDWARE', 'LAWN AND GARDEN', 'PAINT AND ACCESSORIES'],
                    'Hobbies': ['BOOKS AND MAGAZINES', 'CELEBRATION', 'FABRICS AND CRAFTS',
                                'HORTICULTURE AND ACCESS', 'OFFICE SUPPLIES', 'TOYS'],
                    'Household': ['BATH AND SHOWER', 'BEDDING', 'COOK AND DINE', 'FURNITURE', 'HOME DECOR',
                                  'HOME MANAGEMENT', 'HORTICULTURE AND ACCESS', 'HOUSEHOLD CHEMICALS/SUPP',
                                  'HOUSEHOLD PAPER GOODS', 'LARGE HOUSEHOLD GOODS'],
                    'Specialty': ['1-HR PHOTO', 'CONCEPT STORES', 'LARGE HOUSEHOLD GOODS', 'OPTICAL - FRAMES',
                                  'OPTICAL - LENSES', 'OTHER DEPARTMENTS', 'PHARMACY RX'],
                    'Discretionary': ['1-HR PHOTO', 'ACCESSORIES', 'AUTOMOTIVE', 'BATH AND SHOWER', 'BEAUTY',
                                      'BOOKS AND MAGAZINES', 'CAMERAS AND SUPPLIES', 'CANDY, TOBACCO, COOKIES',
                                      'CELEBRATION', 'CONCEPT STORES', 'COOK AND DINE', 'ELECTRONICS',
                                      'FABRICS AND CRAFTS', 'HEALTH AND BEAUTY AIDS', 'HOME DECOR',
                                      'HORTICULTURE AND ACCESS', 'IMPULSE MERCHANDISE', 'JEWELRY AND SUNGLASSES',
                                      'LAWN AND GARDEN', 'LIQUOR,WINE,BEER', 'MEDIA AND GAMING',
                                      'OPTICAL - FRAMES', 'PLAYERS AND ELECTRONICS', 'SPORTING GOODS',
                                      'SWIMWEAR/OUTERWEAR', 'TOYS', 'WIRELESS'],
                    'Non_Discretionary': ['BAKERY', 'BATH AND SHOWER', 'BEAUTY', 'BEDDING', 'BRAS & SHAPEWEAR',
                                          'COMM BREAD', 'COOK AND DINE', 'DAIRY', 'DSD GROCERY',
                                          'FINANCIAL SERVICES', 'GROCERY DRY GOODS', 'HEALTH AND BEAUTY AIDS',
                                          'HOUSEHOLD CHEMICALS/SUPP', 'HOUSEHOLD PAPER GOODS',
                                          'INFANT CONSUMABLE HARDLINES', 'LADIES SOCKS', 'MEAT - FRESH & FROZEN',
                                          'MENS WEAR', 'MENSWEAR', 'OFFICE SUPPLIES', 'OPTICAL - FRAMES',
                                          'OPTICAL - LENSES', 'PERSONAL CARE', 'PETS AND SUPPLIES', 'PHARMACY OTC',
                                          'PHARMACY RX', 'PLUS AND MATERNITY', 'PRE PACKED DELI', 'PRODUCE',
                                          'SEAFOOD', 'SERVICE DELI', 'SHEER HOSIERY', 'SHOES',
                                          'SLEEPWEAR/FOUNDATIONS', 'SWIMWEAR/OUTERWEAR'],
                    'Sin': ['CANDY, TOBACCO, COOKIES', 'IMPULSE MERCHANDISE', 'LIQUOR,WINE,BEER']}


# Create a version of each category prefixed with "-" to indicate returns
for k, v in gross_categories.copy().items():
    gross_categories['-' + k] = ['-' + dept for dept in v]
    

def load():
    "Load the input data set."
    
    print('Reading data...')
    train = pd.read_csv('input/train.csv')
    test = pd.read_csv('input/test.csv')
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    return train, test


def load_mini():
    "Load a very miniature version of the train and test set (for development and debugging)"

    print("Using mini set...")
    train, test = load()
    return train[:5000], test[:5000]


def has_labels(df):
    "Answer true if the given dataframe has a target label column."
    
    return 'TripType' in df.columns

    
def pivot_data(df):
    "Pivot data tables so each row represents one trip."

    df['DepartmentDescription'] = df.apply(lambda row: (np.nan if
                                                        pd.isnull(row['DepartmentDescription'])
                                                        else '-' + row['DepartmentDescription']) if
                                           row['ScanCount'] < 0 else
                                           row['DepartmentDescription'], axis=1)

    dept_pivot_df = (df.pivot_table('ScanCount', ['VisitNumber'], ['DepartmentDescription'], aggfunc=sum))

    fln_upc_pivot_df = df[['ScanCount', 'VisitNumber', 'FinelineNumber', 'Upc']].copy()

    # Filter out all but the top UPC codes (otherwise too big)
    fln_upc_pivot_df['Upc'] = fln_upc_pivot_df['Upc'].apply(lambda upc: upc if upc in top_upcs else np.nan)

    fln_upc_pivot_df.to_sparse(fill_value=0.0)

    # print('\tpivoting fln...')
    fln = fln_upc_pivot_df.pivot_table('ScanCount', ['VisitNumber'], ['FinelineNumber'], aggfunc=sum)
    fln.rename(columns={n: 'FLN_{}'.format(int(n)) for n in fln.columns}, inplace=True)
    fln.to_sparse(fill_value=0.0)

    drop_trip_type = False
    if has_labels(df):
        trip_type = df[['VisitNumber', 'TripType']].drop_duplicates(subset='VisitNumber')
        result = trip_type.join(dept_pivot_df, on='VisitNumber')
    else:
        df['TripType'] = 0.0
        trip_type = df[['VisitNumber', 'TripType']].drop_duplicates(subset='VisitNumber')
        result = trip_type.join(dept_pivot_df, on='VisitNumber')
        drop_trip_type = True

    result = result.join(fln, on='VisitNumber')
    fln = None
    gc.collect()
    
    # print('\tpivoting upc...')
    upc = fln_upc_pivot_df.pivot_table('ScanCount', ['VisitNumber'], ['Upc'], aggfunc=sum)
    upc.rename(columns={n: 'UPC_{}'.format(int(n)) for n in upc.columns}, inplace=True)
    upc.to_sparse(fill_value=0.0)
    result = result.join(upc, on='VisitNumber')
    upc = None
    gc.collect()

    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    wd = df[['VisitNumber', 'Weekday']].drop_duplicates(subset='VisitNumber')
    wd['Weekday'] = wd['Weekday'].apply(lambda x: weekdays.index(x))

    # Returns column: 1 indicates there are returns in this visit, 0 otherwise
    df['Returns'] = df['ScanCount'].apply(lambda x: 1 if x < 0 else 0)
    rtns = df.pivot_table('Returns', ['VisitNumber'], aggfunc=sum)
    

    result['Returns'] = rtns
    result['Weekday'] = wd['Weekday']
    result['Mon_To_Fri'] = result['Weekday'].apply(lambda x: 1 if x < 5 else 0)

    for category, departments in gross_categories.items():
        depts = list(set(departments).intersection(set(result.columns)))
        result[category] = result[depts].sum(axis=1)
    
    result2 = result.fillna(0.0)
    if drop_trip_type:
        result2.drop('TripType', axis=1, inplace=True)
        
    return result2
    

def extract_labels(df):
    "Extract training labels from the given dataframe."

    assert has_labels(df), 'No labels in the given dataframe'
    labels = df['TripType']
    df.drop('TripType', axis=1, inplace=True)
    return labels


def parallel_apply_frequent_itemsets(df):
    """
    Create a column 'ItemSet' in df and fill it with itemsets matching each row or 0 if none.
    Do this as a parallelized computation using all available cores.
    """

    # Helper functions for parallel apply
    def itemset_for_row(row, itemset_map):
        "Answer the id of an itemset if a subset of row's items make it up. Answer nan otherwise."
        columns = [column for column in all_departments if column in row.index and row[column] > 0]

        for i in range(len(columns), 1, -1):
            subset = tuple(columns[:i])
            if subset in itemset_map:
                return itemset_map[subset]
        return float('nan')

        
    def generate_rows(df):
        "Generate rows of the given dataframe, one at a time (with progress logging)"
        for i in range(df.shape[0]):
            if i % 1000 == 0:
                print(i)
            yield df.iloc[i]


    with open('cleaned/itemsets.txt', 'r') as f:
        lines = [line[:-1].split(';') for line in f.readlines()]

    lines = [[item for item in line if item != 'NULL'] for line in lines]
    itemset_map = {tuple(sorted(line[1:])): int(line[0]) for line in lines}
    
    df['ItemSet'] = float('nan')

    print('Processing {} rows...'.format(df.shape[0]))
    
    results = Parallel(n_jobs=-1)(delayed(itemset_for_row)(row, itemset_map) for row in generate_rows(df))
    df['ItemSet'] = results
    df['ItemSet'].fillna(0.0, inplace=True)
    print(df['ItemSet'].value_counts())
    

def train_cluster_departments(train, labels):
    "Use a clustering algorithm to develop a set of clusters of departments based on trip type."

    print('Training department clusters...')
    clf = MiniBatchKMeans(n_clusters=8, verbose=1, random_state=0)
    train_X = train[all_departments].values
    train_y = labels.values

    clf.fit(train_X, train_y)
    return clf


def apply_cluster_departments(clf, df, name):
    "Use the given trained classifier to apply department clusters to the given dataframe."

    print('applying department clusters to {}...'.format(name))
    result = clf.predict(df[all_departments].values)
    df['Dept_Category'] = result


    
def write(directory='cleaned/'):
    "Write cleaned data."

    print('Writing data...')

    # Shuffle train and labels
    global train, labels
    train['TripType'] = labels
    train = train.iloc[np.random.permutation(len(train))]
    labels = train['TripType']
    train.drop('TripType', inplace=True, axis=1)
    
    labels.to_csv(directory + 'labels.csv', index=False, header=True)

    features = pd.DataFrame(data=None, columns=train.columns, index=None)
    features.to_csv(directory + 'features.csv', index=False, header=True)

    test_ids = test['VisitNumber']
    test_ids.to_csv(directory + 'test_ids.csv', index=False, header=True)

    for df in [train, test]:
        df.drop('VisitNumber', axis=1, inplace=True)

    with open(directory + 'train.p', 'wb') as f:
        pickle.dump(sparse.csr_matrix(train.values), f)

    with open(directory + 'test.p', 'wb') as f:
        pickle.dump(sparse.csr_matrix(test.values), f)

    train.to_csv(directory + 'train.csv', index=False, header=True)
    os.system('rm ' + directory + 'train.csv.gz')
    os.system('gzip ' + directory + 'train.csv')

    test.to_csv(directory + 'test.csv', index=False, header=True)
    os.system('rm ' + directory + 'test.csv.gz')
    os.system('gzip ' + directory + 'test.csv')



def run(mini=False):
    global train, test, labels
    if mini:
        train, test = load_mini()
    else:
        train, test = load()
        
    train_ids = train['VisitNumber']
    test_ids = test['VisitNumber']

    both = pd.concat([train, test])

    print("pivoting both datasets...")
    both = pivot_data(both)

    print("Applying frequent itemsets...")
    parallel_apply_frequent_itemsets(both)

    train = both[both['VisitNumber'].isin(train_ids)]
    labels = extract_labels(train)

    test = both[both['VisitNumber'].isin(test_ids)]
    extract_labels(test)

    clf = train_cluster_departments(train, labels)
    apply_cluster_departments(clf, train, 'train')
    apply_cluster_departments(clf, test, 'test')

    write()
