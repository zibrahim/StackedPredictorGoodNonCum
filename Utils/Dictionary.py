import numpy as np
aggregation = { #ZI Check Aggregation function is appropriate
    'Albumin': 'mean',
    'Creatinine' : 'max',
    'C-Reactive-Protein' : 'max',
    'DiasBP' : 'min',
    'FiO2' : 'np.mean',
    'Hb' : 'min',
    'Lymphocytes' : 'np.mean',
    'Neutrophils': 'np.mean',
    'NEWS2' : 'np.mean',
    'PLT': 'min',
    'PO2/FIO2' : 'np.mean',
    'SysBP' : 'min',
    'Urea' : 'max',
    'WBC' : 'np.mean'
}