import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


# Add binary variable to indicate missing values
class MissingIndicator(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables


    def fit(self, X, y=None):
        # to accommodate sklearn pipeline functionality
        self.vars_with_missing_columns = []
        self.vars_with_missing_columns = [var for var in self.variables if X[var].isnull().values.any()]
        return self


    def transform(self, X):
        # add indicator
        X = X.copy()
        
        for feature in self.vars_with_missing_columns:
            X[feature+'_NA'] = np.where(X[feature].isnull(),1,0)
        return X


# categorical missing value imputer
class CategoricalImputer(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables


    def fit(self, X, y=None):
        # to accommodate sklearn pipeline functionality
        self.vars_cat_with_missing_columns = []
        self.vars_cat_with_missing_columns = [var for var in self.variables if X[var].isnull().values.any()]
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.vars_cat_with_missing_columns:
            X[feature] = X[feature].fillna('Missing')
        return X


# Numerical missing value imputer
class NumericalImputer(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # persist mode in a dictionary
        self.imputer_dict_ = {}
        self.vars_with_missing_columns = []
        self.vars_with_missing_columns = [var for var in self.variables if X[var].isnull().values.any()]
        for feature in self.vars_with_missing_columns:
            self.imputer_dict_[feature] = X[feature].median()
        return self

    def transform(self, X):

        X = X.copy()
        
        for feature in self.vars_with_missing_columns:
        	X[feature] = X[feature].fillna(self.imputer_dict_[feature])
        return X


# Extract first letter from string variable
class ExtractFirstLetter(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
        	X[feature] = X[feature].str[0]

        return X

# frequent label categorical encoder
class RareLabelCategoricalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, tol=0.05, variables=None):

        self.tol = tol

        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):

        # persist frequent labels in dictionary
        self.encoder_dict_ = {}
        for feature in self.variables:
        	tmp = pd.Series(X[feature].value_counts()/np.float(len(X)))
        	self.encoder_dict_[feature] = list(tmp[tmp >= self.tol].index)

        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = np.where(X[feature].isin(self.encoder_dict_[
                    feature]), X[feature], 'Rare')
        return X


# string to numbers categorical encoder
class CategoricalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):

        # HINT: persist the dummy variables found in train set
        self.dummies = pd.get_dummies(X[self.variables], drop_first=True).columns
        
        return self

    def transform(self, X):
        # encode labels
        X = X.copy()
        # get dummies
        for feature in self.variables:
        	X = pd.concat([X,
                         pd.get_dummies(X[feature], prefix=feature, drop_first=True)
                         ], axis=1)
        # drop original variables
        X.drop(self.variables, axis = 1, inplace=True)
        # add missing dummies if any
        for var in self.dummies:
        	if var not in X.columns:
        		X[var] = 0

        return X
