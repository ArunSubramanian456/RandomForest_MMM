import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna as opt
from functools import partial
import plotly.graph_objects as go

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

import warnings
import shap


class AdstockGeometric(BaseEstimator, TransformerMixin):
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        
    def fit(self, X, y=None):
        # Input validation on an array
        # By default, the input is checked to be a non-empty 2D array containing only finite values. 
        # If the dtype of the array is object, attempt converting to float, raising on failure.
        X = check_array(X)
         
        # If reset = True, the `n_features_in_` attribute is set to `X.shape[1]`.
        # Else, the attribute must already exist and the function checks
        # that it is equal to `X.shape[1]`.
        self._check_n_features(X, reset=True)
        return self
    
    def transform(self, X: np.ndarray):
        check_is_fitted(self)
        X = check_array(X)
        self._check_n_features(X, reset=False)
        x_decayed = np.zeros_like(X)
        x_decayed[0] = X[0]
        
        for xi in range(1, len(x_decayed)):
            x_decayed[xi] = X[xi] + self.alpha* x_decayed[xi - 1]
        return x_decayed


def plot_adcarryover_effect(adstock_dict):
    
    from RF_MMM_utils import AdstockGeometric

    # Simulate ad spend pulse
    adspend = {key : np.array([100] + [0] * 12).reshape(-1,1) for key in adstock_dict.keys()}

    # Apply adstock
    adstock = {}
    for key, value in adstock_dict.items():
        adstock_instance = AdstockGeometric(alpha=value)
        adstock[key] = (adstock_instance.fit_transform(adspend[key])).flatten() / 100

    # Create Plotly bar charts
    fig2 = go.Figure()
    for channel, effect in adstock.items():
        fig2.add_trace(go.Bar(x=list(range(1,14)), y=list(effect), name=channel))
    
    fig2.update_layout(
        title="Ad Carryover Effect",
        showlegend=True,
        width = 1000,
        height = 600,
        xaxis_title="Week",
        yaxis_title="Percent Carryover",
        barmode='group',
        yaxis=dict(tickformat=".2%")
    )

    return fig2
    
    
def nrmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2)) / (np.max(y_true) - np.min(y_true))

#https://github.com/facebookexperimental/Robyn

def rssd(effect_share, spend_share):
    """RSSD decomposition

    Decomposition distance (root-sum-square distance, a major innovation of Robyn) 
    eliminates the majority of "bad models" 
    (larger prediction error and/or unrealistic media effect like the smallest channel getting the most effect

    Business fit: Aim to minimize decomposition distance (DECOMP.RSSD, decomposition root-sum-square distance). 
    The distance accounts for a relationship between spend share and a channel’s coefficient decomposition share. 
    If the distance is too far, its result can be too unrealistic - e.g. media activity with the smallest spending gets the largest effect.

    Args:
        effect_share ([type]): percentage of effect share
        spend_share ([type]): percentage of spend share

    Returns:
        [type]: [description]
    """
    return np.sqrt(np.sum((effect_share - spend_share) ** 2))


def plot_spend_vs_effect_share(decomp_spend: pd.DataFrame, figure_size = (12, 8)):
    """Spend vs Effect Share plot

    Args:
        decomp_spend (pd.DataFrame): Data with media decompositions. The following columns should be present: media, spend_share, effect_share per media variable
        figure_size (tuple, optional): Figure size. Defaults to (15, 10).

    Example:
        decomp_spend:
        media         spend_share effect_share
        tv_S           0.31        0.44
        ooh_S          0.23        0.34
    
    Returns:
        Seaborn plot
    """
    
    plot_spend_effect_share = decomp_spend.melt(id_vars = ["media"], value_vars = ["spend_share", "effect_share"])

    fig, ax = plt.subplots(figsize = figure_size)
    sns.barplot(data=plot_spend_effect_share, x="media", y="value", hue="variable", dodge=True, orient='v')

    # Add labels and titles
    plt.title("Share of Spend VS Share of Effect")
    plt.ylabel("Share")
    plt.xlabel("Media channel")
    plt.legend(loc="best")

    
    # Hide y-axis labels
    ax.set_yticklabels([])

    # Add percentage labels
    for i in ax.containers:
        for rect in i:
            height = rect.get_height()
            plt.annotate(f"{height*100:.2f}%", xy=(rect.get_x() + rect.get_width() / 2, height), ha='center', va='bottom')

    # Adjust plot layout
    plt.xticks(rotation=90)
    
    plt.show();


def plot_incr_roas(decomp_spend: pd.DataFrame, figure_size = (12, 8)):
    """Incremental ROAS plot

    Args:
        decomp_spend (pd.DataFrame): Data with media decompositions. The following columns should be present: media, incr_roas
        figure_size (tuple, optional): Figure size. Defaults to (15, 10).
    
    Returns:
        Seaborn plot
    """
    fig, ax = plt.subplots(figsize = figure_size)
    sns.barplot(x='media', y='incr_roas', data = decomp_spend, hue="media");
    
    # Add labels and titles
    plt.title("Incremental ROAS by media channels")
    plt.ylabel("Incremental ROAS")
    plt.xlabel("Media channel")
    
    
    # Hide y-axis labels
    ax.set_yticklabels([])
    
    # Add percentage labels
    for i in ax.containers:
        for rect in i:
            height = rect.get_height()
            plt.annotate(f"{height:.2f}", xy=(rect.get_x() + rect.get_width() / 2, height), ha='center', va='bottom')

    plt.show()
    

def calculate_spend_effect_share(df_shap_values: pd.DataFrame, media_channels, df_original: pd.DataFrame):
    """
    Args:
        df_shap_values: data frame of shap values
        media_channels: list of media channel names
        df_original: non transformed original data
    Returns: 
        [pd.DataFrame]: data frame with spend effect shares
    """
    responses = pd.DataFrame(df_shap_values[media_channels].abs().sum(axis = 0), columns = ["effect_share"])
    response_percentages = responses / responses.sum()
    response_percentages

    spends_percentages = pd.DataFrame(df_original[media_channels].sum(axis = 0) / df_original[media_channels].sum(axis = 0).sum(), columns = ["spend_share"])
    spends_percentages

    spend_effect_share = pd.merge(response_percentages, spends_percentages, left_index = True, right_index = True)
    spend_effect_share = spend_effect_share.reset_index().rename(columns = {"index": "media"})
    spend_effect_share['incr_roas'] = spend_effect_share['effect_share'] / spend_effect_share['spend_share']
    
    return spend_effect_share


#https://towardsdatascience.com/explain-your-model-with-the-shap-values-bc36aac4de3d
def shap_feature_importance(shap_values, data, figsize = (12, 8)):

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

    
        feature_list = data.columns
        
        if isinstance(shap_values, pd.DataFrame) == False:
            shap_v = pd.DataFrame(shap_values)
            shap_v.columns = feature_list
        else:
            shap_v = shap_values
        
            
        df_v = data.copy().reset_index().drop('index',axis=1)
        
        # Determine the correlation in order to plot with different colors
        corr_list = list()
        for i in feature_list:
            b = np.corrcoef(shap_v[i],df_v[i])[1][0]
            corr_list.append(b)
        corr_df = pd.concat([pd.Series(feature_list),pd.Series(corr_list)],axis=1).fillna(0)
        # Make a data frame. Column 1 is the feature, and Column 2 is the correlation coefficient
        corr_df.columns  = ['Variable','Corr']
        corr_df['Sign'] = np.where(corr_df['Corr']>0,'green','red')
        
        # Plot it
        shap_abs = np.abs(shap_v)
        k=pd.DataFrame(shap_abs.mean()).reset_index()
        k.columns = ['Variable','SHAP_abs']
        k2 = k.merge(corr_df,left_on = 'Variable',right_on='Variable',how='inner')
        k2 = k2.sort_values(by='SHAP_abs',ascending = True)
        colorlist = k2['Sign']

        fig, ax = plt.subplots(figsize=figsize)
        ax = k2.plot.barh(x='Variable', y='SHAP_abs', color='grey', figsize=figsize, legend=False, ax=ax)
        ax.set_xlabel("SHAP Value")

        fig.show()



    
def model_refit(data, 
                target, 
                features, 
                media_channels, 
                organic_channels, 
                model_params, 
                adstock_params
               ):
    data_refit = data.copy()

    best_params = model_params

    adstock_alphas = adstock_params

    #apply adstock transformation
    for feature in media_channels + organic_channels:
        adstock_alpha = adstock_alphas[feature]
        print(f"applying geometric adstock transformation on {feature} with alpha {adstock_alpha}") 

        #adstock transformation
        x_feature = data_refit[feature].values.reshape(-1, 1)
        temp_adstock = AdstockGeometric(alpha = adstock_alpha).fit_transform(x_feature)
        data_refit[feature] = temp_adstock

    #build the final model on the data
    x_input = data_refit.loc[0:, features]
    y_true_all = data[target].values[0:]

    #build random forest using the best parameters
    random_forest = RandomForestRegressor(random_state=0, **best_params)
    random_forest.fit(x_input, y_true_all) 


    #concentrate on the analysis interval
    y_true_interval = y_true_all
    x_input_interval_transformed = x_input

    #revenue prediction for the analysis interval
    print(f"predicting {len(x_input_interval_transformed)}")
    prediction = random_forest.predict(x_input_interval_transformed)

    #transformed data set for the analysis interval 
    x_input_interval_nontransformed = data

    #shap explainer 
    explainer = shap.TreeExplainer(random_forest)

    # get SHAP values for the data set for the analysis interval from explainer model
    shap_values_train = explainer.shap_values(x_input_interval_transformed)

    # create a dataframe of the shap values for the training set and the test set
    df_shap_values = pd.DataFrame(shap_values_train, columns=features)
    
    return {
            'df_shap_values': df_shap_values, 
            'x_input_interval_nontransformed': x_input_interval_nontransformed, 
            'x_input_interval_transformed' : x_input_interval_transformed,
            'prediction_interval': prediction, 
            'y_true_interval': y_true_interval
           }
    
def plot_shap_vs_spend(df_shap_values, x_input_interval_nontransformed, x_input_interval_transformed, channel, figsize=(16,8)):

    mean_spend = x_input_interval_nontransformed.loc[x_input_interval_nontransformed[channel] > 0, channel].mean()
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.regplot(x = x_input_interval_transformed[channel], y = df_shap_values[channel], label = channel,
                scatter_kws={'alpha': 0.65}, line_kws={'color': 'C2', 'linewidth': 6},
                lowess=True, ax=ax).set(title=f'{channel}: Spend vs Shapley')
    ax.axhline(0, linestyle = "--", color = "black", alpha = 0.5)
    ax.axvline(mean_spend, linestyle = "--", color = "red", alpha = 0.5, label=f"Average Spend: {int(mean_spend)}")
    ax.set_xlabel(f"{channel} spend")
    ax.set_ylabel(f'SHAP Value for {channel}')
    ax.legend()

    plt.show()

def optuna_trial(trial, 
                 data:pd.DataFrame, 
                 target, 
                 features, 
                 adstock_features, 
                 adstock_features_params, 
                 media_features, 
                 tscv, 
                 is_multiobjective = False):
    
    data_temp = data.copy()
    adstock_alphas = {}
    
    for feature in adstock_features:
        adstock_param = f"{feature}_adstock"
        min_, max_ = adstock_features_params[adstock_param]
        adstock_alpha = trial.suggest_float(f"adstock_alpha_{feature}", min_, max_)
        adstock_alphas[feature] = adstock_alpha
        
        #adstock transformation
        x_feature = data[feature].values.reshape(-1, 1)
        temp_adstock = AdstockGeometric(alpha = adstock_alpha).fit_transform(x_feature)
        data_temp[feature] = temp_adstock
        
        
    #Random Forest parameters
    n_estimators = trial.suggest_int("n_estimators", 5, 100)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    max_depth = trial.suggest_int("max_depth", 4,7)
    ccp_alpha = trial.suggest_float("ccp_alpha", 0, 0.3)
    bootstrap = trial.suggest_categorical("bootstrap", [False, True])
    criterion = trial.suggest_categorical("criterion", ["squared_error"])  #"absolute_error"
    
    scores = []
    
    rssds = []
    for train_index, test_index in tscv.split(data_temp):
        x_train = data_temp.iloc[train_index][features]
        y_train =  data_temp[target].values[train_index]
        
        x_test = data_temp.iloc[test_index][features]
        y_test = data_temp[target].values[test_index]
        
        #apply Random Forest
        params = {"n_estimators": n_estimators, 
                   "min_samples_leaf":min_samples_leaf, 
                   "min_samples_split" : min_samples_split,
                   "max_depth" : max_depth, 
                   "ccp_alpha" : ccp_alpha, 
                   "bootstrap" : bootstrap, 
                   "criterion" : criterion
                 }
        
        rf = RandomForestRegressor(random_state=0, **params)
        rf.fit(x_train, y_train)
        prediction = rf.predict(x_test)
        
        mape = mean_absolute_percentage_error(y_true = y_test, y_pred = prediction)
        scores.append(mape)
        
        if is_multiobjective:
            
            #set_trace()
            #calculate spend effect share -> rssd
            # create explainer model by passing trained model to shap
            explainer = shap.TreeExplainer(rf)

            # get Shap values
            shap_values_train = explainer.shap_values(x_train)
            
            df_shap_values = pd.DataFrame(shap_values_train, columns=features)

            spend_effect_share = calculate_spend_effect_share(df_shap_values = df_shap_values, media_channels = media_features, df_original = data.iloc[train_index])

            decomp_rssd = rssd(effect_share = spend_effect_share.effect_share.values, spend_share = spend_effect_share.spend_share.values)
            rssds.append(decomp_rssd)
    
    trial.set_user_attr("scores", scores)
    
    trial.set_user_attr("params", params)
    trial.set_user_attr("adstock_alphas", adstock_alphas)
    
    if is_multiobjective == False:
        return np.mean(scores)
    
    
    trial.set_user_attr("rssds", rssds)
        
    #multiobjective
    return np.mean(scores), np.mean(rssds)


def optuna_optimize(trials, 
                    data: pd.DataFrame, 
                    target, 
                    features, 
                    adstock_features, 
                    adstock_features_params, 
                    media_features, 
                    tscv, 
                    is_multiobjective, 
                    seed = 42):
    print(f"data size: {len(data)}")
    print(f"media features: {media_features}")
    print(f"adstock features: {adstock_features}")
    print(f"features: {features}")
    print(f"is_multiobjective: {is_multiobjective}")
    opt.logging.set_verbosity(opt.logging.WARNING) 
    
    if is_multiobjective == False:
        study_mmm = opt.create_study(direction='minimize', sampler = opt.samplers.TPESampler(seed=seed))  
    else:
        study_mmm = opt.create_study(directions=["minimize", "minimize"], sampler=opt.samplers.NSGAIISampler(seed=seed))
    
    # Partial functions in Python are a way to create new functions from existing ones by fixing some of the arguments. 
    # This allows you to create functions with predefined arguments, making them more flexible and reusable.
    
    optimization_function = partial(optuna_trial, # existing function
                                    data = data, 
                                    target = target, 
                                    features = features, 
                                    adstock_features = adstock_features, 
                                    adstock_features_params = adstock_features_params, 
                                    media_features = media_features, 
                                    tscv = tscv, 
                                    is_multiobjective = is_multiobjective)
    
    
    study_mmm.optimize(optimization_function, n_trials = trials, show_progress_bar = True)
    
    return study_mmm
    

def calculate_incr_effect(data, media_channel_of_interest, media_channels, target, features, organic_channels, model_params, adstock_params):

    from RF_MMM_utils import model_refit

    data_modified = data.copy()

    # Zero out other media channels
    for channel in media_channels:
      if channel != media_channel_of_interest:
        data_modified[channel] = 0

    # Get predictions with simulated spend
    predictions_with_spend = model_refit(data_modified, target, features, media_channels, organic_channels, model_params, adstock_params)["prediction_interval"]

    # Zero out the media channel of interest
    data_modified_2 = data_modified.copy()
    data_modified_2[media_channel_of_interest] = 0

    # Get predictions without the media channel
    predictions_without_spend =model_refit(data_modified_2, target, features, media_channels, organic_channels, model_params, adstock_params)["prediction_interval"]

    # Calculate incremental effect
    incremental_effect = (predictions_with_spend - predictions_without_spend)

    chart_data = pd.DataFrame({'spend' : data_modified[media_channel_of_interest],
                               'incr_effect' : incremental_effect
                              })

    chart_data = chart_data[chart_data.spend>0]
    mean_spend = data_modified.loc[data_modified[media_channel_of_interest] > 0, media_channel_of_interest].mean()

    print(f'Average media spend :{mean_spend}')


    # Plot the simulated spend vs incremental effect
    fig, ax = plt.subplots(figsize=(12,8))
    sns.regplot(data = chart_data, x = 'spend', y = 'incr_effect', scatter_kws={'alpha': 0.65}, line_kws={'color': 'C2', 'linewidth': 6},
                    lowess=True, ax=ax).set(title=f'{media_channel_of_interest}: Spend vs Shapley')
    ax.axhline(0, linestyle = "--", color = "black", alpha = 0.5)
    ax.axvline(mean_spend, linestyle = "--", color = "red", alpha = 0.5, label=f"Average Spend: {int(mean_spend)}")
    ax.set_xlabel(f"{media_channel_of_interest} spend")
    ax.set_ylabel(f'SHAP values for {media_channel_of_interest}')
    plt.legend()
    plt.show();


# Function to calculate YoY change

def calculate_yoy(df, column):
    period_col = df.groupby('period')[column].sum()
    yoy = period_col.diff() / period_col.shift()
    return (period_col.reset_index(), yoy.loc['ttm'])
    
