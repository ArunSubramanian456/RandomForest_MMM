
import streamlit as st

import numpy as pandas
import pyarrow.parquet as pq
import pandas as pd
import json

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import RF_MMM_utils
from RF_MMM_utils import *



# Streamlit app
def main():
    st.title('Marketing Mix Results - Dashboard')
    sns.set(font_scale=1.70)
    # Sidebar navigation
    st.sidebar.header('Navigation')
    option = st.sidebar.selectbox('Select an option', ['Home', 'Feature Importance', 'Share of Spend vs Share of Effect', 'Ad Carryover Effect', 'Shap vs Spend'])

    # Load array as dictionary
    prediction_interval = np.load('./output/prediction_interval.npy')
    y_true_interval = np.load('./output/y_true_interval.npy')
    media_channels = list(np.load('./output/media_channels_array.npy'))
    
    # Load Parquet files as dataframe
    parquet_table = pq.read_table('./output/df_model_2.parquet')
    df_model_2 = parquet_table.to_pandas()
    df_model_2['total_media_spend'] = df_model_2[media_channels].sum(axis=1)
    
    parquet_table = pq.read_table('./output/df_shap_values.parquet')
    df_shap_values = parquet_table.to_pandas()
    
    parquet_table = pq.read_table('./output/x_input_interval_transformed.parquet')
    x_input_interval_transformed = parquet_table.to_pandas()
    
    parquet_table = pq.read_table('./output/x_input_interval_nontransformed.parquet')
    x_input_interval_nontransformed = parquet_table.to_pandas()

    organic_features = ["trend", "season_y", "holiday", "events", "newsletter"]

    with open('./output/final_adstock.json', 'r') as f:
        final_adstock = json.load(f)

    # features =  media_channels + organic_features

    # Home page
    if option == 'Home':
        
        sales_result = calculate_yoy(df_model_2, 'y')
        df_sales_period = sales_result[0]
        sales_yoy = sales_result[1]

        fig = go.Figure()
        fig.add_trace(go.Bar(x=df_sales_period.period, y=df_sales_period.y, marker_color = 'grey', name= 'Sales'))
        fig.add_annotation(text=f"{sales_yoy*100:.2f}% YoY", x=0.5, 
                           y= df_sales_period.y.max() + 2,  
                           font = dict(size = 12), showarrow=False, 
                           xanchor='center', yanchor='bottom')

        fig.update_layout(title='YoY Change in Sales', width = 800, height = 400, font = dict(size = 10), 
                          xaxis=dict(title="Period"),
                          yaxis =dict(title='sales'),
                          paper_bgcolor='rgba(255,255,255, 1)',
                          plot_bgcolor='rgba(255,255,255, 1)',
                         )

        spend_result = calculate_yoy(df_model_2, 'total_media_spend')
        df_spend_period = spend_result[0]
        spend_yoy = spend_result[1]

        fig1 = go.Figure()
        fig1.add_trace(go.Bar(x=df_spend_period.period, y=df_spend_period.total_media_spend, marker_color = 'grey', name= 'Media spend'))
        fig1.add_annotation(text=f"{spend_yoy*100:.2f}% YoY", x=0.5, y= df_spend_period.total_media_spend.max() + 2, showarrow=False, xanchor='center', yanchor='bottom')

        fig1.update_layout(title='YoY Change in Ad Spend', width = 800, height = 400, font = dict(size = 14), 
                          xaxis=dict(title="Period"),
                          yaxis =dict(title='Ad Spend'),
                          paper_bgcolor='rgba(255,255,255, 1)',
                          plot_bgcolor='rgba(255,255,255, 1)',
                         )

        st.subheader('YoY Change')
        
        st.plotly_chart(fig)
        st.plotly_chart(fig1)

        

        st.subheader('Organic sales vs Media Incremental sales - PTM vs TTM')

        

        df_shap_values2 = df_shap_values.copy().assign(
            period='ptm',
            shap_organic=lambda df: df.loc[:, organic_features].abs().sum(axis=1),
            shap_media=lambda df: df.loc[:, media_channels].abs().sum(axis=1)
        )
        
        df_shap_values2.iloc[-52:, df_shap_values2.columns.get_loc('period')] = 'ttm'
        
        df_shap_agg = df_shap_values2[['period', 'shap_organic', 'shap_media']].copy()
        df_shap_agg_period = df_shap_agg.groupby('period').sum().reset_index()
        
        df_shap_agg_period['perc_organic'] = df_shap_agg_period['shap_organic'] / (
            df_shap_agg_period['shap_organic'] + df_shap_agg_period['shap_media']
        )
        df_shap_agg_period['perc_media'] = df_shap_agg_period['shap_media'] / (
            df_shap_agg_period['shap_organic'] + df_shap_agg_period['shap_media']
        )
        
        # Create Plotly bar charts
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=df_shap_agg_period['period'], y=df_shap_agg_period['perc_organic'], name='Organic'))
        fig2.add_trace(go.Bar(x=df_shap_agg_period['period'], y=df_shap_agg_period['perc_media'], name='Media'))
        
        # Add text annotations
        for i, (organic, media) in enumerate(zip(df_shap_agg_period['perc_organic'], df_shap_agg_period['perc_media'])):
            fig2.add_annotation(
                text=f"Media contribution = {media*100:.2f}%",
                x=i,  # Adjust x position based on index
                y=media + 1,  # Adjust y position slightly above the bar
                showarrow=False,
                font=dict(size=12)  # Adjust font size as needed
            )
        
        fig2.update_layout(
            title="Organic vs. Media Contribution - PTM vs TTM",
            width = 800,
            height = 400,
            xaxis_title="Period",
            yaxis_title="Percentage Contribution (%)",
            barmode='stack',
            yaxis=dict(tickformat=".2%")
        )
        
        st.plotly_chart(fig2)

        st.write("**Observations:**")
        st.markdown(
            """
            - **Observation 1:** Despite 44% YoY decline in Ad spend, media contribution is up from 12% to 14%.
            - **Observation 2:** Marketing investment declined from being 3.25% of PTM sales to only 1.9% of TTM sales
            """
        )

        st.subheader('Model Overview')

        mape_metric = mean_absolute_percentage_error(y_true =y_true_interval, y_pred = prediction_interval)
        r2_metric = r2_score(y_true = y_true_interval, y_pred = prediction_interval)

        st.markdown(f'**MAPE:** {mape_metric*100:.2f}%')
        st.markdown(f'**RSQUARE:{r2_metric*100:.2f}%')


        
        chart_data = pd.DataFrame({'week' : df_model_2.ds.to_numpy(),
                                   'actual_sales' : y_true_interval,
                                   'predicted_sales' : prediction_interval
                                  })
        
        chart_data['error'] = chart_data['actual_sales'] - chart_data['predicted_sales']
        
        fig2 = go.Figure()
        
        fig2.add_trace(go.Scatter(x=chart_data.week
                                  , y=chart_data["actual_sales"]
                                  , mode = 'lines'
                                  , name = 'Actual sales'
                                  , line=dict(color='coral')))
        
        fig2.add_trace(go.Scatter(x=chart_data.week
                                  , y=chart_data["predicted_sales"]
                                  , mode = 'lines'
                                  , name = 'Predicted sales'
                                  , line=dict(color='blue', dash = 'dash')))
        
        
        # Update layout with labels and title
        fig2.update_layout(title = "Actual vs. Predicted sales",
                           width = 1000,
                           height = 600,
                           showlegend = True,
                           font = dict(size = 14),
                           xaxis=dict(title="week",tickangle=270),
                           yaxis =dict(title='sales'),
                           paper_bgcolor='rgba(255,255,255, 1)',
                           plot_bgcolor='rgba(255,255,255, 1)',
                          )
        
        # Show the plot
        st.plotly_chart(fig2)

        st.write("**Observations:**")
        st.markdown(
            """
            - **Observation 1:** Low MAPE (<10%) and strong RSQ (~90%)
            - **Observation 2:** Except for missing 4 big spikes that are not seasonal, predicted sales tracks the actual sales well.
            - **Observation 3:** Additional business understanding and dive deep required to fix the prediction misses for those 4 big spikes
            """
        )
        

    # Other pages (implement as needed)
    elif option == 'Feature Importance':
        st.subheader('Feature Importance')
        st.pyplot(shap_feature_importance(df_shap_values, x_input_interval_transformed, figsize = (16, 8)))

        st.write("**Observations:**")
        st.markdown(
            """
            - **Observation 1:** Seasonality is the largest driver of sales
            - **Observation 2:** Among the media channels, Facebook, Search and TV are the top 3 drivers
            - **Observation 3:** Holiday and events has the least effect on sales
            """
        )

    elif option == 'Share of Spend vs Share of Effect':
    
        st.subheader('Share of Spend vs. Share of Effect')
        spend_effect_share = calculate_spend_effect_share(df_shap_values = df_shap_values
                                                       , media_channels = media_channels
                                                       , df_original = x_input_interval_nontransformed)
        
        decomp_rssd = rssd(effect_share = spend_effect_share.effect_share.values, spend_share = spend_effect_share.spend_share.values)
        print(f"DECOMP.RSSD: {decomp_rssd}")
        st.pyplot(plot_spend_vs_effect_share(spend_effect_share, figure_size = (15, 7)))

        st.write("**Observations:**")
        st.markdown(
            """
            - **Observation 1:** Facebook,Search and Print drive more share of effect compare of their spend share
            - **Observation 2:** OOH is the least effective channel in driving any effect despite being the highest spend channel
            - **Observation 3:** TV spend share and effect share are more or less proportional
            """
        )

    
        st.subheader('Incremental ROAS by Media Channels')

        st.pyplot(plot_incr_roas(spend_effect_share, figure_size = (16, 8)))

        st.write("**Observations:**")
        st.markdown(
            """
            - **Observation 1:** Facebook delivers the highest incremental ROAS of 5.9 followed by Print and Search (both delivering 1.8 approximately)
            - **Observation 2:** OOH has the least incremental ROAS
            - **Observation 3:** TV incremental ROAS is below breakeven
            """
        )
         
    elif option == 'Ad Carryover Effect':

        st.subheader('Ad Carryover Effect')
        st.plotly_chart(plot_adcarryover_effect(final_adstock))

        st.write("**Observations:**")
        st.markdown(
            """
            - **Observation 1:** TV has the longest carryover effect. A 100 dollars ad spend in week 1 will have 6 dollars carryover effect in week 8
            - **Observation 2:** Search has the least carryover effect. A 100 dollars ad spend in week 1 will have 2 dollars carryover effect in week 3
            """
        )
    
        
    else:
        st.subheader('Media Response curves using SHAP values')

        selected_channel = st.selectbox("Select a Media Channel", media_channels)
        st.pyplot(plot_shap_vs_spend(df_shap_values, x_input_interval_nontransformed, x_input_interval_transformed, selected_channel))
        

if __name__ == "__main__":
    main()