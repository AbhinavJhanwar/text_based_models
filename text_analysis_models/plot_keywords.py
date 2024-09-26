# %%
import pandas as pd
import plotly.express as px

from tqdm import tqdm
tqdm.pandas()

from nltk.util import ngrams

def plot_bubble_chart(data: pd.core.frame.DataFrame, 
                    x_column: str, 
                    y_column: str, 
                    bubble_size: str, 
                    color: str, 
                    save_file: str,
                    max_size: float, 
                    title: str) -> None:
    """Plots the bubble chart of the given input data and saves it in the given dir

    Args:
        data (pd.core.frame.DataFrame): _description_
        x_column (str): _description_
        y_column (str): _description_
        bubble_size (str): _description_
        color (str): _description_
        save_file (str): _description_
        max_size (float): _description_
        title (str): _description_
    """

    data['Top Keywords'] = data[color].apply(lambda x: x[:50])
    data.sort_values('importance', ascending=False, inplace=True)
    fig = px.scatter(data, 
                    x=x_column, 
                    y=y_column, 
                    color='Top Keywords',
                    size=bubble_size,
                    size_max=max_size
                    )

    fig.update_layout(
        title=title,
        xaxis=dict(
            title=x_column,
            gridcolor='white',
            type='log',
            gridwidth=2,
        )
    )
    fig.write_html(f"{save_file}.html")

def plot_data(data: pd.core.frame.DataFrame, 
            keyword_column: str, 
            weights_column: str,  
            plot_type: str='phrase', 
            x_axis: str='count',
            y_axis: str='importance', 
            bubble_size: str='count', 
            title_text: str='Scatter Plot',
            save_file: str='scatter_plot', 
            sort_data: str='count', 
            min_size: float=0.001,
            number_of_keywords_to_plot: int=20) -> None:
    """_summary_

    Args:
        data (pd.core.frame.DataFrame): input dataframe
        keyword_column (str): column containing keywords
        weights_column (str): column containing weights for respective keywords
        plot_type (str, optional): choose from unigram, bigram, trigram and phrase. Defaults to 'phrase'.
        x_axis (str, optional): variable to keep as x axis to be selected from importance/count. Defaults to 'count'.
        y_axis (str, optional): variable to keep as y axis to be selected from importance/count. Defaults to 'importance'.
        bubble_size (str, optional): column to decide bubble size. Defaults to 'count'.
        title_text (str, optional): title text. Defaults to 'Scatter Plot'.
        save_file (str, optional): name of the file to save as. Defaults to 'scatter_plot'.
        sort_data (str, optional): column as per data to sort. Defaults to 'count'.
        min_size (float, optional): minimum size of bubble. Defaults to 0.001.
        number_of_keywords_to_plot (int, optional): integer indicating the number of keywords to plot, -1 for all. Defaults to 20.
    """
    
    # calculate importance of phrases across product by taking mean of all the weights corresponding to a phrase
    data['importance'] = data.groupby([keyword_column])[weights_column].transform('mean')
    
    # get total count of each phrase
    data['count'] = data.groupby([keyword_column])[keyword_column].transform('count')
    
    if bubble_size not in ['count', 'importance']:
        # calculate rank of phrases in similar way
        data[bubble_size] = data.groupby([keyword_column])[bubble_size].transform('mean')
    # if for some reason x_min and x_max are same then modify the method
    # normalize the bubble size variable
    try:
        x_min= min(data[bubble_size])
        x_max= max(data[bubble_size])
        data[bubble_size] = data[bubble_size].apply(lambda x: (x-x_min)/(x_max-x_min)).replace(0, min_size)
    except:
        data[bubble_size] = data[bubble_size].replace(x_min, 0)
    
    # remove duplicates
    data.drop_duplicates([keyword_column, 'importance', 'count'], inplace=True)

    if plot_type=='unigram':
        # convert phrases into unigrams 
        data[keyword_column] = data[keyword_column].apply(lambda x: x.split())

        # get weighted importance as per count of words
        data['importance'] = data.apply(lambda x: x['importance']/len(x[keyword_column]), axis=1)

        # expand keywords to unigrams
        data = data.explode(keyword_column).reset_index(drop=True)

        # calculate importance of phrases across data by taking mean of all the weights corresponding to a phrase
        data['importance'] = data.groupby([keyword_column])['importance'].transform('mean')

        # get total count of each phrase
        data['count'] = data.groupby([keyword_column])['count'].transform('sum')

        # remove duplicates
        data.drop_duplicates([keyword_column, 'importance', 'count'], inplace=True)
        
    elif plot_type=='bigram':
        # convert phrases into bigrams 
        data[keyword_column] = data[keyword_column].apply(
            lambda x: list(ngrams(x.split(), 2)))
        data[keyword_column] = data[keyword_column].apply(lambda x: [' '.join([j for j in i]) for i in x])
        data = data[data[keyword_column].astype(bool)]

        # get weighted importance as per count of words
        data['importance'] = data.apply(lambda x: x['importance']/len(x[keyword_column]), axis=1)

        # expand gnn_searchterm_recom to bigrams
        data = data.explode(keyword_column).reset_index(drop=True)
        
        # calculate importance of bigrams across product by taking mean of all the weights corresponding to a phrase
        data['importance'] = data.groupby([keyword_column])['importance'].transform('mean')

        # get total count of each bigram
        data['count'] = data.groupby([keyword_column])['count'].transform('sum')
        
        # remove duplicates
        data.drop_duplicates([keyword_column, 'importance', 'count'], inplace=True)
        
    elif plot_type=='trigram':
        # convert phrases into trigrams 
        data[keyword_column] = data[keyword_column].apply(
            lambda x: list(ngrams(x.split(), 3)))
        data[keyword_column] = data[keyword_column].apply(lambda x: [' '.join([j for j in i]) for i in x])
        data = data[data[keyword_column].astype(bool)]

        # get weighted importance as per count of words
        data['importance'] = data.apply(lambda x: x['importance']/len(x[keyword_column]), axis=1)

        # expand gnn_searchterm_recom to trigrams
        data = data.explode(keyword_column).reset_index(drop=True)
        
        # calculate importance of trigrams across product by taking mean of all the weights corresponding to a phrase
        data['importance'] = data.groupby([keyword_column])['importance'].transform('mean')

        # get total count of each trigram
        data['count'] = data.groupby([keyword_column])['count'].transform('sum')
        
        # remove duplicates
        data.drop_duplicates([keyword_column, 'importance', 'count'], inplace=True)
        
    # sort data
    data.sort_values(sort_data, ascending=False, inplace=True)

    # plot
    plot_bubble_chart(data.iloc[:number_of_keywords_to_plot].copy(), x_column=x_axis, y_column=y_axis, 
                      bubble_size=bubble_size, color=keyword_column, 
                      save_file=save_file, max_size=60, title=title_text)
