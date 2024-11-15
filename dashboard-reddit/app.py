import faicons as fa
import plotly.express as px

# Load data and compute static values
from shared import app_dir, df, post, free_churro, final_topic_overview, load_model, lime_predict, get_lime_highlighted_html, get_ig_highlighted_html, get_lime_word_contributions
from shinywidgets import render_plotly

from shiny import reactive, render
from shiny.express import input, ui

import pandas as pd
from wordcloud import WordCloud
from collections import Counter
from keybert import KeyBERT
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import os
import numpy as np
from transformers import BertForSequenceClassification, BertTokenizer
import torch
from lime.lime_text import LimeTextExplainer
import plotly.graph_objects as go
from IPython.display import display, HTML
from captum.attr import IntegratedGradients
from transformers import AutoTokenizer, AutoModel
import pickle

earliest_date = df['timestamp'].min()
latest_date = df['timestamp'].max()

topic_to_words = df.drop_duplicates(subset=['topic_number']).set_index('topic_number')['topic_words'].to_dict()
topic_to_words = df[df['topic_number'] != -1].drop_duplicates(subset=['topic_number']).set_index('topic_number')['topic_words'].to_dict()
# topic_choices = ["All"] + [f"{topic}: {topic_to_words[topic]}" for topic in sorted(topic_to_words.keys(), key=int)]
topic_choices = [f"{topic}: {topic_to_words[topic]}" for topic in sorted(topic_to_words.keys())]
topic_choices_2 = ["None"] + [f"{topic}: {topic_to_words[topic]}" for topic in sorted(topic_to_words.keys(), key=int)]

subreddit = df['subreddit'].unique().tolist()
subreddit = [x for x in subreddit if str(x) != 'nan']
subreddit_choices = ["All"] + subreddit

# model, tokenizer = load_model()
best_model = load_model()
explainer = LimeTextExplainer(class_names=["Non-trigger", "Trigger"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open(app_dir/"vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
transformer_model = AutoModel.from_pretrained(model_name).to(device)
kw_model = KeyBERT(model=transformer_model)
def generate_keyword_wordcloud(docs, ngram_range=(2, 2), top_n=15, diversity=0.8):
    # Use the preloaded KeyBERT model with GPU support
    global kw_model

    # Extract keywords (processed on GPU)
    keywords = kw_model.extract_keywords(
        docs=list(docs), 
        keyphrase_ngram_range=ngram_range, 
        top_n=top_n, 
        diversity=diversity
    )
    
    # Flatten all keyword phrases across lists
    all_phrases = [phrase for sublist in keywords for phrase, _ in sublist]

    # Count the frequency of each phrase
    phrase_counts = Counter(all_phrases)

    # Get the top 20 most frequent phrases
    top_20_phrases = phrase_counts.most_common(20)
    top_20_phrase_counts = dict(top_20_phrases)

    # Create and display word cloud based on top 20 phrases
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(top_20_phrase_counts)

    # Save the generated word cloud image as a base64 string
    img_buffer = BytesIO()
    wordcloud.to_image().save(img_buffer, format="PNG")
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
    
    return img_base64


ui.page_opts(title="Reddit Comment Analysis Dashboard", fillable=False)

# Icons for the main content
ICONS = {
    "house": fa.icon_svg("house"),
    "eye": fa.icon_svg("eye"),
    "calendar": fa.icon_svg("calendar"),
    "hashtag": fa.icon_svg("hashtag"),
    "filter": fa.icon_svg("filter"),
    "arrow-trend-up": fa.icon_svg("arrow-trend-up"),
    'percent': fa.icon_svg('percent'),
}

# Main content with value boxes and plots
with ui.nav_panel('Time Series Analysis'):
    with ui.layout_sidebar():
        with ui.sidebar(open="desktop"):
            ui.input_date_range(
                "date_range", 
                "Select Date Range:", 
                start=str(earliest_date.date()),  # Convert to string for display
                end=str(latest_date.date())       # Convert to string for display
            )
            ui.input_action_button("reset_date", "Reset Date Range")
            # Create the topic selection input with "All Topics" as the first option
            ui.input_selectize(
                "topicSelect", 
                "Choose Topic(s):", 
                choices=topic_choices, 
                multiple=True,  # Allow multiple selections
                options={"placeholder": "Select one or more topics..."},
                remove_button=True  # Show the remove button for each selected item
            )
            # ui.input_action_button("add_all", "Add All Topics")
            ui.input_action_button("reset", "Reset Selection")

        with ui.layout_columns(fill=False):
            with ui.value_box(showcase=ICONS["eye"]):
                "Filtered Comments"

                @render.express
                def total_comments():
                    filtered_time_series_data().shape[0]

            with ui.value_box(showcase=ICONS["arrow-trend-up"]):
                "Trend Analysis"
                
                @render.text
                def trend_display():
                    # Get the filtered data
                    data = filtered_time_series_data()

                    # Group data by week and calculate counts
                    weekly_counts = data.groupby(pd.Grouper(key='timestamp', freq='W')).size().reset_index(name='Count')
                    
                    # Check if there is enough data to calculate a trend
                    if weekly_counts.empty or len(weekly_counts) < 2:
                        return "No data available for trend analysis."
                    
                    # Convert timestamps to numeric values for regression
                    weekly_counts['timestamp_numeric'] = (weekly_counts['timestamp'] - weekly_counts['timestamp'].min()).dt.days
                    x_values = weekly_counts['timestamp_numeric']
                    y_values = weekly_counts['Count']
                    
                    # Perform linear regression to calculate the slope
                    slope, _ = np.polyfit(x_values, y_values, 1)
                    
                    if slope > 0:
                        trend = "Increasing"
                    elif slope < 0:
                        trend = "Decreasing"
                    else:
                        trend = "Stable"
                    
                    # Return the trend based on the slope
                    return trend


            with ui.value_box(showcase=ICONS["percent"]):
                "Percentage Change"

                @render.text
                def regression_percentage_change_display():
                    # Get the filtered data
                    data = filtered_time_series_data()

                    # Check if data is available
                    if data.empty:
                        return "Choose a topic"

                    # Group data by week and calculate counts
                    weekly_counts = data.groupby(pd.Grouper(key='timestamp', freq='W')).size().reset_index(name='Count')
                    
                    # Ensure there is enough data to calculate the trend
                    if weekly_counts.empty or len(weekly_counts) < 2:
                        return "Not enough data to calculate percentage change of trend."

                    # Convert timestamps to numeric values for regression
                    weekly_counts['timestamp_numeric'] = (weekly_counts['timestamp'] - weekly_counts['timestamp'].min()).dt.days
                    x_values = weekly_counts['timestamp_numeric']
                    y_values = weekly_counts['Count']
                    
                    # Perform linear regression to get the slope and intercept
                    slope, intercept = np.polyfit(x_values, y_values, 1)
                    
                    # Calculate the y-values at the start and end of the period
                    start_x = x_values.iloc[0]
                    end_x = x_values.iloc[-1]
                    
                    start_y = slope * start_x + intercept
                    end_y = slope * end_x + intercept
                    
                    # Calculate the percentage change of the y-value of the regression line
                    if start_y == 0:
                        return "Percentage change cannot be calculated due to zero starting y-value on the regression line."
                    
                    percentage_change = ((end_y - start_y) / start_y) * 100

                    # Return the percentage change formatted to two decimal places
                    return f"{percentage_change:.2f}%"

                
        with ui.layout_columns(height="500px"):

            with ui.card(full_screen=True, min_height="400px"):
                with ui.card_header(class_="fixed-size-card"):
                    ui.card_header("Time Series")
                    @render_plotly
                    def topic_plot():
                        # Get the filtered data
                        data = filtered_time_series_data()

                        # Group data by week and by topic number, counting occurrences for each
                        topic_data = data.groupby([pd.Grouper(key='timestamp', freq='W'), 'topic_number', 'topic_words']).size().reset_index(name='Count')

                        # Check if the data is empty
                        if topic_data.empty:
                            # Create an empty plot with a placeholder title
                            fig = px.line(title="No data available for the selected topics")
                            fig.update_layout(height=400)
                            return fig

                        # Aggregate data across all topics by week
                        overall_weekly_counts = topic_data.groupby('timestamp')['Count'].sum().reset_index()

                        # Create the Plotly figure for individual topic counts, including topic names in the legend
                        fig = px.line(topic_data, x='timestamp', y='Count', color='topic_number', title="Weekly Count with Combined Linear Trend Line",
                                    labels={'color': 'Topic'})  # Customize legend label to show as 'Topic'

                        # Update trace names to include topic names in the legend
                        for trace in fig.data:
                            topic_number = trace.name  # This is the topic_number used in the color legend
                            topic_name = topic_data[topic_data['topic_number'] == int(topic_number)]['topic_words'].iloc[0]
                            trace.name = f"Topic {topic_number} - {topic_name}"

                        # Calculate the combined trend line based on overall weekly counts
                        x_values = np.arange(len(overall_weekly_counts))  # X-axis as indices for linear regression
                        y_values = overall_weekly_counts['Count']
                        slope, intercept = np.polyfit(x_values, y_values, 1)
                        combined_trend_line = slope * x_values + intercept

                        # Add the combined trend line to the plot
                        fig.add_scatter(x=overall_weekly_counts['timestamp'], y=combined_trend_line, mode='lines', name="Combined Trend Line")

                        # Customize layout
                        fig.update_layout(height=400)
                        return fig

                
                # @render_plotly
                # def topic_plot():
                #     # Group data by week and count occurrences
                #     topic_data = filtered_by_topic_data().groupby(pd.Grouper(key='timestamp', freq='W')).size().reset_index(name='Count')

                #     # Check if the data is empty
                #     if topic_data.empty:
                #         # Create an empty plot with a placeholder title
                #         fig = px.line(title="No data available for the selected topics")
                #         fig.update_layout(height=400)
                #         return fig

                #     # Perform linear regression to calculate the trend line
                #     x_values = np.arange(len(topic_data))  # X-axis as indices for linear regression
                #     y_values = topic_data['Count']
                #     slope, intercept = np.polyfit(x_values, y_values, 1)
                #     trend_line = slope * x_values + intercept

                #     # Create the Plotly figure
                #     fig = px.line(topic_data, x='timestamp', y='Count', title="Weekly Count with Linear Trend Line")

                #     # Add the linear trend line to the plot
                #     fig.add_scatter(x=topic_data['timestamp'], y=trend_line, mode='lines', name="Trend Line (Linear Regression)")

                #     # Customize layout
                #     fig.update_layout(height=400)
                #     return fig
                                
            # with ui.card(full_screen=True):
            #     ui.card_header("Filtered Data Table")

            #     @render.data_frame
            #     def table():
            #         return render.DataGrid(filtered_time_series_data())

            # with ui.card(full_screen=True):
            #     with ui.card_header(class_="d-flex justify-content-between align-items-center"):
            #         "Username Frequency"
            #         ICONS["filter"]

            #     @render_plotly
            #     def username_frequency():
            #         # Get the username data from the filtered dataset, excluding "[deleted]"
            #         usernames = filtered_by_topic_data()['username']
            #         usernames = usernames[usernames != "[deleted]"]  # Exclude "[deleted]"
                    
            #         # Calculate the most frequent usernames
            #         username_freq = usernames.value_counts().nlargest(20)
                    
            #         # Create a bar plot for the top 20 most frequent usernames
            #         fig = px.bar(username_freq, x=username_freq.index, y=username_freq.values, 
            #                     labels={'x': 'Username', 'y': 'Frequency'}, 
            #                     title="Top 20 Most Frequent Usernames (Excluding [deleted])")
            #         fig.update_layout(xaxis={'categoryorder': 'total descending'})
            #         return fig

        



with ui.nav_panel('Trend Drivers'):
    with ui.layout_sidebar():
        with ui.sidebar(open="desktop"):
            ui.input_date_range(
                "date_range_2", 
                "Select Date Range:", 
                start=str(earliest_date.date()),  # Convert to string for display
                end=str(latest_date.date())       # Convert to string for display
            )
            # Create the topic selection input with "All Topics" as the first option
            ui.input_selectize(
                "topicSelect_2", 
                "Choose Topic(s):", 
                choices=topic_choices_2, 
                selected=None,  # Ensures no default selection
                multiple=False,  # Allow multiple selections if desired
                options={"placeholder": "Select one or more topics..."}
            )
            # ui.input_action_button("add_all", "Add All Topics")
            ui.input_action_button("reset_2", "Reset Selection")
        
        with ui.layout_columns(fill=False):
            # with ui.card():
            #     ui.card_header("Monster")

            #     @render.ui
            #     def keyword_analysis():
            #         return ui.HTML("""I lost my best friend, I lost my mentor, my mom <br>
            #                         Five hundred men gone, this can't go on <br>
            #                         I must get to see Penelope and Telemachus <br>
            #                         So if we must sail through dangerous oceans and beaches <br>
            #                         I'll go where Poseidon won't reach us <br>
            #                         And if I gotta drop another infant from a wall <br>
            #                         In an instant so we all don't die""")
                
            # with ui.card():
            #     ui.card_header("Free Churro")

            #     @render.text
            #     def free_churro_text():
            #         return free_churro

            with ui.card(height='800px'):
                ui.card_header("Keyword Wordcloud (This Little Maneuver's Gonna Cost Us 51 Years)")

                @render.ui
                def keyword_wordcloud():
                    # Check if data is available from filtered_time_series_data()
                    data = unfiltered_data()
                    selected_topic = input.topicSelect_2()  # Get the topic as a single string like "1: topic1"
                    if selected_topic is None or selected_topic == "None":
                        return ui.HTML("<p>No data or topic selected for word cloud generation.</p>")
                    else:
                        topic_number = int(selected_topic.split(":")[0].strip()) 
                        additional_words = {"singapore", "singaporean", "http", "gon", "na", "gt"}
                        final_topics_overview = final_topic_overview
                        start_date = input.date_range_2()[0]
                        end_date = input.date_range_2()[1]
                        start_date = start_date.strftime('%Y-%m-%d')
                        end_date = end_date.strftime('%Y-%m-%d')
                        
                        def get_topic_words_set(topics_overview, topic_number, additional_words=None):
                            pos = list(topics_overview[topics_overview['Topic'] == topic_number]['POS'].reset_index(drop=True).iloc[0])
                            kbmmr = list(topics_overview[topics_overview['Topic'] == topic_number]['KeyBERT_MMR'].reset_index(drop=True).iloc[0])
                            kb = list(topics_overview[topics_overview['Topic'] == topic_number]['KeyBERT'].reset_index(drop=True).iloc[0])
                            main = list(topics_overview[topics_overview['Topic'] == topic_number]['Representation'].reset_index(drop=True).iloc[0])
                            topic_words_set = set(pos + kbmmr + kb + main)
                            if additional_words:
                                topic_words_set.update(additional_words)
                            return topic_words_set

                        def remove_words(text, words_to_remove):
                            return ' '.join([word for word in text.split() if word.lower() not in words_to_remove])

                        def prepare_document(topic_number, start_date, end_date, topics_overview, additional_words):
                            topic_comments = df[df["new_topic_number"] == topic_number].reset_index(drop=True)
                            filtered_comments = topic_comments[(topic_comments['timestamp'] >= start_date) & (topic_comments['timestamp'] < end_date)]
                            
                            topic_words_set = get_topic_words_set(topics_overview, topic_number, additional_words)
                            filtered_comments['text_cleaned'] = filtered_comments['nltk_processed_text'].apply(lambda x: remove_words(x, topic_words_set))
                            filtered_comments = filtered_comments[filtered_comments['text_cleaned'].notna()]
                            
                            return filtered_comments['text_cleaned']

                        # Assuming `topic_number` is already defined as a single integer
                        all_docs = []

                        # Prepare the document for the single topic number
                        docs = prepare_document(
                            topic_number=topic_number,
                            start_date=start_date,
                            end_date=end_date,
                            topics_overview=final_topics_overview,
                            additional_words=additional_words
                        )
                        all_docs.extend(docs)

                        # Generate and display the word cloud
                        img_base64 = generate_keyword_wordcloud(all_docs)
                        return ui.HTML(f'<img src="data:image/png;base64,{img_base64}" alt="Word Cloud" style="width: 100%;">')



with ui.nav_panel("Subreddit Post Analysis"):
    with ui.layout_sidebar():
        with ui.sidebar(open="desktop"):
            # Create the topic selection input with "All Topics" as the first option
            ui.input_select(
                "subredditSelect", 
                "Choose Subreddit:", 
                choices=subreddit_choices
            )
            ui.input_date_range(
                "date_range_3", 
                "Select Date Range:", 
                start=str(earliest_date.date()),  # Convert to string for display
                end=str(latest_date.date())       # Convert to string for display
            )
            ui.input_text("search_keywords", "Enter keywords (comma-separated):", placeholder="e.g., elections, polling, votes")
            @reactive.Calc
            def get_keywords_to_search():
                # Get the text input from the search bar
                keywords_text = input.search_keywords().strip()
                # Split the input into a list of keywords based on commas
                keywords_to_search = [keyword.strip() for keyword in keywords_text.split(",") if keyword.strip()]
                return keywords_to_search
        
        

        with ui.layout_column_wrap(height="300px"):
            with ui.card():
                ui.card_header("Wordcloud Of Posts With The Most Harmful Content")

                # @render.ui
                # def wordcloud_img():
                #     # Generate the word cloud from RAKE keywords
                #     data = filtered_by_post_data()  # Assumes this function returns a DataFrame
                #     all_text = ' '.join([' '.join(eval(row)) if isinstance(row, str) else '' for row in data['rake_keywords']])

                #     # Generate the word cloud
                #     wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_text)

                #     # Save the word cloud to an in-memory file
                #     img_io = BytesIO()
                #     wordcloud.to_image().save(img_io, format="PNG")
                #     img_io.seek(0)

                #     # Encode the image in base64
                #     base64_img = base64.b64encode(img_io.getvalue()).decode('utf-8')
                #     img_src = f"data:image/png;base64,{base64_img}"

                #     # Return the image as an HTML <img> tag
                #     return ui.HTML(f'<img src="{img_src}" alt="Word Cloud" style="width: 100%;">')

                @render.ui
                def wordcloud_img():
                    # Assume selected_subreddit() is a function that returns the selected subreddit
                    selected_subreddit = input.subredditSelect()  # Replace with the actual input name

                    # Convert subreddit name to match the filename format, e.g., "r_Singapore.png"
                    filename = f"{selected_subreddit.replace('/', '_')}.png"
                    # filepath = os.path.join(png_directory, filename)
                    filepath = f'{app_dir}\\wordclouds\\{filename}'

                    # Check if the file exists
                    if os.path.exists(filepath):
                        # Load the image file and convert to base64
                        with open(filepath, "rb") as img_file:
                            base64_img = base64.b64encode(img_file.read()).decode("utf-8")
                        img_src = f"data:image/png;base64,{base64_img}"
                        
                        # Display the image
                        return ui.HTML(f'<img src="{img_src}" alt="Word Cloud" style="width: 100%;">')
                    else:
                        # Display a placeholder message if the file is not found
                        return ui.HTML("<p>No word cloud available for the selected subreddit.</p>")
                    
            with ui.card(full_screen=True, min_height="300px"):
                ui.card_header("Top 20 Posts With Comments Given Time")

                @render.ui
                def top_20_posts():
                    # Get the selected subreddit from the dropdown
                    selected_subreddit = input.subredditSelect()  # Assuming you have a dropdown for subreddit selection
                    data = filtered_by_post_data()  # Assuming this function returns a DataFrame

                    data = data[~data['post_title'].str.startswith('/r/singapore random discussion and small questions thread for', na=False)]
                    
                    # Filter data based on the selected subreddit
                    if selected_subreddit == "All":
                        data = data  # Use the entire dataset when "All" is selected
                    else:
                        data = data[data['subreddit'] == selected_subreddit]

                    # Check if there's data for the selected subreddit or all data
                    if data.empty:
                        return ui.HTML("<p>No posts available for the selected subreddit.</p>")
                    
                    # Get top 20 posts by comment count, removing duplicate rows
                    top_posts = data[['post_title', 'comment_count']].drop_duplicates().nlargest(20, 'comment_count')

                    # Convert the top posts to an HTML list
                    post_list_html = "<ol>"
                    for _, row in top_posts.iterrows():
                        post_list_html += f"<li><strong>{row['post_title']}</strong> - {row['comment_count']} comments</li>"
                    post_list_html += "</ol>"

                    # Display the top 20 posts in the card
                    return ui.HTML(post_list_html)
            
            with ui.card():
                ui.card_header("Most Upvoted Posts Over Time")

                @render.ui
                def top_upvoted_posts():
                    # Filter the data to get unique posts by title and get the top 20 by vote_score
                    data = filtered_by_post_data()
                    unique_posts = data[['post_title', 'vote_score', 'timestamp']].drop_duplicates(subset='post_title')
                    top_posts = unique_posts.nlargest(20, 'vote_score')

                    # Check if there's data available
                    if top_posts.empty:
                        return ui.HTML("<p>No posts available.</p>")

                    # Convert the top posts to an HTML list with title, vote score, and timestamp
                    post_list_html = "<ol>"
                    for _, row in top_posts.iterrows():
                        post_list_html += f"<li><strong>{row['post_title']}</strong> - {row['vote_score']} upvotes on {row['timestamp'].strftime('%Y-%m-%d')}</li>"
                    post_list_html += "</ol>"

                    # Display the top 20 upvoted posts in the card
                    return ui.HTML(post_list_html)


            
        
        with ui.layout_column_wrap(height="400px"):
            # with ui.card():
            #     ui.card_header("Post Title hate analysis")

            #     @render.ui
            #     def general_proportion_of_hate_comments_img():
            #         filepath = f'{app_dir}\\wordclouds\\general_proportion_of_hate_comments.png'

            #     # Check if the file exists
            #         if os.path.exists(filepath):
            #             # Load the image file and convert to base64
            #             with open(filepath, "rb") as img_file:
            #                 base64_img = base64.b64encode(img_file.read()).decode("utf-8")
            #             img_src = f"data:image/png;base64,{base64_img}"
                        
            #             # Display the image
            #             return ui.HTML(f'<img src="{img_src}" alt="Word Cloud" style="width: 100%;">')
            #         else:
            #             # Display a placeholder message if the file is not found
            #             return ui.HTML("<p>No word cloud available for the selected subreddit.</p>")
            with ui.card():
                ui.card_header("Keyword Distribution Over Time")

                @render_plotly
                def display_keyword_trend():
                    # Get the keywords from the user input and process them
                    keywords_text = input.search_keywords()  # Assume `search_keywords` input holds comma-separated keywords
                    keywords_to_search = [kw.strip() for kw in keywords_text.split(",") if kw.strip()]  # Clean up and split keywords
                    
                    # If no keywords are provided, return an empty figure
                    if not keywords_to_search:
                        return go.Figure().update_layout(title="No keywords entered.")

                    # Fetch the filtered data
                    data = filtered_by_post_data()

                    # Check if data is empty after filtering
                    if data.empty:
                        return go.Figure().update_layout(title="No data available for the selected keywords.")

                    # Convert 'timestamp' to 'month' for aggregation
                    data['month'] = data['timestamp'].dt.to_period('M')

                    # Initialize a DataFrame to hold keyword counts
                    keyword_counts = pd.DataFrame()

                    # Count occurrences of each keyword and store in `keyword_counts`
                    for keyword in keywords_to_search:
                        keyword_counts[keyword] = data['rake_keywords'].str.count(keyword)

                    # Sum keyword counts by month
                    monthly_keyword_counts = keyword_counts.groupby(data['month']).sum().reset_index()

                    # Calculate total counts of all keywords each month
                    monthly_keyword_counts['total_counts'] = monthly_keyword_counts[keywords_to_search].sum(axis=1)
                    plot_data = monthly_keyword_counts[['month', 'total_counts']]
                    plot_data['month'] = plot_data['month'].astype(str)

                    # Create the Plotly line plot
                    fig = px.line(
                        plot_data, x='month', y='total_counts', title='Total Keyword Occurrences Over Time',
                        markers=True, line_shape='linear'
                    )

                    # Customize the layout
                    fig.update_layout(
                        xaxis_title='Month',
                        yaxis_title='Total Count of Keywords',
                        xaxis_tickangle=-45,
                        template='plotly_white'
                    )

                    return fig
                

with ui.nav_panel("User Analysis By Subreddit"):
    with ui.layout_sidebar():
        with ui.sidebar(open="desktop"):
            # Create the subreddit selection input with "All Topics" as the first option
            ui.input_select(
                "subredditSelect_2", 
                "Choose Subreddit:", 
                choices=subreddit_choices
            )
            
            # Add an input field for entering a username
            ui.input_text(
                "usernameInput",
                "Enter Username:"
            )
            
            # Add a slider for selecting toxicity level between 0 and 1
            ui.input_slider(
                "toxicityLevel",
                "Choose Top Harm %:",
                min=0,
                max=100,
                value=10,  # Default starting value
                step=1   # Slider increments by 0.01 for finer control
            )
        with ui.layout_columns(height="400px"):
            with ui.card():
                ui.card_header("Top 20 Harmful Users")

                # @render.ui
                # def subreddit_frequency():
                #     # Get the subreddit data from the filtered dataset
                #     subreddits = filtered_by_post_data_2()['username']
                    
                #     # Exclude entries with "[deleted]", "sneakpeek_bot", and "AutoModerator"
                #     subreddits = subreddits[~subreddits.isin(["[deleted]", "sneakpeek_bot", "AutoModerator"])]
                    
                #     # Calculate the most frequent subreddits
                #     subreddit_freq = subreddits.value_counts().nlargest(20)
                    
                #     # Convert to a list of strings in the format "Rank. Username: Frequency"
                #     top_subreddits_list = [f"{rank}. {subreddit}: {count}" 
                #                         for rank, (subreddit, count) in enumerate(subreddit_freq.items(), start=1)]
                    
                #     # Render as a UI component, separated by line breaks
                #     return ui.HTML("<br>".join(top_subreddits_list))
                @render.ui
                def top_usernames():
                    # Sort the DataFrame by hate_comment_count in descending order and get the top 20
                    data = filtered_by_post_data_2()
                    top_users = data.sort_values(by='hate_comment_count', ascending=False).head(20)

                    # Convert the top users to an HTML list
                    user_list_html = "<ol>"
                    for _, row in top_users.iterrows():
                        user_list_html += f"<li><strong>{row['username']}</strong> - {row['hate_comment_count']} comments</li>"
                    user_list_html += "</ol>"

                    # Display the top 20 usernames in the card
                    return ui.HTML(user_list_html)
            
            # with ui.card():
            #     ui.card_header("Legolas, what do your elf eyes see?")

            #     @render.text
            #     def legolas():
            #         return "The Uruks turn north-east. They're taking the hobbits to Isengard!"
            
            with ui.card():
                ui.card_header("Cumulative Plot Of Harmfulness")

                @render_plotly
                def cumulative_hate_plot():
                    # Use the filtered data
                    data = filtered_by_post_data_2()  # Assuming this function applies the subreddit and username filter

                    # Check if the filtered data is empty
                    if data.empty:
                        fig = go.Figure()
                        fig.add_annotation(
                            x=0.5, y=0.5,
                            text="No data available for the selected criteria.",
                            showarrow=False,
                            font=dict(size=16)
                        )
                        fig.update_layout(
                            xaxis=dict(visible=False),
                            yaxis=dict(visible=False)
                        )
                        return fig

                    # Sort data by hate_percentage in descending order
                    data = data.sort_values(by='hate_percentage', ascending=False).reset_index(drop=True)

                    # Limit to top 10% or top 20 users for more focused plotting
                    top_limit = int(0.1 * len(data))  # Adjust percentage as desired
                    data = data.head(top_limit) if top_limit > 20 else data.head(20)

                    # Calculate cumulative hate percentage and cumulative username count
                    data['cumulative_hate_percentage'] = data['hate_percentage'].cumsum()

                    # Normalize cumulative hate percentage to ensure it reaches exactly 100%
                    data['cumulative_hate_percentage'] = (data['cumulative_hate_percentage'] / data['cumulative_hate_percentage'].iloc[-1]) * 100
                    data['cumulative_user_count'] = range(1, len(data) + 1)

                    # Create the cumulative plot with Plotly
                    fig = go.Figure()

                    fig.add_trace(
                        go.Scatter(
                            x=data['cumulative_user_count'],
                            y=data['cumulative_hate_percentage'],
                            mode='lines+markers',
                            line=dict(width=2),
                            marker=dict(size=6),
                            name="Cumulative Harmful Percentage"
                        )
                    )

                    # Update layout for readability
                    fig.update_layout(
                        title="Cumulative Harmful Percentage by Username Count",
                        xaxis_title="Cumulative Username Count",
                        yaxis_title="Cumulative Harm Percentage (%)",
                        yaxis=dict(range=[0, 100]),  # Set y-axis range to 100% for clarity
                        template="plotly_white",
                        font=dict(size=12)
                    )

                    return fig

                @render.ui
                def display_top_percent_hate_usernames():
                    # Retrieve inputs
                    threshold = input.toxicityLevel()  # Use toxicity level as the threshold for cumulative hate percentage
                    selected_subreddit = input.subredditSelect_2()  # Selected subreddit
                    
                    # Retrieve the filtered data for hate comments using filtered_by_post_data_2 function
                    subreddit_data = filtered_by_post_data_2()

                    # Filter for the selected subreddit if "All" is not selected
                    if selected_subreddit != "All":
                        subreddit_data = subreddit_data[subreddit_data['subreddit'] == selected_subreddit]

                    # Sort the data by subreddit and hate_percentage in descending order
                    df_sorted = subreddit_data.sort_values(by=['subreddit', 'hate_percentage'], ascending=[True, False])

                    # Define a function to get the count of usernames contributing to top 10% hate_percentage
                    def count_top_users_by_hate_threshold(group, threshold):
                        # Calculate cumulative sum of hate_percentage
                        group = group.copy()  # Avoid modifying the original DataFrame
                        group['cumulative_hate_percentage'] = group['hate_percentage'].cumsum()

                        # Calculate the target cumulative hate percentage
                        target_percentage = group['hate_percentage'].sum() * (threshold / 100)

                        # Count usernames until the cumulative hate percentage reaches or exceeds the target
                        count = group[group['cumulative_hate_percentage'] <= target_percentage].shape[0]

                        # Add one more user if necessary to reach or slightly exceed the threshold
                        if count < group.shape[0] and group['cumulative_hate_percentage'].iloc[count] < target_percentage:
                            count += 1

                        return count

                    # If "All" is selected, apply the function on the entire dataset
                    if selected_subreddit == "All":
                        total_usernames_count = count_top_users_by_hate_threshold(df_sorted, threshold)
                        result_text = f"<h4>Total Harm Contribution Analysis</h4><p>{total_usernames_count} usernames contribute to the top {threshold}% harmful comments across all subreddits.</p>"
                    else:
                        # Apply the function to the selected subreddit group
                        top_usernames_by_hate_threshold = df_sorted.groupby('subreddit').apply(
                            lambda group: count_top_users_by_hate_threshold(group, threshold)
                        ).reset_index(name='username_count')

                        # Show results for the selected subreddit only
                        username_count = top_usernames_by_hate_threshold.loc[
                            top_usernames_by_hate_threshold['subreddit'] == selected_subreddit, 'username_count'
                        ].values[0]
                        result_text = f"<h4>{selected_subreddit} Harm Contribution Analysis</h4><p>{username_count} usernames contribute to the top {threshold}% harmful comments in {selected_subreddit}.</p>"

                    # Display the result as HTML text in the UI
                    return ui.HTML(result_text)


                    
        with ui.layout_column_wrap(height="400px"):
            with ui.card():
                ui.card_header("Time Plot Of Chosen Username Comment Behaviour")

                @render_plotly
                def plot_user_comment_time_series_interactive():
                    # Retrieve the username from the input
                    username = input.usernameInput()
                    selected_subreddit = input.subredditSelect_2()  # Selected subreddit
                    Data = unfiltered_data()  # Assuming this function returns the filtered data

                    if selected_subreddit != "All":
                        # Filter data for the specified username
                        Data = Data[Data['subreddit'] == selected_subreddit]

                    # Check if username is provided
                    if not username:
                        fig = go.Figure()
                        fig.add_annotation(
                            x=0.5, y=0.5,
                            text="Please enter a username.",
                            showarrow=False,
                            font=dict(size=16)
                        )
                        fig.update_layout(
                            xaxis=dict(visible=False),
                            yaxis=dict(visible=False)
                        )
                        return fig

                    # Step 1: Filter data for the specified username
                    user_data = Data[Data['username'] == username].copy()

                    # Ensure there is data for the username
                    if user_data.empty:
                        fig = go.Figure()
                        fig.add_annotation(
                            x=0.5, y=0.5,
                            text=f"No data available for username: {username}",
                            showarrow=False,
                            font=dict(size=16)
                        )
                        fig.update_layout(
                            xaxis=dict(visible=False),
                            yaxis=dict(visible=False)
                        )
                        return fig

                    # Step 2: Ensure 'post_timestamp' is in datetime format if it's not already
                    user_data['post_timestamp'] = pd.to_datetime(user_data['post_timestamp'], errors='coerce')

                    # Step 3: Set 'post_timestamp' as the index to enable resampling
                    user_data.set_index('post_timestamp', inplace=True)

                    # Step 4: Resample data monthly, counting only hate comments
                    monthly_hate_counts = user_data[user_data['BERT_2_hate'] == True].resample('M').size()  # Hate comments per month

                    # Step 5: Create the interactive plot with Plotly
                    fig = go.Figure()

                    # Plot hate comments
                    fig.add_trace(go.Scatter(
                        x=monthly_hate_counts.index,
                        y=monthly_hate_counts.values,
                        mode='lines+markers',
                        name='Harmful Comments',
                        line=dict(color='red'),
                        marker=dict(size=6)
                    ))

                    # Update layout for better readability
                    fig.update_layout(
                        title=f"Monthly Harmful Comment Activity for {username}",
                        xaxis_title="Month",
                        yaxis_title="Number of Harmful Comments",
                        hovermode="x unified",  # Shows all hover data for a specific x-axis value
                        template="plotly_white",
                        showlegend=True
                    )

                    return fig

with ui.nav_panel("Trigger Analysis"):
    with ui.layout_sidebar():
        with ui.sidebar(open="desktop"):
            ui.input_select(
                "card_selection",
                "Select Model To Display:",
                choices=["Lime", "Integrated Gradients"]
            )
            # Text input and submit/reset buttons
            ui.input_text("input_text", "Enter Text:")
            # ui.input_action_button("submit_text", "Submit")  # Submit button
            ui.input_action_button("reset_text", "Reset")    # Reset button

        with ui.layout_columns(col_widths=[4, 8]):
            with ui.card():
                ui.card_header("Prediction Probabilities")

                # Function to predict probabilities for a given input text
                @reactive.Calc
                def prediction_probabilities():
                    entered_text = input.input_text()

                    # Check if there is text to analyze
                    if not entered_text:
                        return None

                    # Use the lime_predict function with best_model and vectorizer
                    probs = lime_predict([entered_text])

                    # Get probabilities for "Non-trigger" and "Trigger" classes
                    non_trigger_prob = probs[0][0]  # Assumes 0th index is "Non-trigger"
                    trigger_prob = probs[0][1]      # Assumes 1st index is "Trigger"
                    
                    return {"Non-trigger": non_trigger_prob, "Trigger": trigger_prob}
                
                @reactive.Calc
                def integrated_gradients_probabilities():
                    entered_text = input.input_text()
                    
                    # Check if there is text to analyze
                    if not entered_text:
                        return None

                    # Vectorize the input text using TF-IDF vectorizer
                    input_tfidf = vectorizer.transform([entered_text]).toarray()
                    input_tensor = torch.tensor(input_tfidf, dtype=torch.float32).to(next(best_model.parameters()).device)
                    
                    # Ensure the model is in evaluation mode
                    best_model.eval()
                    
                    # Compute the prediction probabilities
                    with torch.no_grad():
                        outputs = best_model(input_tensor)
                        probabilities = torch.sigmoid(outputs).squeeze().cpu().numpy()
                    
                    # Return probabilities as a dictionary
                    return {
                        "Non-trigger": 1 - probabilities,  # Assumes non-trigger is the complement of trigger probability
                        "Trigger": probabilities
                    }

                @render_plotly
                def display_prediction_probabilities():
                    xai = input.card_selection()
                    if xai == "Lime":
                        probs = prediction_probabilities()

                        if probs is None:
                            return go.Figure()  # Return an empty figure if no text is entered

                        # Create the Plotly bar plot with updated colors
                        fig = go.Figure(
                            data=[
                                go.Bar(
                                    x=[probs["Non-trigger"], probs["Trigger"]],
                                    y=["Non-trigger", "Trigger"],
                                    orientation='h',
                                    marker_color=["#98FB98", "#FF6347"],  # Updated colors to match Integrated Gradients scheme
                                    text=[f"{probs['Non-trigger']:.2f}", f"{probs['Trigger']:.2f}"],  # Display values
                                    textposition="auto",  # Automatically place text inside or outside the bar
                                    textfont=dict(size=16, color="black", family="Arial, bold"),  # Enlarge and bold the text inside bars
                                    hovertemplate='%{text}<extra></extra>'  # Show text on hover
                                )
                            ]
                        )

                        # Update layout for readability
                        fig.update_layout(
                            title="Prediction Probabilities",
                            xaxis_title="Probability",
                            yaxis=dict(
                                autorange="reversed",  # Display "Non-trigger" on top
                                tickfont=dict(size=18, family="Arial, bold")  # Make y-axis labels larger and bold
                            ),
                            height=200,
                            margin=dict(l=80, r=20, t=30, b=20),  # Increase left margin for larger y-axis labels
                        )

                        return fig
                    else:
                        probs = integrated_gradients_probabilities()

                        if probs is None:
                            return go.Figure()  # Return an empty figure if no text is entered

                        # Create the Plotly bar plot
                        fig = go.Figure(
                            data=[
                                go.Bar(
                                    x=[probs["Non-trigger"], probs["Trigger"]],
                                    y=["Non-trigger", "Trigger"],
                                    orientation='h',
                                    marker_color=["#98FB98", "#FF6347"],  # Green for Non-trigger, Red for Trigger
                                    text=[f"{probs['Non-trigger']:.2f}", f"{probs['Trigger']:.2f}"],  # Display values
                                    textposition="auto",  # Automatically place text inside or outside the bar
                                    textfont=dict(size=16, color="black", family="Arial, bold"),
                                    hovertemplate='%{text}<extra></extra>'  # Show text on hover
                                )
                            ]
                        )

                        # Update layout for readability
                        fig.update_layout(
                            title="Prediction Probabilities via Integrated Gradients",
                            xaxis_title="Probability",
                            yaxis=dict(
                                autorange="reversed",
                                tickfont=dict(size=18, family="Arial, bold")
                            ),
                            height=200,
                            margin=dict(l=80, r=20, t=30, b=20),
                        )

                        return fig

            with ui.card():  ##### NUMBER 2222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222
                ui.card_header("Predicted Topic With Explanation")

                ig = IntegratedGradients(best_model)

                @reactive.Calc
                def integrated_gradients_word_attributions():
                    entered_text = input.input_text()
                    
                    if not entered_text:
                        return None

                    # Vectorize the input text using TF-IDF vectorizer
                    input_tfidf = vectorizer.transform([entered_text]).toarray()
                    input_tensor = torch.tensor(input_tfidf, dtype=torch.float32).to(next(best_model.parameters()).device)
                    
                    # Compute attributions using Integrated Gradients
                    attributions, _ = ig.attribute(input_tensor, target=0, return_convergence_delta=True)
                    attributions_np = attributions.cpu().detach().numpy()[0]

                    # Get the words corresponding to the features from the TF-IDF vectorizer
                    words = vectorizer.get_feature_names_out()

                    # Zip words with their corresponding attribution values
                    word_attributions = list(zip(words, attributions_np))

                    # Filter out words with zero attributions
                    non_zero_word_attributions = [(word, attr) for word, attr in word_attributions if abs(attr) > 0]

                    # Sort by absolute value of attribution and take the top 10
                    sorted_word_attributions = sorted(non_zero_word_attributions, key=lambda x: abs(x[1]), reverse=True)[:10]

                    # Return the top words and their scores as a list of tuples
                    return sorted_word_attributions

                # Render the LIME explanation as a Plotly bar plot
                @render_plotly
                def display_explanation():
                    xai = input.card_selection()
                    if xai == "Lime":
                        entered_text = input.input_text()
    
                        if not entered_text:
                            return go.Figure()  # Return an empty figure if no text is entered

                        # Get the word contributions from LIME
                        word_contributions = get_lime_word_contributions(entered_text)

                        # Separate words and scores
                        words = [item[0] for item in word_contributions]
                        scores = [item[1] for item in word_contributions]

                        # Create a Plotly bar plot for LIME word contributions
                        fig = go.Figure(
                            data=[
                                go.Bar(
                                    x=scores,  # Contribution scores as x-axis values
                                    y=words,  # Words as y-axis labels
                                    orientation='h',
                                    marker=dict(
                                        color=["#FF6347" if score > 0 else "#98FB98" for score in scores],  # Red for positive, green for negative
                                        line=dict(color="rgba(255,255,255,0.8)", width=1)  # White border for readability
                                    ),
                                    text=[f"{score:.2f}" for score in scores],  # Display contribution scores
                                    textposition="auto",
                                    textfont=dict(size=16, color="black", family="Arial, bold"),
                                    hovertemplate='%{text}<extra></extra>'
                                )
                            ]
                        )
                        
                        # Update layout for better readability
                        fig.update_layout(
                            title="Word Contributions via LIME",
                            xaxis_title="Contribution Score",
                            yaxis=dict(
                                autorange="reversed",
                                tickfont=dict(size=18, family="Arial, bold")
                            ),
                            height=400,
                            margin=dict(l=80, r=20, t=30, b=20),
                        )
                        
                        return fig
                    
                    else:

                        word_attributions = integrated_gradients_word_attributions()

                        if word_attributions is None:
                            return go.Figure()  # Return an empty figure if no text is entered

                        # Separate words and scores
                        words = [item[0] for item in word_attributions]
                        scores = [item[1] for item in word_attributions]

                        # Set colors with varying intensity based on the attribution score
                        max_abs_score = max(abs(score) for score in scores)
                        colors = [
                            f'rgba(255, 0, 0, {min(1, abs(score) / max_abs_score)})' if score > 0 else 
                            f'rgba(0, 128, 0, {min(1, abs(score) / max_abs_score)})' 
                            for score in scores
                        ]

                        # Create the Plotly bar plot for IG word attributions
                        fig = go.Figure(
                            data=[
                                go.Bar(
                                    x=scores,  # Contribution scores as x-axis values
                                    y=words,  # Words as y-axis labels
                                    orientation='h',
                                    marker=dict(
                                        color=colors,
                                        line=dict(color="rgba(255,255,255,0.8)", width=1)  # White border for readability
                                    ),
                                    text=[f"{score:.2f}" for score in scores],  # Display contribution scores
                                    textposition="auto",
                                    textfont=dict(size=16, color="black", family="Arial, bold"),
                                    hovertemplate='%{text}<extra></extra>'
                                )
                            ]
                        )
                        
                        # Update layout for better readability
                        fig.update_layout(
                            title="Top Word Contributions via Integrated Gradients",
                            xaxis_title="Contribution Score",
                            yaxis=dict(
                                autorange="reversed",
                                tickfont=dict(size=18, family="Arial, bold")
                            ),
                            height=400,
                            margin=dict(l=80, r=20, t=30, b=20),
                        )
                        
                        return fig

                

        with ui.layout_columns():
            with ui.card():
                ui.card_header("Text Attribution Explanation")

                @render.ui
                def display_lime_highlighted_text_ui():
                    ig = IntegratedGradients(best_model)
                    xai = input.card_selection()
                    if xai == "Lime":
                        entered_text = input.input_text()
                        
                        if not entered_text:
                            return HTML("<p>Please enter text for analysis.</p>")

                        # Generate highlighted HTML based on LIME attributions
                        highlighted_html = get_lime_highlighted_html(entered_text)

                        # Wrap the HTML in HTML() to ensure it renders as HTML
                        return HTML(f"<div>{highlighted_html}</div>")
                    else:
                        entered_text = input.input_text()
    
                        if not entered_text:
                            return HTML("<p>Please enter text for analysis.</p>")

                        # Generate highlighted HTML based on IG attributions
                        highlighted_html = get_ig_highlighted_html(entered_text)

                        # Wrap the HTML in HTML() to ensure it renders correctly in the UI
                        return HTML(f"<div>{highlighted_html}</div>")




# # --------------------------------------------------------
# # Reactive calculations and effects
# # --------------------------------------------------------

@reactive.calc
def filtered_time_series_data():
    # Retrieve the selected topics (could be a list if multiple are selected)
    selected_topics = input.topicSelect()
    selected_date_range = input.date_range()  # Retrieve selected date range
    
    # Start with the full dataset
    data = df.copy()
    
    # Check if "All" is in the selection, return the entire dataset
    if "All" not in selected_topics:
        # Convert selected topics to integers if they contain topic numbers
        topic_numbers = [
            int(topic.split(":")[0]) if ":" in topic else int(topic)
            for topic in selected_topics
        ]
        # Filter the dataset for the selected topics
        data = data[data['topic_number'].isin(topic_numbers)]
    
    # Filter the data for the selected date range
    if selected_date_range:
        start_date, end_date = pd.to_datetime(selected_date_range[0]), pd.to_datetime(selected_date_range[1])
        data = data[(data['timestamp'] >= start_date) & (data['timestamp'] <= end_date)]

    return data

def filtered_topic_data():
    # Retrieve the selected topics (could be a list if multiple are selected)
    selected_topics = input.topicSelect_2()
    
    # Start with the full dataset
    data = df.copy()
    
    # Check if "All" is in the selection, return the entire dataset
    if "All" in selected_topics:
        return data
    
    # Convert selected topics to integers if they contain topic numbers
    topic_numbers = [
        int(topic.split(":")[0]) if ":" in topic else int(topic)
        for topic in selected_topics
    ]
    
    # Filter the dataset for the selected topics
    data = data[data['topic_number'].isin(topic_numbers)]
    
    return data

def filtered_by_post_data():
    try:
        # Retrieve the selected subreddit
        selected_subreddit = input.subredditSelect()
        print(f"Selected Subreddit: {selected_subreddit}")  # Debugging output

        # Retrieve selected date range
        selected_date_range = input.date_range_3()  # New: Retrieve the date range

        # Start with a copy of the DataFrame
        data = df.copy()
        print(f"Initial data shape: {data.shape}")  # Debugging output

        # Filter by subreddit if not "All"
        if selected_subreddit != "All":
            data = data[data['subreddit'] == selected_subreddit]
            print(f"Filtered data shape by subreddit: {data.shape}")  # Debugging output

        # Filter by date range if provided
        if selected_date_range:
            start_date, end_date = pd.to_datetime(selected_date_range[0]), pd.to_datetime(selected_date_range[1])
            data = data[(data['timestamp'] >= start_date) & (data['timestamp'] <= end_date)]
            print(f"Filtered data shape by date range: {data.shape}")  # Debugging output

        # Check if data is empty and handle it gracefully
        if data.empty:
            print("Warning: Filtered data is empty.")

        return data

    except Exception as e:
        print(f"Error in filtered_by_post_data: {e}")
        return pd.DataFrame()
    
def filtered_by_post_data_2():
    # Group by subreddit and username, counting the number of hate comments
    Data = post.copy()
    Data = Data.drop(columns=['Unnamed: 0'])
    # Filter for hate comments and remove specific usernames
    hate_comments = Data[(Data['BERT_2_hate'] == True) & 
                         (Data['username'] != '[deleted]') & 
                         (Data['username'] != 'sneakpeek_bot') & 
                         (Data['username'] != 'AutoModerator')]
    
    selected_subreddit = input.subredditSelect_2()

    if selected_subreddit != "All":
        hate_comments = hate_comments[hate_comments['subreddit'] == selected_subreddit]

    # Group by subreddit and username, counting the number of hate comments
    hate_counts = hate_comments.groupby(['subreddit', 'username']).size().reset_index(name='hate_comment_count')

    # Calculate total hate comments per subreddit
    total_hate_per_subreddit = hate_counts.groupby('subreddit')['hate_comment_count'].sum().reset_index(name='total_hate_comments')

    # Merge to get total hate comments alongside username hate comment count
    hate_counts = hate_counts.merge(total_hate_per_subreddit, on='subreddit')

    # Calculate percentage of hate comments
    hate_counts['hate_percentage'] = (hate_counts['hate_comment_count'] / hate_counts['total_hate_comments']) * 100

    # Sort by subreddit and hate percentage in descending order
    top_hate_users = hate_counts.sort_values(['subreddit', 'hate_comment_count'], ascending=[True, False])

    # Return the result
    return top_hate_users   

def unfiltered_data():
    return df

# @reactive.Effect
# @reactive.event(input.add_all)  # Trigger when "Add All Topics" button is clicked
# def add_all_topics():
#     # Set the `topicSelect` input to all available topics
#     all_topics = [topic for topic in topic_choices]
#     ui.update_selectize("topicSelect", selected=all_topics)

@reactive.Effect
@reactive.event(input.reset_date)  # Trigger when "Reset Date Range" button is clicked
def reset_date_range():
    # Reset the `date_range` input to the initial values
    ui.update_date_range(
        "date_range", 
        start=str(earliest_date.date()),  # Reset to initial start date
        end=str(latest_date.date())       # Reset to initial end date
    )

@reactive.Effect
@reactive.event(input.reset)  # Trigger when "Reset Selection" button is clicked
def reset_topics():
    # Clear the `topicSelect` input selection
    ui.update_selectize("topicSelect", selected=[])

@reactive.Effect
@reactive.event(input.reset_2)  # Trigger when "Reset Selection" button is clicked
def reset_topics():
    # Clear the topicSelect input selection
    ui.update_selectize("topicSelect_2", selected=[])

@reactive.Calc
@reactive.event(input.submit_text)  # Trigger only on submit button click
def submitted_text():
    return input.input_text()  # Capture and return the current value of input_text


@reactive.Effect
@reactive.event(input.reset_text)  # Trigger when "Reset" button is clicked
def reset_input():
    # Clear the text input field
    ui.update_text("input_text", value="")