import faicons as fa
import plotly.express as px

# Load data and compute static values
from shared import app_dir, df, free_churro, load_model, lime_predict, generate_lime_html
from shinywidgets import render_plotly

from shiny import reactive, render
from shiny.express import input, ui

import pandas as pd
from wordcloud import WordCloud
from io import BytesIO
import base64
import os
import numpy as np
from transformers import BertForSequenceClassification, BertTokenizer
import torch
from lime.lime_text import LimeTextExplainer
import plotly.graph_objects as go
from IPython.display import display, HTML

topic_to_words = df.drop_duplicates(subset=['topic_number']).set_index('topic_number')['topic_words'].to_dict()
topic_to_words = df[df['topic_number'] != -1].drop_duplicates(subset=['topic_number']).set_index('topic_number')['topic_words'].to_dict()
# topic_choices = ["All"] + [f"{topic}: {topic_to_words[topic]}" for topic in sorted(topic_to_words.keys(), key=int)]
topic_choices = [f"{topic}: {topic_to_words[topic]}" for topic in sorted(topic_to_words.keys())]

subreddit = df['subreddit'].unique().tolist()
subreddit = [x for x in subreddit if str(x) != 'nan']
# subreddit_choices = ["All"] + subreddit
subreddit_choices = subreddit

model, tokenizer = load_model()
explainer = LimeTextExplainer(class_names=["Non-trigger", "Trigger"])

ui.page_opts(title="Reddit Comment Analysis Dashboard", fillable=False)

# Icons for the main content
ICONS = {
    "house": fa.icon_svg("house"),
    "eye": fa.icon_svg("eye"),
    "calendar": fa.icon_svg("calendar"),
    "hashtag": fa.icon_svg("hashtag"),
    "filter": fa.icon_svg("filter"),
    "arrow-trend-up": fa.icon_svg("arrow-trend-up"),
}

# Main content with value boxes and plots
with ui.nav_panel('Time series analysis'):
    with ui.layout_sidebar():
        with ui.sidebar(open="desktop"):
            # Create the topic selection input with "All Topics" as the first option
            ui.input_selectize(
                "topicSelect", 
                "Choose Topic(s):", 
                choices=topic_choices, 
                multiple=True,  # Allow multiple selections
                options={"placeholder": "Select one or more topics..."}
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
                "Trend analysis"
                
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


            with ui.value_box(showcase=ICONS["hashtag"]):
                "Selected Topic(s)"

                @render.text
                def selected_topic():
                    return "AND ITHACA'S WAITING"
                
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
            #         return render.DataGrid(filtered_by_topic_data())

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

        with ui.card(full_screen=True):
            ui.card_header("Filtered Data Table")

            @render.data_frame
            def table():
                return render.DataGrid(filtered_time_series_data())

with ui.nav_panel('Topic analysis'):
    with ui.layout_sidebar():
        with ui.sidebar(open="desktop"):
            # Create the topic selection input with "All Topics" as the first option
            ui.input_selectize(
                "topicSelect_2", 
                "Choose Topic(s):", 
                choices=topic_choices, 
                multiple=True,  # Allow multiple selections
                options={"placeholder": "Select one or more topics..."}
    )
            # ui.input_action_button("add_all", "Add All Topics")
            ui.input_action_button("reset_2", "Reset Selection")
        
        with ui.layout_columns(fill=False):
            with ui.card():
                ui.card_header("Keyword Frequency")

                @render.text
                def keyword_analysis():
                    return free_churro

with ui.nav_panel("Post title analysis"):
    with ui.layout_sidebar():
        with ui.sidebar(open="desktop"):
            # Create the topic selection input with "All Topics" as the first option
            ui.input_select(
                "subredditSelect", 
                "Choose subreddit:", 
                choices=subreddit_choices
            )
        with ui.layout_column_wrap(height="300px"):
            with ui.card(full_screen=True, min_height="300px"):
                ui.card_header("Top 20 Posts by Comments")

                @render.ui
                def top_20_posts():
                    # Filter data based on the selected subreddit
                    selected_subreddit = input.subredditSelect()  # Assuming you have a dropdown for subreddit selection
                    data = df[df['subreddit'] == selected_subreddit]

                    # Check if there's data for the selected subreddit
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
                ui.card_header("Post Title Word Cloud")

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
                    

            with ui.card():
                ui.card_header("Username Frequency")

                @render.ui
                def subreddit_frequency():
                    # Get the subreddit data from the filtered dataset
                    subreddits = filtered_by_post_data()['username']
                    
                    # Exclude entries with "[deleted]", "sneakpeek_bot", and "AutoModerator"
                    subreddits = subreddits[~subreddits.isin(["[deleted]", "sneakpeek_bot", "AutoModerator"])]
                    
                    # Calculate the most frequent subreddits
                    subreddit_freq = subreddits.value_counts().nlargest(20)
                    
                    # Convert to a list of strings in the format "Rank. Username: Frequency"
                    top_subreddits_list = [f"{rank}. {subreddit}: {count}" 
                                        for rank, (subreddit, count) in enumerate(subreddit_freq.items(), start=1)]
                    
                    # Render as a UI component, separated by line breaks
                    return ui.HTML("<br>".join(top_subreddits_list))
        
        with ui.layout_column_wrap():
            with ui.card():
                ui.card_header("Post Title hate analysis")

                @render.ui
                def general_proportion_of_hate_comments_img():
                    filepath = f'{app_dir}\\wordclouds\\general_proportion_of_hate_comments.png'

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
                    
                    

with ui.nav_panel("Trigger analysis"):
    with ui.layout_sidebar():
        with ui.sidebar(open="desktop"):
            # Text input and submit/reset buttons
            ui.input_text("input_text", "Enter text:")
            ui.input_action_button("submit_text", "Submit")  # Submit button
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

                    # Use LIME's prediction function to get probabilities
                    probs = lime_predict([entered_text], model, tokenizer)
                    
                    # Get probabilities for "Non-trigger" and "Trigger" classes
                    non_trigger_prob = probs[0][0]  # Assumes 0th index is "Non-trigger"
                    trigger_prob = probs[0][1]      # Assumes 1st index is "Trigger"
                    
                    return {"Non-trigger": non_trigger_prob, "Trigger": trigger_prob}

                # Render the prediction probabilities as a Plotly bar chart
                @render_plotly
                def display_prediction_probabilities():
                    probs = prediction_probabilities()

                    if probs is None:
                        return go.Figure()  # Return an empty figure if no text is entered

                    # Create the Plotly bar plot
                    fig = go.Figure(
                        data=[
                            go.Bar(
                                x=[probs["Non-trigger"], probs["Trigger"]],
                                y=["Non-trigger", "Trigger"],
                                orientation='h',
                                marker_color=["#add8e6", "#ffa07a"],  # Colors for each class
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

            with ui.card():
                ui.card_header("Predicted Topic with Explanation")

                # Function to get LIME explanation as a list of words and their contribution scores
                @reactive.Calc
                def lime_explanation():
                    entered_text = input.input_text()

                    # Check if there is text to analyze
                    if not entered_text:
                        return None

                    # Generate explanation with LIME, passing model and tokenizer using a lambda
                    exp = explainer.explain_instance(
                        entered_text,
                        lambda text: lime_predict(text, model, tokenizer),
                        num_features=10  # Number of words to highlight
                    )

                    # Extract the explanation as a list of words and scores
                    explanation_list = exp.as_list()  # List of words with their contribution scores
                    return explanation_list

                # Render the LIME explanation as a Plotly bar plot
                @render_plotly
                def display_lime_explanation():
                    explanation_list = lime_explanation()

                    if explanation_list is None:
                        return go.Figure()  # Return an empty figure if no text is entered

                    # Unpack words and scores
                    words, scores = zip(*explanation_list)  # Separate words and their scores

                    # Define colors based on contribution direction
                    colors = ["#ffa07a" if score > 0 else "#add8e6" for score in scores]

                    # Create the Plotly bar plot
                    fig = go.Figure(
                        data=[
                            go.Bar(
                                x=scores,
                                y=words,
                                orientation='h',
                                marker_color=colors,
                                text=[f"<b>{word}</b>: {score:.2f}" for word, score in zip(words, scores)],  # Bold words and show score
                                textposition="auto",  # Automatically place text on top of each bar
                                textfont=dict(size=16, color="black", family="Arial, bold"),  # Increase font size, set color, and bold
                                hovertemplate='%{text}<extra></extra>'  # Show text on hover
                            )
                        ]
                    )

                    # Update layout for readability
                    fig.update_layout(
                        title="LIME Explanation for Input Text",
                        xaxis_title="Contribution Score",
                        yaxis_title="",
                        yaxis=dict(autorange="reversed", showticklabels=False),  # Remove y-axis labels
                        height=400,
                        margin=dict(l=60, r=20, t=60, b=20),
                    )

                    return fig
                

        with ui.layout_columns():
            with ui.card():
                ui.card_header("Text Attribution Explanation")

                @render.ui
                def display_lime_html_text():
                    text = input.input_text()

                    if text is None or not text:
                        return ui.HTML("<p>Enter text to see the explanation.</p>")
                    # Generate HTML text with token colors and styles based on LIME
                    html_content = generate_lime_html(text)
                    
                    # Display the HTML content within the Shiny Express UI
                    return ui.HTML(html_content)

# # --------------------------------------------------------
# # Reactive calculations and effects
# # --------------------------------------------------------

@reactive.calc
def filtered_time_series_data():
    # Retrieve the selected topics (could be a list if multiple are selected)
    selected_topics = input.topicSelect()
    
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

        # Start with a copy of the DataFrame
        data = df.copy()
        print(f"Initial data shape: {data.shape}")  # Debugging output

        # Filter by subreddit if not "All"
        if selected_subreddit != "All":
            data = data[data['subreddit'] == selected_subreddit]
            print(f"Filtered data shape: {data.shape}")  # Debugging output

        # Check if data is empty and handle it gracefully
        if data.empty:
            print("Warning: Filtered data is empty.")

        return data

    except Exception as e:
        print(f"Error in filtered_by_post_data: {e}")
        return pd.DataFrame() 

# @reactive.Effect
# @reactive.event(input.add_all)  # Trigger when "Add All Topics" button is clicked
# def add_all_topics():
#     # Set each topic individually to allow easier removal later
#     all_topics = [topic for topic in topic_choices if topic != "All"]
#     for topic in all_topics:
#         # Append the topic to the current selection
#         current_selection = input.topicSelect() or []
#         if topic not in current_selection:
#             ui.update_selectize("topicSelect", selected=current_selection + [topic])

@reactive.Effect
@reactive.event(input.reset)  # Trigger when "Reset Selection" button is clicked
def reset_topics():
    # Clear the topicSelect input selection
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