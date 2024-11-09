import faicons as fa
import plotly.express as px

# Load data and compute static values
from shared import app_dir, df, free_churro, final_topic_overview, load_model, lime_predict, generate_lime_html, compute_attributions, generate_integrated_gradients_html
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

earliest_date = df['timestamp'].min()
latest_date = df['timestamp'].max()

topic_to_words = df.drop_duplicates(subset=['topic_number']).set_index('topic_number')['topic_words'].to_dict()
topic_to_words = df[df['topic_number'] != -1].drop_duplicates(subset=['topic_number']).set_index('topic_number')['topic_words'].to_dict()
# topic_choices = ["All"] + [f"{topic}: {topic_to_words[topic]}" for topic in sorted(topic_to_words.keys(), key=int)]
topic_choices = [f"{topic}: {topic_to_words[topic]}" for topic in sorted(topic_to_words.keys())]

subreddit = df['subreddit'].unique().tolist()
subreddit = [x for x in subreddit if str(x) != 'nan']
subreddit_choices = ["All"] + subreddit

model, tokenizer = load_model()
explainer = LimeTextExplainer(class_names=["Non-trigger", "Trigger"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
}

# Main content with value boxes and plots
with ui.nav_panel('Time series analysis'):
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

        



with ui.nav_panel('Trend Drivers'):
    with ui.layout_sidebar():
        with ui.sidebar(open="desktop"):
            # Create the topic selection input with "All Topics" as the first option
            ui.input_selectize(
                "topicSelect_2", 
                "Choose Topic(s):", 
                choices=topic_choices, 
                multiple=False,  # Allow multiple selections
                options={"placeholder": "Select one or more topics..."}
    )
            # ui.input_action_button("add_all", "Add All Topics")
            ui.input_action_button("reset_2", "Reset Selection")
        
        with ui.layout_columns(fill=False):
            with ui.card():
                ui.card_header("Keyword Frequency")

                @render.ui
                def keyword_analysis():
                    return ui.HTML("""And deep down I know this well <br>
                                    I lost my best friend, I lost my mentor, my mom <br>
                                    Five hundred men gone, this can't go on <br>
                                    I must get to see Penelope and Telemachus <br>
                                    So if we must sail through dangerous oceans and beaches <br>
                                    I'll go where Poseidon won't reach us <br>
                                    And if I gotta drop another infant from a wall <br>
                                    In an instant so we all don't die""")
            with ui.card(height='800px'):
                ui.card_header("Keyword Word Cloud")

                @render.ui
                def keyword_wordcloud():
                    # Check if data is available from filtered_time_series_data()
                    data = unfiltered_data()
                    selected_topic = input.topicSelect_2()  # Get the topic as a single string like "1: topic1"
                    topic_number = int(selected_topic.split(":")[0].strip()) 
                    additional_words = {"singapore", "singaporean", "http", "gon", "na", "gt"}
                    final_topics_overview = final_topic_overview
                    start_date = input.date_range()[0]
                    end_date = input.date_range()[1]
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

                    if data.empty or not topic_number:
                        return ui.HTML("<p>No data or topic selected for word cloud generation.</p>")

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



with ui.nav_panel("Post title analysis"):
    with ui.layout_sidebar():
        with ui.sidebar(open="desktop"):
            # Create the topic selection input with "All Topics" as the first option
            ui.input_select(
                "subredditSelect", 
                "Choose subreddit:", 
                choices=subreddit_choices
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
            with ui.card(full_screen=True, min_height="300px"):
                ui.card_header("Top 20 Posts by Comments")

                @render.ui
                def top_20_posts():
                    # Get the selected subreddit from the dropdown
                    selected_subreddit = input.subredditSelect()  # Assuming you have a dropdown for subreddit selection
                    
                    # Filter data based on the selected subreddit
                    if selected_subreddit == "All":
                        data = df  # Use the entire dataset when "All" is selected
                    else:
                        data = df[df['subreddit'] == selected_subreddit]

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
                ui.card_header("Total Keyword Occurrences Over Time")

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


with ui.nav_panel("Trigger analysis"):
    with ui.layout_sidebar():
        with ui.sidebar(open="desktop"):
            ui.input_select(
                "card_selection",
                "Select model to display:",
                choices=["Lime", "Integrated Gradients"]
            )
            # Text input and submit/reset buttons
            ui.input_text("input_text", "Enter text:")
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

                    # Use LIME's prediction function to get probabilities
                    probs = lime_predict([entered_text], model, tokenizer)
                    
                    # Get probabilities for "Non-trigger" and "Trigger" classes
                    non_trigger_prob = probs[0][0]  # Assumes 0th index is "Non-trigger"
                    trigger_prob = probs[0][1]      # Assumes 1st index is "Trigger"
                    
                    return {"Non-trigger": non_trigger_prob, "Trigger": trigger_prob}
                
                @reactive.Calc
                def integrated_gradients_probabilities():
                    # Get input text from the app input
                    input_text = input.input_text()
                    
                    # Tokenize the input text
                    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
                    input_ids = inputs["input_ids"]
                    attention_mask = inputs["attention_mask"]
                    
                    # Generate embeddings from input_ids using BERT’s embedding layer
                    embeddings = model.bert.embeddings(input_ids).detach().requires_grad_(True)
                    
                    # Define a forward function for Integrated Gradients
                    def forward_func(embeddings, attention_mask):
                        outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask)
                        logits = outputs.logits
                        return torch.softmax(logits, dim=-1)
                    
                    # Initialize Integrated Gradients
                    integrated_gradients = IntegratedGradients(forward_func)
                    
                    # Compute attributions with Integrated Gradients
                    attributions = integrated_gradients.attribute(
                        embeddings,
                        baselines=torch.zeros_like(embeddings).to(device),  # Baseline as zero tensor
                        additional_forward_args=(attention_mask,),
                        target=1,  # Assuming class index 1 corresponds to "Trigger"
                        n_steps=50
                    )
                    
                    # Compute probabilities for each class
                    outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask)
                    probabilities = torch.softmax(outputs.logits, dim=-1).squeeze().tolist()  # Convert to a list for readability

                    return {"Non-trigger": probabilities[0], "Trigger": probabilities[1]}

                # Render the prediction probabilities as a Plotly bar chart
                @render_plotly
                def display_prediction_probabilities():
                    xai = input.card_selection()
                    if xai == "Lime":
                        probs = prediction_probabilities()

                        if probs is None:
                            return go.Figure()  # Return an empty figure if no text is entered

                        # Create the Plotly bar plot with updated colors for Integrated Gradients style
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
                        input_text = input.input_text()
                        if not input_text or input_text.strip() == "":
                            return go.Figure()

                        # Create the Plotly bar plot for Integrated Gradients
                        fig = go.Figure(
                            data=[
                                go.Bar(
                                    x=[probs["Non-trigger"], probs["Trigger"]],
                                    y=["Non-trigger", "Trigger"],
                                    orientation='h',
                                    marker_color=["#98FB98", "#FF6347"],  # Different colors for Integrated Gradients
                                    text=[f"{probs['Non-trigger']:.2f}", f"{probs['Trigger']:.2f}"],  # Display values
                                    textposition="auto",
                                    textfont=dict(size=16, color="black", family="Arial, bold"),
                                    hovertemplate='%{text}<extra></extra>'
                                )
                            ]
                        )

                        # Update layout for readability
                        fig.update_layout(
                            title="Prediction Probabilities (Integrated Gradients)",
                            xaxis_title="Probability",
                            yaxis=dict(
                                autorange="reversed",
                                tickfont=dict(size=18, family="Arial, bold")
                            ),
                            height=200,
                            margin=dict(l=80, r=20, t=30, b=20),
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
                def display_explanation():
                    xai = input.card_selection()
                    if xai == "Lime":
                        explanation_list = lime_explanation()

                        if explanation_list is None:
                            return go.Figure()  # Return an empty figure if no text is entered

                        # Unpack words and scores
                        words, scores = zip(*explanation_list)  # Separate words and their scores

                        # Define colors based on score direction (positive or negative)
                        colors = ["#98FB98" if score > 0 else "#FF6347" for score in scores]  # Light green for positive, red for negative

                        # Create the Plotly bar plot with the new color scheme
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
                    else:
                        text = input.input_text()  # Retrieve text from input

                        if text:
                            tokens, scores = compute_attributions(text)
                            
                            # Prepare data and sort by absolute attribution values
                            df = pd.DataFrame({"Token": tokens, "Attribution Score": scores})
                            df = df.reindex(df["Attribution Score"].abs().sort_values(ascending=False).index).head(10)
                            df["Token + Score"] = df["Token"] + " (" + df["Attribution Score"].round(2).astype(str) + ")"
                            df = df[::-1]  # Reverse for descending y-axis order

                            # Set colors based on score intensity and sign
                            colors = [f'rgba(255, 0, 0, {abs(score) / max(df["Attribution Score"].abs())})' if score > 0 else 
                                    f'rgba(0, 128, 0, {abs(score) / max(df["Attribution Score"].abs())})' for score in df["Attribution Score"]]

                            # Create the bar chart
                            fig = go.Figure(go.Bar(
                                x=df["Attribution Score"],
                                y=df["Token + Score"],
                                orientation='h',
                                marker=dict(color=colors),
                                text=["<b>" + label + "</b>" for label in df["Token + Score"]],
                                textposition="auto",
                                texttemplate='%{text}'
                            ))

                            # Customize layout
                            fig.update_layout(
                                title="Top 10 Token Attributions",
                                xaxis_title="Attribution Score",
                                yaxis_title="",
                                yaxis=dict(categoryorder="array"),
                                font=dict(family="Arial", size=12, color="black")
                            )
                            
                            return fig

                        # Return an empty figure if no text is entered
                        return go.Figure()

                

        with ui.layout_columns():
            with ui.card():
                ui.card_header("Text Attribution Explanation")

                @render.ui
                def display_lime_html_text():
                    xai = input.card_selection() 
                    text = input.input_text()

                    if text is None or not text:
                            return ui.HTML("<p>Enter text to see the explanation.</p>")
                    
                    if xai == "Lime":
                        # Generate HTML text with token colors and styles based on LIME
                        html_content = generate_lime_html(text)
                        
                        # Display the HTML content within the Shiny Express UI
                        return ui.HTML(html_content)
                    else:
                        # Generate HTML with token colors based on Integrated Gradients
                        html_content = generate_integrated_gradients_html(text)
                        
                    # Display the HTML content within the Shiny Express UI
                    return ui.HTML(html_content)

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