import faicons as fa
import plotly.express as px

# Load data and compute static values
from shared import app_dir, df
from shinywidgets import render_plotly

from shiny import reactive, render
from shiny.express import input, ui

import pandas as pd
from wordcloud import WordCloud
from io import BytesIO
import base64

topic_to_words = df.drop_duplicates(subset=['topic_number']).set_index('topic_number')['topic_words'].to_dict()
topics = ["All"] + df['topic_number'].unique().tolist()
topic_choices = ["All"] + list(map(str, sorted(df['topic_number'].unique(), key=int)))

subreddit = df['subreddit'].unique().tolist()
subreddit = [x for x in subreddit if str(x) != 'nan']
subreddit_choices = ["All"] + subreddit

ui.page_opts(title="Reddit Comment Analysis Dashboard", fillable=True)

# Icons for the main content
ICONS = {
    "house": fa.icon_svg("house"),
    "eye": fa.icon_svg("eye"),
    "calendar": fa.icon_svg("calendar"),
    "hashtag": fa.icon_svg("hashtag"),
    "filter": fa.icon_svg("filter"),
}

# Main content with value boxes and plots
with ui.nav_panel('Time series analysis'):
    with ui.layout_sidebar():
        with ui.sidebar(open="desktop"):
            # Create the topic selection input with "All Topics" as the first option
            ui.input_select(
                "topicSelect", 
                "Choose Topic:", 
                choices=topic_choices
            )

        with ui.layout_columns(fill=False):
            with ui.value_box(showcase=ICONS["eye"], style="height: 100px;"):
                "Filtered Comments"

                @render.express
                def total_comments():
                    filtered_by_topic_data().shape[0]

            with ui.value_box(showcase=ICONS["calendar"], style="height: 100px;"):
                "idk what to put here"

                @render.express
                def date_range_display():
                    "AND ITHACA'S WAITING"

            with ui.value_box(showcase=ICONS["hashtag"], style="height: 100px; font-size: 8px;"):
                "Selected Topic"

                @render.text
                def selected_topic():
                    # Get the filtered data
                    data = filtered_by_topic_data()
                    
                    # Check if "All" is selected and return all topics and words if true
                    if input.topicSelect() == "All":
                        return 'Singapore'
                    else:
                        # Display only the selected topic
                        if not data.empty:
                            topic_words = data['topic_words'].iloc[0]
                            topic_number = data['topic_number'].iloc[0]
                            return f"Topic {topic_number}: {topic_words}"
                        else:
                            return "No words available for the selected topic."
                        
        with ui.layout_columns(col_widths=[12, 6, 6]):

            with ui.card(full_screen=True):
                with ui.card_header(class_="d-flex justify-content-between align-items-center"):
                    ui.card_header("Time Series")

                @render_plotly
                def topic_plot():
                    topic_data = filtered_by_topic_data().groupby(pd.Grouper(key='timestamp', freq='W')).size().reset_index(name='Count')
                    topic_data['EMA_4'] = topic_data['Count'].rolling(window=4).mean()

                    fig = px.line(topic_data, x='timestamp', y='Count', title="Weekly Count and 4-Week EMA")
                    fig.add_scatter(x=topic_data['timestamp'], y=topic_data['EMA_4'], mode='lines', name="4-Week EMA")

                    fig.update_layout(height=400)
                    return fig
                
            with ui.card(full_screen=True):
                ui.card_header("Filtered Data Table")

                @render.data_frame
                def table():
                    return render.DataGrid(filtered_by_topic_data())

            with ui.card(full_screen=True):
                with ui.card_header(class_="d-flex justify-content-between align-items-center"):
                    "Username Frequency"
                    ICONS["filter"]

                @render_plotly
                def username_frequency():
                    # Get the username data from the filtered dataset, excluding "[deleted]"
                    usernames = filtered_by_topic_data()['username']
                    usernames = usernames[usernames != "[deleted]"]  # Exclude "[deleted]"
                    
                    # Calculate the most frequent usernames
                    username_freq = usernames.value_counts().nlargest(20)
                    
                    # Create a bar plot for the top 20 most frequent usernames
                    fig = px.bar(username_freq, x=username_freq.index, y=username_freq.values, 
                                labels={'x': 'Username', 'y': 'Frequency'}, 
                                title="Top 20 Most Frequent Usernames (Excluding [deleted])")
                    fig.update_layout(xaxis={'categoryorder': 'total descending'})
                    return fig


with ui.nav_panel("Post title analysis"):
    with ui.layout_sidebar():
        with ui.sidebar(open="desktop"):
            # Create the topic selection input with "All Topics" as the first option
            ui.input_select(
                "subredditSelect", 
                "Choose subreddit:", 
                choices=subreddit_choices
            )
        with ui.layout_column_wrap():
            with ui.card():
                ui.card_header(":)")

                @render.text
                def post_table():
                    return 'MY KINGDOM IS WAITING'
            
            with ui.card():
                ui.card_header("Post Title Word Cloud")
            
                # @render.image
                # def wordcloud_img():
                #     # Generate the word cloud from RAKE keywords
                #     data = filtered_by_post_data()  # Assumes this function returns a DataFrame
                #     all_text = ' '.join([' '.join(eval(row)) if isinstance(row, str) else '' for row in data['rake_keywords']])


                #     # Convert the image to bytes
                #     wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_text)
                #     file_path = f"{app_dir}/temp/wordcloud.png"
                #     wordcloud.to_file(file_path)
                    
                #     # Return the path for Shiny Express to display it as an image
                #     return {"src": file_path, "alt": "Word Cloud"}
                @render.ui
                def wordcloud_img():
                    # Generate the word cloud from RAKE keywords
                    data = filtered_by_post_data()  # Assumes this function returns a DataFrame
                    all_text = ' '.join([' '.join(eval(row)) if isinstance(row, str) else '' for row in data['rake_keywords']])

                    # Generate the word cloud
                    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_text)

                    # Save the word cloud to an in-memory file
                    img_io = BytesIO()
                    wordcloud.to_image().save(img_io, format="PNG")
                    img_io.seek(0)

                    # Encode the image in base64
                    base64_img = base64.b64encode(img_io.getvalue()).decode('utf-8')
                    img_src = f"data:image/png;base64,{base64_img}"

                    # Return the image as an HTML <img> tag
                    return ui.HTML(f'<img src="{img_src}" alt="Word Cloud" style="width: 100%;">')


with ui.nav_panel("XAI analysis"):
    "PENELOPE'S WAITING FOR ME, so full speed ahead'."


# # --------------------------------------------------------
# # Reactive calculations and effects
# # --------------------------------------------------------

@reactive.calc
def filtered_by_topic_data():
    # # Retrieve the selected topic number
    selected_topic = input.topicSelect()
    
    # # Start with the full dataset
    data = df.copy()

    if selected_topic != "All":
        data = data[data['topic_number'] == int(selected_topic)]
    
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

