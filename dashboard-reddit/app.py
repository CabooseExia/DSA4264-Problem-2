import faicons as fa
import plotly.express as px

# Load data and compute static values
from shared import app_dir, hate
from shinywidgets import render_plotly

from shiny import reactive, render
from shiny.express import input, ui

import pandas as pd

topic_to_words = hate.drop_duplicates(subset=['topic_number']).set_index('topic_number')['topic_words'].to_dict()
topics = ["All"] + hate['topic_number'].unique().tolist()
topic_choices = ["All"] + list(map(str, sorted(hate['topic_number'].unique(), key=int)))

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
            with ui.value_box(showcase=ICONS["eye"]):
                "Filtered Comments"

                @render.express
                def total_comments():
                    filtered_data().shape[0]

            with ui.value_box(showcase=ICONS["calendar"]):
                "Epic"

                @render.express
                def date_range_display():
                    "AND ITHICA'S WAITING, MY KINGDOM IS WAITING, PENELOPE'S WAITING, FOR ME. So full speed ahead"

            with ui.value_box(showcase=ICONS["hashtag"]):
                "Selected Topic"

                @render.text
                def selected_topic():
                    # Get the filtered data
                    data = filtered_data()
                    
                    # Ensure there is data after filtering
                    if not data.empty:
                        # Retrieve unique topic words from the filtered data
                        topic_words = data['topic_words'].iloc[0]  # Assuming only one unique set of words per topic
                        topic_number = data['topic_number'].iloc[0]  # Get the topic number
                        return f"Topic {topic_number}: {topic_words}"
                    else:
                        return "No words available for the selected topic."
        with ui.layout_columns(col_widths=[12, 6, 6]):

            with ui.card(full_screen=True):
                with ui.card_header(class_="d-flex justify-content-between align-items-center"):
                    ui.card_header("Time Series")

                @render_plotly
                def topic_plot():
                    topic_data = filtered_data().groupby(pd.Grouper(key='timestamp', freq='W')).size().reset_index(name='Count')
                    topic_data['EMA_4'] = topic_data['Count'].rolling(window=4).mean()

                    fig = px.line(topic_data, x='timestamp', y='Count', title="Weekly Count and 4-Week EMA")
                    fig.add_scatter(x=topic_data['timestamp'], y=topic_data['EMA_4'], mode='lines', name="4-Week EMA")
                    return fig
                
            with ui.card(full_screen=True):
                ui.card_header("Filtered Data Table")

                @render.data_frame
                def table():
                    return render.DataGrid(filtered_data())

            with ui.card(full_screen=True):
                with ui.card_header(class_="d-flex justify-content-between align-items-center"):
                    "Username Frequency"
                    ICONS["filter"]

                @render_plotly
                def username_frequency():
                    # Get the username data from the filtered dataset, excluding "[deleted]"
                    usernames = filtered_data()['username']
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
    "This is the second 'page'."

with ui.nav_panel("XAI Analysis"):
    "This is the third 'page'."


# ui.include_css(app_dir / "styles.css")

# # --------------------------------------------------------
# # Reactive calculations and effects
# # --------------------------------------------------------

@reactive.calc
def filtered_data():
    # # Retrieve the selected topic number
    selected_topic = input.topicSelect()
    
    # # Start with the full dataset
    data = hate

    if selected_topic != "All":
        data = data[data['topic_number'] == int(selected_topic)]
    
    return data


# @reactive.effect
# @reactive.event(input.reset)
# def _():
#     # ui.update_date_range("dateRange", value=date_range)
#     ui.update_checkbox("show_hate", value=False)
#     ui.update_select("topic_select", selected="All")


#     #i give up
