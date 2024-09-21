from dash import html
import dash_bootstrap_components as dbc
import dash

from .side_bar_heuristicAlgorithm import sidebar_heuristicAlgorithm
from .SMA import layout_SMA
from .topic_2 import layout_2
from .topic_3 import layout_3

def title(topic=None):
    # This will show in browser tab and the meta tags
    return f"About page: {topic}"


def description(topic=None):
    # This is the description for the meta tags.  It will show when you share a link to this page.
    if topic == "topic1":
        return "Here is more information on topic 1"
    return "Here is general info about the topics on this page"


dash.register_page(
    __name__,
    path_template="/meta_heuristic/<topic>",
    title=title,
    description=description,
    # sets a default for the path variable
    path="/meta_heuristic/SMA",
    # prevents showing a Page Not Found if someone enters /about in the browser
    redirect_from=["/meta_heuristic", "/meta_heuristic/"],
)


def layout(topic=None, **other_unknown_query_strings):
    parent_card =  dbc.Card(" Here is the main About Page content", body=True)

    if topic == "SMA":
        topic_card =  layout_SMA()
    elif topic == "topic-2":
        topic_card =  layout_2
    elif topic == "topic-3":
        topic_card =  layout_3
    else:
        topic_card= ""

    return dbc.Row(
        [dbc.Col(sidebar_heuristicAlgorithm(), width=2), dbc.Col([parent_card, topic_card], width=10)]
    )


