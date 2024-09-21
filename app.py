import dash

from dash import html
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import ThemeSwitchAIO
suppress_callback_exceptions=True

# http://127.0.0.1:8050/
# http://127.0.0.1:5012
PORT = 5012
ADDRESS = '127.0.0.1'

# select the Bootstrap stylesheets and figure templates for the theme toggle here:
url_theme1 = dbc.themes.FLATLY
url_theme2 = dbc.themes.DARKLY

# This stylesheet defines the "dbc" class.  Use it to style dash-core-components
# and the dash DataTable with the bootstrap theme.
# dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"

app = dash.Dash(
    __name__,
    use_pages=True,
    # external_stylesheets=[dbc.themes.BOOTSTRAP,url_theme2], #  
    external_stylesheets=[dbc.themes.CERULEAN,'/assets/custom.css',], # SLATE
    suppress_callback_exceptions=True,
    prevent_initial_callbacks = True
)


theme_toggle = ThemeSwitchAIO(
    aio_id="theme",
    themes=[url_theme2, url_theme1],
    icons={"left": "fa fa-sun", "right": "fa fa-moon"},
)

footer = dbc.Container(
    dbc.Row(
        [
            dbc.Col(html.A("Richie Bao | GitHub", href="https://github.com/richieBao"), align="right"),
        ],
    ),
    className="footer",
    fluid=True,
)

navbar = dbc.NavbarSimple(
    dbc.Nav(
        [
            dbc.NavLink(page["name"], href=page["path"])
            for page in dash.page_registry.values()
        ],
    ),
    brand="Daisy——算法交互图解",
    color="primary",
    dark=True,
    className="mb-2",
)


app.layout = dbc.Container(
    [navbar, dash.page_container,theme_toggle,footer], # theme_toggle,
    fluid=True,
)


if __name__ == "__main__":
    # print([(page['name'],page['path']) for page in  dash.page_registry.values()])
    app.run_server(
        port=PORT,
        host=ADDRESS,
        debug=True)