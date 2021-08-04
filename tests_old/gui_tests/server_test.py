import pytest

from deep_cave.server.server import app


@pytest.mark.skip
def test_sese001_start_up(dash_duo):
    # start the page as defined in server
    dash_duo.start_server(app)
    # wait some time until the page has loaded
    dash_duo.wait_for_page(timeout=5)
    # check for any errors
    assert dash_duo.get_logs() == [], "browser console should contain no error"
    # do a snap shot so later you can inspect the visual output
    dash_duo.percy_snapshot("bsly001-layout")


# 1. imports of your dash app
import dash
import dash_html_components as html


# 2. give each testcase a tcid, and pass the fixture
# as a function argument, less boilerplate
@pytest.mark.skip
def test_bsly001_falsy_child(dash_duo):

    # 3. define your app inside the test function
    app = dash.Dash(__name__)
    app.layout = html.Div(id="nully-wrapper", children=0)

    # 4. host the app locally in a thread, all dash server configs could be
    # passed after the first app argument
    dash_duo.start_server(app)

    # 5. use wait_for_* if your target element is the result of a callback,
    # keep in mind even the initial rendering can trigger callbacks
    dash_duo.wait_for_text_to_equal("#nully-wrapper", "0", timeout=4)

    # 6. use this form if its present is expected at the action point
    assert dash_duo.find_element("#nully-wrapper").text == "0"

    # 7. to make the checkpoint more readable, you can describe the
    # acceptance criterion as an assert message after the comma.
    assert dash_duo.get_logs() == [], "browser console should contain no error"

    # 8. visual testing with percy snapshot
    dash_duo.percy_snapshot("bsly001-layout")