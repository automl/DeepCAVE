import automl_sphinx_theme

from deepcave import author, copyright, name, version

options = {
    "copyright": copyright,
    "author": author,
    "version": version,
    "name": name,
    "html_theme_options": {
        "github_url": "https://github.com/automl/DeepCAVE",
        "twitter_url": "https://twitter.com/automl_org?lang=de",
    },
    "sphinx_gallery_conf": {
        "examples_dirs": "../examples",
        "ignore_pattern": ".*logs$|.*__pycache__$|.*_pending$",
    },
    "linkcheck_request_headers" : {
        r'https://docs.github.com/': {
            'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux i686; rv:24.0) '
                          'Gecko/20100101 Firefox/24.0'}
    }
}

automl_sphinx_theme.set_options(globals(), options)
