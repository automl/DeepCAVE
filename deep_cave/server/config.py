import os

external_stylesheets = ["https://cdn.jsdelivr.net/npm/bootstrap@5.0.0-beta1/dist/css/bootstrap.min.css"]

# location of all global variables for the server.
# every variable has a default value and can be set via environment variables
# Do not modify this file directly. Use environment variables instead.

# location on where to find the studies. Scheme is used to select the correct backed
studies_location = os.environ.get('STUDIES_LOCATION', '.') # ../../data
# If the data is in a third party format, specify a valid converter, and a converter is used instead of backend_storage
converter = os.environ.get('CONVERTER', None) # 'BOHBConverter'
# location on where to find the models.
models_location = os.environ.get('MODELS_LOCATION', None) # None
# Locations (separated by comma) for the autoimporter to import external plugins
external_plugins = os.environ.get('EXTERNAL_PLUGIN_LOCATIONS', '').split(',')
# Locations (separated by comma) for the autoimporter to import external converters
external_converters = os.environ.get('EXTERNAL_CONVERTER_LOCATIONS', '').split(',')
