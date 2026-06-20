# Init core module
import os
import warnings

# Disable LangChain telemetry globally
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

# Suppress annoying warnings globally
def warn(*args, **kwargs):
    pass
warnings.warn = warn
warnings.filterwarnings('ignore')
