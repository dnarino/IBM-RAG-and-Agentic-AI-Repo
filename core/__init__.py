# Init core module
import os
import warnings
from dotenv import load_dotenv

# Automatically load API keys from .env
load_dotenv()

# Disable LangChain telemetry globally
os.environ['ANONYMIZED_TELEMETRY'] = 'False'


# Suppress annoying warnings globally
def warn(*args, **kwargs):
    pass
warnings.warn = warn
warnings.filterwarnings('ignore')
