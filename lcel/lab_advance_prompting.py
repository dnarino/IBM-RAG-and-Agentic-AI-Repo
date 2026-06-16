import warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn
warnings.filterwarnings('ignore')

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from core.watson_client import get_watsonx_llm

# 1. Get the model engine
llm = get_watsonx_llm()

# 2. Write a Zero-shot prompt
prompt_txt = """Classify the following statement as true or false: 
'The Eiffel Tower is located in Berlin.'

Answer:"""

# 3. Invoke and print!
print("Sending prompt to Watson...")
response = llm.invoke(prompt_txt)
print(response)
