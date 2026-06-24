from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from core.watson_client import get_watsonx_chat
import json

# 1. Setup the LLM
model_id='meta-llama/llama-3-3-70b-instruct'
parameters = {
    "max_new_tokens": 512,
    "temperature": 0.2,
    "top_k": 2,
    "top_p": 0.1
}
llama_LLM = get_watsonx_chat(model_id=model_id, params=parameters)

# 2. Define our 3 Prompts
location_prompt = PromptTemplate.from_template(
    "Come up with a classic dish from {location}. Output ONLY the dish name, nothing else."
)

dish_prompt = PromptTemplate.from_template(
    "Provide a brief 3-step recipe for {meal}."
)

recipe_prompt = PromptTemplate.from_template(
    "Estimate the total cooking time for this recipe:\n\n{recipe}\n\nOutput ONLY the estimated time (e.g. '45 minutes')."
)

# 3. Define our 3 independent LCEL Chains
location_chain = location_prompt | llama_LLM | StrOutputParser()
dish_chain = dish_prompt | llama_LLM | StrOutputParser()
recipe_chain = recipe_prompt | llama_LLM | StrOutputParser()

# 4. The Master Sequential Chain!
# This uses RunnablePassthrough.assign() to act like an assembly line!
overall_chain = (
    # Step 1: Accept the raw string and assign it to the dictionary key "location"
    {"location": RunnablePassthrough()} 
    
    # Step 2: Run location_chain (which looks for {location}) and assign the output to "meal"
    | RunnablePassthrough.assign(meal=location_chain)
    
    # Step 3: Run dish_chain (which looks for {meal}) and assign the output to "recipe"
    | RunnablePassthrough.assign(recipe=dish_chain)
    
    # Step 4: Run recipe_chain (which looks for {recipe}) and assign the output to "time"
    | RunnablePassthrough.assign(time=recipe_chain)
)

if __name__== "__main__":
    
    print("Starting the LCEL Sequential Assembly Line...\n")
    
    # We invoke the massive overall_chain with just the starting location!
    # Because of RunnablePassthrough, the output will be a beautiful Dictionary containing every step!
    result = overall_chain.invoke("Italy")
    
    print("--- RAW LCEL PIPELINE RESULT DICTIONARY ---")
    print(json.dumps(result, indent=4))
    
    print("\n--- FORMATTED OUTPUT ---")
    print(f"🌍 Location: {result['location']}")
    print(f"🍽️  Meal: {result['meal']}")
    print(f"⏱️  Time: {result['time']}")
    print(f"📜 Recipe:\n{result['recipe']}")