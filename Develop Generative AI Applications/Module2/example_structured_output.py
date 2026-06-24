from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from core.openai_client import get_openai_chat

# 1. Setup the OpenAI LLM
parameters = {
    "max_tokens": 1024,
    "temperature": 0.2,
    "model_kwargs": {
        "top_p": 0.1
    }
}
openai_LLM = get_openai_chat(params=parameters)

# ----------------- LOCATION CHAIN -----------------
class LocationPydantic(BaseModel):
    locations: list[str] = Field(description='A list of 3 city names being visited')

location_prompt = PromptTemplate.from_template(
    template="""
    City: Bogota
    locations: ['monserrate', 'candelaria','simon bolivar park']
    
    Come up with a list of 3 locations from {country}. Output only the names.
    """
)
# 🌟 THE UPGRADE: We bind the Pydantic class directly to the LLM! No Parsers!
structured_location_llm = openai_LLM.with_structured_output(LocationPydantic)
location_chain = location_prompt | structured_location_llm


# ----------------- ACTIVITIES CHAIN -----------------
class ActivitiesPydantic(BaseModel):
    activities: dict[str, list[str]] = Field(description="A dictionary mapping each city name to a list of 3 specific activities.")     

activities_prompt = PromptTemplate.from_template(
    template="""
    locations: ['monserrate', 'candelaria','simon bolivar park']
    activities: {{
        'monserrate': ['subir escaleras','comer en el restaurante de la cima','subir al telesferico'],
        'candelaria': ['visitar las actividades bibliteca luis angel arango','comer en la puerta false','ir al chorro de quevedo'],
        'simon bolivar park': ['correr','ir a los conciertos al aire libre','yoga']
    }}
    
    Provide 3 activities to do in these locations: {locations} 
    Remember to provide exactly 3 activities for each location. 
    """
)
structured_activities_llm = openai_LLM.with_structured_output(ActivitiesPydantic)
activities_chain = activities_prompt | structured_activities_llm


# ----------------- TOUR CHAIN -----------------
class DailySchedule(BaseModel):
    location: str = Field(description="The city visited that day")
    Morning: str = Field(description="Morning activity")
    Afternoon: str = Field(description="Afternoon activity")
    Evening: str = Field(description="Evening activity")

class TourPydantic(BaseModel):
    tour: dict[str, DailySchedule] = Field(description="A dictionary mapping the Day to the DailySchedule")

tour_prompt = PromptTemplate.from_template(
    template="""
    You are an expert travel agent. 
    You have planned a trip to {country} with the following activities:
    {activities}
    
    Create a logical, day-by-day tourist itinerary (tour). 
    Allocate one location per day. For each day, organize the 3 activities into Morning, Afternoon, and Evening slots.
    """
)
structured_tour_llm = openai_LLM.with_structured_output(TourPydantic)
tour_chain = tour_prompt | structured_tour_llm


# ----------------- MASTER PIPELINE -----------------
overall_chain = (
    {"country": RunnablePassthrough()}
    | RunnablePassthrough.assign(locations=location_chain)
    | RunnablePassthrough.assign(activities=activities_chain)
    | RunnablePassthrough.assign(tour=tour_chain)
)

if __name__== "__main__":
    print("Running Ultra-Modern Structured Output travel chain using OpenAI...")
    result = overall_chain.invoke("Italy")
    
    print("\n--- FINAL STRUCTURED TOUR OBJECT ---")
    # The result is natively returned as a living Pydantic object!
    print(result["tour"].model_dump_json(indent=4))
