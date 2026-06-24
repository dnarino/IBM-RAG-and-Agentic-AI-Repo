from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_classic.output_parsers.fix import OutputFixingParser
from core.openai_client import get_openai_chat
import json

# 1. Setup the OpenAI LLM
# Note: OpenAI uses max_tokens instead of max_new_tokens!
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

location_parser = PydanticOutputParser(pydantic_object=LocationPydantic)
location_instructions = location_parser.get_format_instructions()

location_prompt = PromptTemplate.from_template(
    template="""
    City: Bogota
    locations: ['monserrate', 'candelaria','simon bolivar park']
    
    Come up with a list of 3 locations from {country}. Output only the names.
    
    CRITICAL: Output ONLY valid JSON. Do not write any code or conversational text.
    \n{location_instructions}
    """,
    partial_variables={'location_instructions': location_instructions}
)
location_chain = location_prompt | openai_LLM | location_parser

# ----------------- ACTIVITIES CHAIN -----------------
class ActivitiesPydantic(BaseModel):
    activities: dict[str, list[str]] = Field(description="A dictionary mapping each city name to a list of 3 specific activities.")     

activity_parser = PydanticOutputParser(pydantic_object=ActivitiesPydantic)
activity_instructions = activity_parser.get_format_instructions()

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
    
    CRITICAL: Output ONLY valid JSON. Do not write any code or conversational text.
    \n{activity_instructions}
    """,
    partial_variables={'activity_instructions': activity_instructions}
)

fixing_activity_parser = OutputFixingParser.from_llm(parser=activity_parser, llm=openai_LLM)
activities_chain = activities_prompt | openai_LLM | fixing_activity_parser

# ----------------- TOUR CHAIN -----------------
class DailySchedule(BaseModel):
    location: str = Field(description="The city visited that day")
    Morning: str = Field(description="Morning activity")
    Afternoon: str = Field(description="Afternoon activity")
    Evening: str = Field(description="Evening activity")

class TourPydantic(BaseModel):
    tour: dict[str, DailySchedule] = Field(description="A dictionary mapping the Day to the DailySchedule")

tour_parser = PydanticOutputParser(pydantic_object=TourPydantic)
tour_instructions = tour_parser.get_format_instructions()

tour_prompt = PromptTemplate.from_template(
    template="""
    You are an expert travel agent. 
    You have planned a trip to {country} with the following activities:
    {activities}
    
    Create a logical, day-by-day tourist itinerary (tour). 
    Allocate one location per day. For each day, organize the 3 activities into Morning, Afternoon, and Evening slots.
    
    CRITICAL: Output ONLY valid JSON. Do not write any code or conversational text.
    \n{tour_instructions}
    """,
    partial_variables={'tour_instructions': tour_instructions}
)

fixing_tour_parser = OutputFixingParser.from_llm(parser=tour_parser, llm=openai_LLM)
tour_chain = tour_prompt | openai_LLM | fixing_tour_parser

# ----------------- MASTER PIPELINE -----------------
overall_chain = (
    {"country": RunnablePassthrough()}
    | RunnablePassthrough.assign(locations=location_chain)
    | RunnablePassthrough.assign(activities=activities_chain)
    | RunnablePassthrough.assign(tour=tour_chain)
)

if __name__== "__main__":
    print("Running strict Pydantic travel chain using OpenAI (gpt-4o-mini)...")
    result = overall_chain.invoke("Italy")
    
    print("\n--- FINAL PYDANTIC TOUR OBJECT ---")
    print(result["tour"].model_dump_json(indent=4))