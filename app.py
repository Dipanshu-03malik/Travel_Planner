import os
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import gradio as gr
import os
# ... (other imports)
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

llm = ChatGroq(
    temperature = 0,
    groq_api_key = os.getenv("GROQ_API_KEY"), 
    model_name = "llama-3.3-70b-versatile"
)
# Define the PlannerState TypedDict
class PlannerState(TypedDict):
  messages : Annotated[List[HumanMessage | AIMessage], "the messages in the conversation"]
  city: str
  interests: List[str]
  itinerary: str


# Define the itinerary prompt
itinerary_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful travel assistant. Create a day trip itinerary for {city} based on the user's interests: {interests}. Provide a brief, bulleted itinerary."),
    ("human", "Create an itinerary for my day trip."),
])

# Define the functions for the graph nodes
def input_city(state: PlannerState) -> PlannerState:
  # In the Gradio interface, the city input will be directly passed,
  # so no console input is needed here.
  # We update the state with the city from the Gradio input.
  return state

def input_interests(state: PlannerState) -> PlannerState:
  # In the Gradio interface, the interests input will be directly passed,
  # so no console input is needed here.
  # We update the state with the interests from the Gradio input.
  return state

def create_itinerary(state: PlannerState) -> str:
  print(f"Creating an itinerary for {state['city']} based on interests : {', '.join(state['interests'])}")
  response = llm.invoke(itinerary_prompt.format_messages(city = state['city'], interests = ','.join(state['interests'])))
  print("\nFinal Itinerary: ")
  print(response.content)
  
  # Update the state with the generated itinerary and AI message
  state["itinerary"] = response.content
  state["messages"].append(AIMessage(content=response.content))
  
  return response.content

# Define the Gradio application function
def travel_planner_gradio(city: str, interests_str: str) -> str:
    # Initialize state for each new request
    state: PlannerState = {
        "messages": [],
        "city": "",
        "interests": [],
        "itinerary": "",
    }

    # Manually update the state based on Gradio inputs
    state["city"] = city
    state["messages"].append(HumanMessage(content=city))
    
    parsed_interests = [interest.strip() for interest in interests_str.split(',') if interest.strip()]
    state["interests"] = parsed_interests
    state["messages"].append(HumanMessage(content=interests_str))

    # Since the graph is not strictly necessary for this linear flow in Gradio,
    # we can call the create_itinerary function directly.
    # However, to simulate the graph's intent, we'll pass the state.
    # The `create_itinerary` function is designed to return the itinerary string directly for Gradio output.
    final_itinerary = create_itinerary(state)
    return final_itinerary

# Build the Gradio interface
interface = gr.Interface(
    fn=travel_planner_gradio,
    theme='Yntec/HaleyCH_Theme_Orange_Green',
    inputs=[
        gr.Textbox(label="Enter the city for your day trip"),
        gr.Textbox(label="Enter your interests (comma-separated)"),
    ],
    outputs=gr.Textbox(label="Generated Itinerary"),
    title="Travel Itinerary Planner",
    description="Enter a city and your interests to generate a personalized day trip itinerary.",
)

# Launch the Gradio application
if __name__ == "__main__":
    interface.launch()