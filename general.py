import os
import datetime
import logging
from logging.handlers import RotatingFileHandler
import sys
import json
import re
from typing import List, Dict, Any, Optional, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, FunctionMessage
from langchain_openai import ChatOpenAI
import openai
from langgraph.graph import StateGraph, END
import requests
from datetime import datetime
import pytz
import logging
import time

import mysql.connector
from dotenv import load_dotenv
import os
import requests
from typing import Optional, Dict

load_dotenv()

# Use environment variables for DB configuration
DB_config = {
    'host': os.getenv('DB_HOST'),
    'port': int(os.getenv('DB_PORT')),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME')
}
# Load environment variables
load_dotenv()
ai_response = []
import queue
ai_response_queue = queue.Queue()

# Constants
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables")

load_dotenv()
GEOCODE_API_KEY = os.getenv("GEOCODE_API_KEY")  # Replace with your actual key

def get_formatted_address_tool(address: str) -> Dict:
    """
    Tool to get the formatted full address from the geocoding service.
    If an error occurs, return a message asking the user to confirm or re-enter the address.
    """
    if not address or not isinstance(address, str):
        return {
            "status": "invalid",
            "message": "The address provided is empty or invalid. Please provide a valid address."
        }

    try:
        if not GEOCODE_API_KEY:
            raise ValueError("No API key provided")

        encoded_address = address.replace(' ', '+')
        url = f"https://maps.googleapis.com/maps/api/geocode/json?address={encoded_address}&key={GEOCODE_API_KEY}"

        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"Geocoding API returned status code {response.status_code}")

        data = response.json()

        if data["status"] != "OK":
            raise Exception(f"Geocoding failed: {data['status']}")

        return {
            "status": "valid",
            "formatted_address": data["results"][0]["formatted_address"],
            "coordinates": data["results"][0]["geometry"]["location"]
            }

    except Exception as e:
        print(f"Warning: Geocoding error: {str(e)}")
        return {
            "status": "invalid",
            "message": f"The address '{address}' could not be validated. Please confirm or provide a valid address."
        }
# Logger setup
def get_last_5_records():
    """
    Fetches the 5 most recent ride booking records from the database.
    
    Returns:
        list: A list of tuples containing the ride booking records
    """
    # Load environment variables
    
    try:
        # Connect to the MySQL database
        conn = mysql.connector.connect(**DB_config)
        cursor = conn.cursor()
        
        # Set session timezone to Indian Standard Time (IST)
        cursor.execute("SET time_zone = '+05:30'")
        
        # Fetch the last 5 records, ordered by ID in descending order
        cursor.execute("SELECT * FROM ride_bookings ORDER BY id DESC LIMIT 5")
        records = cursor.fetchall()
        
        # Close the connection
        cursor.close()
        conn.close()
        
        return records
        
    except mysql.connector.Error as err:
        print(f"Database Error: {err}")
        return []

def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    # File handler
    file_handler = RotatingFileHandler('app.log', maxBytes=10*1024*1024, backupCount=5)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

# Initialize logger
logger = setup_logger(__name__)


# Config class
class AppConfig:
    def __init__(self):
        self.LLM_MODEL = "gpt-3.5-turbo-0125"  # Using OpenAI model
        self.CALLER_PA_PROMPT = """You are a ride booking assistant designed for voice conversations. Your primary function is to help users book rides through API calls.

CRITICAL REQUIREMENTS:
1. You MUST use the provided functions for every step of the booking process:
   - validate_location function for BOTH pickup and dropoff locations
   - book_ride function to create the actual booking

2. MANDATORY SEQUENCE:
   a. Collect pickup location → Call validate_location (notify user if invalid)
   b. Collect dropoff location → Call validate_location (notify user if invalid)
   c. Check if booking is "now" or "later"
   d. For "later" bookings, collect date and time in YYYY-MM-DD HH:MM format
   e. Confirm all details with user (in short)
   f. Call book_ride function
   g. Only confirm booking when book_ride returns a valid BookingID

3. NEVER skip steps in this sequence. NEVER proceed until each step is completed.

4. BOOKING ENFORCEMENT:
   - NEVER state a ride is booked unless book_ride function returns a successful response with a BookingID
   - NEVER create or make up booking IDs
   - If book_ride fails, inform the user and try again

COMMUNICATION GUIDELINES WITH USERS:
- Be concise and conversational (within 5 to 10 words if possible)
- Use natural language (contractions, casual tone)
- Don't announce function calls to users
- Avoid saying full addresses to users
- Keep confirmations brief

Current time: {current_time}
User's phone: {current_phone}"""

    def get_india_time_and_day(self):
        india_time = datetime.now(pytz.timezone('Asia/Kolkata'))
        formatted_time = india_time.strftime('%Y-%m-%d %H:%M:%S')
        weekday = india_time.strftime('%A')  # Gets full weekday name like "Monday"
        return formatted_time, weekday

# Initialize config
config = AppConfig()

# Global state
class GlobalState:
    def __init__(self):
        self.users = {}
        self.rides = []
        self.appointments = []

state = GlobalState()

# Tools implementation as LangChain tools
# Define tools as dictionaries to avoid Pydantic serialization issues

logger = logging.getLogger(__name__)

def get_india_time_and_day():
    india_time = datetime.now(pytz.timezone('Asia/Kolkata'))
    formatted_time = india_time.strftime('%Y-%m-%d %H:%M:%S')
    weekday = india_time.strftime('%A')  # Gets full weekday name like "Monday"
    return formatted_time, weekday

def validate_location(location_name: str, location: str) -> Dict:
    """
    Validate a location and return appropriate response.
    """
    validation = get_formatted_address_tool(location)
    if validation["status"] == "invalid":
        return {
            "status": "invalid",
            "type": location_name,
            "message": validation["message"]
        }
    else:
        #return "Location is valid, you can go to the next step for booking."
        return {
            "status": "valid",
            "type": location_name,
            "formatted_address": validation["formatted_address"]
        }

def book_ride(
    pickup_location: str,
    drop_location: str,
    booking_type: str,  # 'now' or 'later'
    scheduled_time: Optional[str] = None
) -> Dict:
    """
    Book a ride with the given details and save the booking to the database.
    """

    # Validate pickup location
    pickup_validation = get_formatted_address_tool(pickup_location)
    if pickup_validation["status"] == "invalid":
        return {
            "status": "error",
            "message": f"Invalid pickup location: {pickup_validation['message']}"
        }

    # Validate drop-off location
    drop_validation = get_formatted_address_tool(drop_location)
    if drop_validation["status"] == "invalid":
        return {
            "status": "error",
            "message": f"Invalid drop-off location: {drop_validation['message']}"
        }

    # Validate booking type
    if booking_type.lower() not in ["now", "later"]:
        return {
            "status": "error",
            "message": "Booking type must be 'now' or 'later'."
        }

    # Validate scheduled time if booking type is 'later'
    if booking_type.lower() == "later" and not scheduled_time:
        return {
            "status": "error",
            "message": "Scheduled time is required for future bookings."
        }

    # Proceed with booking if all validations pass
    pickup_formatted_address = pickup_validation["formatted_address"]
    drop_formatted_address = drop_validation["formatted_address"]
    india_time = datetime.now(pytz.timezone('Asia/Kolkata'))
    current_time = india_time.strftime('%Y-%m-%d %H:%M:%S')

    if booking_type.lower() == "now":
        BookDate, BookTime = current_time.split()
        scheduled_timestamp = current_time
    else:
        try:
            # Parse the scheduled time
            BookDate, BookTime = scheduled_time.replace("T", " ").split()
            scheduled_timestamp = f"{BookDate} {BookTime}"
        except Exception as e:
            return {
                "status": "error",
                "message": f"Invalid scheduled time format. Please use format 'YYYY-MM-DD HH:MM:SS': {str(e)}"
            }

    # Insert the ride booking into the database
    data = {
        "pickup_address": pickup_formatted_address,
        "dropoff_address": drop_formatted_address,
        "booking_type": booking_type,
        "schedule_time": scheduled_timestamp,
        "booking_date": current_time
    }
    print(data)
    try:
        conn = mysql.connector.connect(**DB_config)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO ride_bookings (
                pickup_address, dropoff_address, booking_type, schedule_time, booking_date
            ) VALUES (%s, %s, %s, %s, %s)
        """, (
            pickup_formatted_address, drop_formatted_address, booking_type,
            scheduled_timestamp, current_time
        ))

        booking_id = cursor.lastrowid  # ✅ Get inserted booking ID
        conn.commit()
        cursor.close()
        conn.close()

    except mysql.connector.Error as err:
        return {
            "status": "error",
            "message": f"Failed to save booking to database: {str(err)}"
        }

    # Success response
    return {
        "status": "success",
        "message": f"Ride has been booked successfully. Pickup: {pickup_formatted_address}, Drop-off: {drop_formatted_address}",
        "BookingID": booking_id
    }

# Create schema descriptions for tool definitions
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_formatted_address_tool",
            "description": "Get the formatted full address from the geocoding service. If invalid, prompt the user to confirm or re-enter.",
            "parameters": {
                "type": "object",
                "properties": {
                    "address": {
                        "type": "string",
                        "description": "The address to validate and format."
                    }
                },
                "required": ["address"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "validate_location",
            "description": "Validate a pickup or drop-off location and return appropriate response.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location_name": {
                        "type": "string",
                        "description": "Name of the location (pickup or dropoff)"
                    },
                    "location": {
                        "type": "string",
                        "description": "The address to validate"
                    }
                },
                "required": ["location_name", "location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "book_ride",
            "description": "MANDATORY function to book a ride. You MUST call this function to create a booking and get a valid booking ID. Never tell a user their ride is booked until you call this function and get a successful response.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pickup_location": {
                        "type": "string",
                        "description": "The validated pickup location"
                    },
                    "drop_location": {
                        "type": "string",
                        "description": "The validated drop-off location"
                    },
                    "booking_type": {
                        "type": "string",
                        "description": "The type of booking (now/later)"
                    },
                    "scheduled_time": {
                        "type": "string",
                        "description": "The scheduled time for the ride (required if booking_type is 'later')"
                    }
                },
                "required": ["pickup_location", "drop_location", "booking_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_india_time_and_day",
            "description": "Get the current time and day in India",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]

# Create a mapping from function name to actual function
tool_map = {
    "get_formatted_address_tool": get_formatted_address_tool,
    "validate_location": validate_location,
    "book_ride": book_ride,
    "get_india_time_and_day": get_india_time_and_day
}

# Agent State Type
class AgentState(TypedDict):
    messages: List[Any]
    current_time: str
    current_phone: str

# Global conversation state
CONVERSATION: List[Any] = []
CURRENT_PHONE: str = ""

def set_current_phone(phone: str) -> None:
    global CURRENT_PHONE
    CURRENT_PHONE = phone

def reset_session():
    global CONVERSATION, state
    print(f"Before reset: CONVERSATION={CONVERSATION}, CURRENT_PHONE={CURRENT_PHONE}, state={state.__dict__}")
    CONVERSATION.clear()
    #CURRENT_PHONE = ""
    state.users.clear()
    state.rides.clear()
    state.appointments.clear()

def receive_message_from_caller(message: str, phone_number: str = None) -> None:
    if phone_number:
        set_current_phone(phone_number)
    elif not CURRENT_PHONE:
        # If no phone number is provided and none is set, use a default
        set_current_phone("unknown")
    
    CONVERSATION.append(HumanMessage(content=message))
    current_time, current_day = config.get_india_time_and_day()
    state_data: AgentState = {
        "messages": CONVERSATION,
        "current_time": f"{current_time} ({current_day})",
        "current_phone": CURRENT_PHONE
    }
    
    try:
        new_state = caller_app.invoke(state_data)
        CONVERSATION.extend(new_state["messages"][len(CONVERSATION):])
        
        # Print the AI's response
        for msg in new_state["messages"][len(state_data["messages"]):]:
            if isinstance(msg, AIMessage):
                print(f"\nAssistant: {msg.content}")
    except Exception as e:
        print(f"Error in receive_message_from_caller: {str(e)}")
        raise

def should_continue_caller(state: AgentState) -> str:
    messages = state["messages"]
    if not messages:
        return "end"
        
    last_message = messages[-1]
    
    # If the last message is from the AI and it's requesting a tool call
    if isinstance(last_message, AIMessage) and hasattr(last_message, 'additional_kwargs') and last_message.additional_kwargs.get('tool_calls'):
        return "tools"
    
    # If it's a regular AI message, we're done
    if isinstance(last_message, AIMessage):
        return "end"
        
    # Otherwise continue the conversation
    return "continue"

def call_caller_model(state: AgentState) -> AgentState:
    #ai_response = []
    messages = state["messages"]
    current_time = state["current_time"]
    current_phone = state["current_phone"]

    try:
        system_message = config.CALLER_PA_PROMPT.format(
            current_time=current_time,
            current_phone=current_phone
        )
        formatted_messages = [
            SystemMessage(content=system_message)
        ]
        for m in messages:
            formatted_messages.append(m)

        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        openai_messages = []
        for msg in formatted_messages:
            if isinstance(msg, SystemMessage):
                openai_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                openai_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                openai_messages.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, FunctionMessage):
                openai_messages.append({"role": "function", "name": msg.name, "content": msg.content})
        # >>> TIMING START <<<
        start_time = time.time()
        # >>> END TIMING START <<<

        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=openai_messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.3,
            max_tokens=200  # Increased token limit for more detailed responses
        )

        # >>> TIMING END <<<
        elapsed = time.time() - start_time
        print(f"⏱️ LLM response latency: {elapsed:.3f} seconds")
        # >>> END TIMING END <<<

        response_message = response.choices[0].message
        if response_message.content:
            #print(f"AI: {response_message.content}")
            ai_response.append(response_message.content)
            #print(f"AI response: {ai_response}")
        
        # Check if there are tool calls
        tool_calls = response_message.tool_calls
        if tool_calls:
            ai_message = AIMessage(
                content=response_message.content or "",
                additional_kwargs={"tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    } for tc in tool_calls
                ]}
            )
        else:
            ai_message = AIMessage(content=response_message.content or "")
        
        return {
            "messages": messages + [ai_message],
            "current_time": current_time,
            "current_phone": current_phone
        }

    except Exception as e:
        print(f"Error in call_caller_model: {str(e)}")
        return {
            "messages": messages + [
                AIMessage(content="I'm sorry, I encountered an error. Could you please try again?")
            ],
            "current_time": current_time,
            "current_phone": current_phone
        }

def run_tools(state: AgentState) -> AgentState:
    messages = state["messages"]
    last_message = messages[-1]
    
    if isinstance(last_message, AIMessage) and hasattr(last_message, 'additional_kwargs'):
        tool_calls = last_message.additional_kwargs.get('tool_calls', [])
        
        if tool_calls:
            new_messages = list(messages)
            booking_id = None
            booking_attempted = False
            
            for tool_call in tool_calls:
                try:
                    # Extract function call details
                    function_name = tool_call['function']['name']
                    function_args = json.loads(tool_call['function']['arguments'])
                    
                    print(f"Executing function: {function_name} with args: {function_args}")
                    
                    # Track if book_ride was called
                    if function_name == "book_ride":
                        booking_attempted = True
                    
                    # Find the corresponding tool function
                    if function_name in tool_map:
                        # Execute the tool
                        result = tool_map[function_name](**function_args)
                        print(f"Tool result: {result}")
                        
                        # Track successful booking
                        if function_name == "book_ride" and isinstance(result, dict):
                            if result.get("status") == "success" and "BookingID" in result:
                                booking_id = result.get("BookingID")
                                print(f"Booking ID: {booking_id}")
                            
                        # Add function message with result
                        new_messages.append(FunctionMessage(
                            content=json.dumps(result),
                            name=function_name
                        ))
                    else:
                        error_msg = f"Function {function_name} not found"
                        print(f"Error: {error_msg}")
                        new_messages.append(FunctionMessage(
                            content=f"Error: {error_msg}",
                            name=function_name
                        ))
                        
                except Exception as e:
                    error_msg = f"Error executing tool {function_name}: {str(e)}"
                    print(f"Exception: {error_msg}")
                    new_messages.append(FunctionMessage(
                        content=f"Error: {error_msg}",
                        name=function_name if 'function_name' in locals() else "unknown_function"
                    ))
            
            # If booking was attempted but no booking ID was obtained, add a warning message
            if booking_attempted and booking_id is None:
                new_messages.append(FunctionMessage(
                    content=json.dumps({
                        "status": "warning",
                        "message": "IMPORTANT: Booking not done. Do NOT tell the user the booking was successful. Take the user through the booking process again."
                    }),
                    name="booking_verification" 
                ))
            
            return {
                "messages": new_messages,
                "current_time": state["current_time"],
                "current_phone": state["current_phone"]
            }
    
    return state
# Graph
caller_workflow = StateGraph(AgentState)

# Add Nodes
caller_workflow.add_node("agent", call_caller_model)
caller_workflow.add_node("tools", run_tools)

# Add Edges
caller_workflow.add_conditional_edges(
    "agent",
    should_continue_caller,
    {
        "continue": "agent",
        "tools": "tools",
        "end": END,
    },
)
caller_workflow.add_edge("tools", "agent")

# Set Entry Point and compile the graph
caller_workflow.set_entry_point("agent")
caller_app = caller_workflow.compile()
logger.info("Caller workflow compiled successfully")

def main():
    """Main function to run the ride booking system."""
    print("Welcome to the AI Ride Booking System!")
    print("Type your message or 'exit' to quit.")
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'exit':
            print("\nThank you for using our service. Goodbye!")
            reset_session()  # Reset session on exit
            break
        
        # Check if user is setting a phone number
        try:
            receive_message_from_caller(user_input)
            if ai_response:
                print(f"AI: {ai_response[-1]}")
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()