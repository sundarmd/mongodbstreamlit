import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pymongo import MongoClient
import plotly.express as px
from dotenv import load_dotenv
import os
import plotly.graph_objects as go
import numpy as np
from openai import AzureOpenAI

# Load environment variables
load_dotenv()

# LLM Connection Details
API_KEY = os.getenv('API_KEY')
MODEL_NAME = os.getenv('MODEL_NAME')
ENDPOINT = os.getenv('ENDPOINT')

# Embedding model
EMB_API_KEY = os.getenv('EMB_API_KEY')
EMB_MODEL_NAME = os.getenv('EMB_MODEL_NAME')
EMB_ENDPOINT = os.getenv('EMB_ENDPOINT')

# MongoDB connection
MONGO_URI = os.getenv('MONGO_URI')
client = MongoClient(MONGO_URI)
db = client['chatbot_database']
collection = db['iris_dataset']

# Initialize Azure OpenAI client
azure_client = AzureOpenAI(
    api_key=API_KEY,
    api_version="2023-05-15",
    azure_endpoint=ENDPOINT
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat interface
st.title("Data Visualization Chatbot")

def create_llm_prompt(df, query):
    # Get DataFrame schema
    schema = df.dtypes.to_string()
    
    # Get sample data (first few rows)
    sample_data = df.head(3).to_string()
    
    # Construct the prompt
    prompt = f"""You are a data visualization code generator. Your task is to generate ONLY the Python code needed to create a visualization based on the user's request.

DataFrame Information:
Schema:
{schema}

Sample Data:
{sample_data}

Available libraries (already imported):
- plotly.express as px
- plotly.graph_objects as go
- numpy as np
- DataFrame is named 'df'

User Request: {query}

IMPORTANT INSTRUCTIONS:
1. Return ONLY the Python code, nothing else
2. The code MUST create a variable named 'fig'
3. Do not include any explanations, markdown, or comments
4. Do not include any print statements
5. Do not include any imports
6. Start directly with the code that creates the visualization

Example of expected response format:
fig = px.scatter(df, x='column1', y='column2')
"""
    
    return prompt

def process_query(query):
    print("\n=== Query Processing Started ===")
    print(f"User Query: {query}")
    
    # Convert MongoDB data to DataFrame
    data = pd.DataFrame(list(collection.find({}, {'_id': 0})))
    print(f"DataFrame Shape: {data.shape}")
    print("DataFrame Columns:", data.columns.tolist())
    
    try:
        # Generate prompt for LLM
        prompt = create_llm_prompt(data, query)
        
        # Get visualization code from LLM
        print("\n=== Getting LLM Response ===")
        llm_response = get_llm_response(prompt)
        
        # Execute the visualization code safely
        try:
            print("\n=== Executing Visualization Code ===")
            print("Code to execute:\n", llm_response)
            
            # Create local namespace with required imports and data
            namespace = {
                'px': px,
                'go': go,
                'df': data,
                'np': np
            }
            
            # Execute the LLM's code in the namespace
            exec(llm_response, namespace)
            
            # The code should create a 'fig' variable
            if 'fig' in namespace:
                print("Successfully created visualization")
                return {"type": "viz", "content": namespace['fig']}
            else:
                print("No 'fig' variable created in namespace")
                print("Namespace keys:", list(namespace.keys()))
                return {"type": "text", "content": f"Visualization code didn't create a 'fig' variable. Generated code:\n{llm_response}"}
                
        except Exception as e:
            print(f"\n=== Execution Error ===\n{str(e)}")
            return {"type": "text", "content": f"Error creating visualization: {str(e)}\nGenerated code:\n{llm_response}"}
            
    except Exception as e:
        print(f"\n=== Processing Error ===\n{str(e)}")
        return {"type": "text", "content": f"Error processing query: {str(e)}"}

def get_llm_response(prompt):
    """
    Function to get response from LLM API using Azure OpenAI client.
    """
    try:
        print("\n=== LLM Request ===")
        print("Sending prompt:", prompt[:200] + "...")
        
        response = azure_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        
        if not response.choices:
            raise ValueError("No response content received from API")
        
        print("\n=== LLM Response ===")
        raw_response = response.choices[0].message.content.strip()
        print("Raw response:", raw_response)
        
        # Clean the response by removing markdown code blocks
        cleaned_response = raw_response
        if "```" in cleaned_response:
            # Extract code between triple backticks
            cleaned_response = cleaned_response.split("```")[1]
            # Remove language identifier if present
            if cleaned_response.startswith("python"):
                cleaned_response = cleaned_response[6:]
        
        # Remove any remaining backticks and whitespace
        cleaned_response = cleaned_response.strip('`').strip()
        
        print("Cleaned response:", cleaned_response)
        
        # Basic validation
        if not cleaned_response.startswith("fig ="):
            print("WARNING: Response doesn't start with 'fig ='")
        
        return cleaned_response
        
    except Exception as e:
        print(f"\n=== LLM Error ===\n{str(e)}")
        raise Exception(f"Error getting LLM response: {str(e)}")

# Chat input
user_input = st.chat_input("Ask me anything about your data...")

if user_input:
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Process response
    response = process_query(user_input)
    
    # Add response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Display chat history
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if isinstance(message["content"], dict) and message["content"]["type"] == "viz":
            st.plotly_chart(message["content"]["content"], key=f"viz_{i}")
        else:
            st.write(message["content"])
