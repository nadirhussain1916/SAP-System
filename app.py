import os
import pandas as pd
from flask import Flask, request, jsonify, render_template_string,render_template
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
import openai
import threading
from datetime import datetime
import time
# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize Flask app
app = Flask(__name__)

# Global variables
df = None
langchain_agent = None
pandasai_agent = None
evaluator_llm = None
thread_conversations = {}  # Store conversations by thread_id
thread_lock = threading.Lock()

def initialize_agents():
    """Initialize both LangChain and PandasAI agents"""
    global df, langchain_agent, pandasai_agent, evaluator_llm
    
    try:
        # Load Excel data
        df = pd.read_excel("sap_logs.xlsx")
        print(f"Loaded Excel file with {len(df)} rows and {len(df.columns)} columns")
        
        # Initialize LangChain agent
        langchain_llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=openai_api_key)
        langchain_agent = create_pandas_dataframe_agent(
            langchain_llm, 
            df, 
            verbose=True, 
            allow_dangerous_code=True
        )
        
        # Initialize PandasAI agent with cache configuration
        pandasai_llm = OpenAI(api_token=openai_api_key)
        pandasai_config = {
            "llm": pandasai_llm,
            "enable_cache": False,  # Disable cache to avoid file conflicts
            "verbose": False
        }
        pandasai_agent = SmartDataframe(df, config=pandasai_config)
        
        # Initialize evaluator LLM for result assessment
        evaluator_llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=openai_api_key)
        
        print("Both agents initialized successfully")
        
    except Exception as e:
        print(f"Error initializing agents: {e}")
        raise e

def get_thread_context(thread_id):
    """Get conversation context for a thread"""
    with thread_lock:
        if thread_id not in thread_conversations:
            thread_conversations[thread_id] = {
                'messages': [],
                'created_at': datetime.now(),
                'last_activity': datetime.now()
            }
        return thread_conversations[thread_id]

def update_thread_context(thread_id, user_query, assistant_response, agent_used):
    """Update conversation context for a thread"""
    with thread_lock:
        context = thread_conversations[thread_id]
        context['messages'].append({
            'user': user_query,
            'assistant': assistant_response,
            'agent_used': agent_used,
            'timestamp': datetime.now()
        })
        context['last_activity'] = datetime.now()
        
        # Keep only last 10 exchanges to manage memory
        if len(context['messages']) > 10:
            context['messages'] = context['messages'][-10:]

def build_context_prompt(thread_id, current_query):
    """Build context-aware prompt from thread history"""
    context = get_thread_context(thread_id)
    
    if not context['messages']:
        return current_query
    
    # Build context from recent messages
    context_parts = ["Previous conversation context:"]
    for msg in context['messages'][-5:]:  # Last 5 exchanges
        context_parts.append(f"User: {msg['user']}")
        context_parts.append(f"Assistant: {msg['assistant'][:200]}...")  # Truncate long responses
    
    context_parts.append(f"\nCurrent question: {current_query}")
    context_parts.append("\nPlease answer considering the conversation context above.")
    
    return "\n".join(context_parts)

def evaluate_result(query, result, agent_type):
    """Evaluate if the result is satisfactory for the given query"""
    try:
        evaluation_prompt = f"""
        You are an expert data analyst evaluating the quality of responses to data queries.
        
        Original Query: "{query}"
        Agent Type: {agent_type}
        Agent Response: "{result}"
        
        Evaluate this response based on the following criteria:
        1. Does it directly answer the query?
        2. Is the response meaningful and not empty?
        3. Does it appear to contain actual data/insights?
        4. Is it not an error message or "I don't know" response?
        
        Respond with only one of these options:
        - "SATISFACTORY" if the response adequately answers the query
        - "UNSATISFACTORY" if the response is empty, wrong, or doesn't answer the query
        
        Be strict in your evaluation. If there's any doubt, choose UNSATISFACTORY.
        """
        
        evaluation = evaluator_llm.predict(evaluation_prompt)
        return "SATISFACTORY" in evaluation.upper()
        
    except Exception as e:
        print(f"Error in evaluation: {e}")
        return False

def query_langchain(query, thread_id):
    """Query using LangChain agent with context"""
    try:
        context_query = build_context_prompt(thread_id, query)
        result = langchain_agent.run(context_query)
        return result, None
    except Exception as e:
        return None, str(e)

def query_pandasai(query, thread_id):
    """Query using PandasAI agent with context"""
    try:
        context_query = build_context_prompt(thread_id, query)
        result = pandasai_agent.chat(context_query)
        return result, None
    except Exception as e:
        return None, str(e)

def generate_final_response(query, langchain_result, pandasai_result, chosen_agent, thread_id):
    """Generate a final, well-formatted response using the evaluator LLM"""
    try:
        if chosen_agent == "langchain":
            raw_result = langchain_result
            agent_used = "LangChain"
        else:
            raw_result = pandasai_result
            agent_used = "PandasAI"
        
        # Get conversation context for better formatting
        context = get_thread_context(thread_id)
        context_info = ""
        if context['messages']:
            context_info = "Consider the ongoing conversation context when formatting your response."
        
        formatting_prompt = f"""
        You are a helpful data analyst assistant. Format this response in a clear, professional manner.
        
        Original Query: "{query}"
        Raw Result from {agent_used}: "{raw_result}"
        {context_info}
        
        Please provide a clean, well-formatted response that:
        1. Directly answers the user's question
        2. Presents the information clearly
        3. Includes relevant context if needed
        4. Is professional and helpful
        5. Maintains conversation flow if this is part of an ongoing discussion
        
        Do not mention which agent was used or any technical details about the process.
        Just provide a clean answer to the user's question.
        """
        
        final_response = evaluator_llm.predict(formatting_prompt)
        return final_response
        
    except Exception as e:
        print(f"Error in final response generation: {e}")
        return raw_result  # Fallback to raw result

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query_data():
    """Enhanced endpoint that uses both LangChain and PandasAI agents with threading support"""
    try:
        # Check if agents are initialized
        if langchain_agent is None or pandasai_agent is None:
            return jsonify({
                "success": False,
                "error": "Agents not initialized. Please check server logs."
            }), 500
        
        # Get query and thread_id from request
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({
                "success": False,
                "error": "Missing 'query' field in request body"
            }), 400
        
        query = data['query'].strip()
        thread_id = data.get('thread_id', 'default_thread')
        
        if not query:
            return jsonify({
                "success": False,
                "error": "Query cannot be empty"
            }), 400
        
        print(f"Processing query for thread {thread_id}: {query}")
        
        # Step 1: Try LangChain agent first
        print(f"Trying LangChain agent for query: {query}")
        langchain_result, langchain_error = query_langchain(query, thread_id)
        
        if langchain_result and evaluate_result(query, langchain_result, "LangChain"):
            print("LangChain result is satisfactory")
            final_response = generate_final_response(query, langchain_result, None, "langchain", thread_id)
            
            # Update thread context
            update_thread_context(thread_id, query, final_response, "LangChain")
            
            return jsonify({
                "success": True,
                "result": final_response,
                "agent_used": "LangChain",
                "query": query,
                "thread_id": thread_id
            })
        
        # Step 2: If LangChain fails or is unsatisfactory, try PandasAI
        print("LangChain result unsatisfactory, trying PandasAI agent")
        pandasai_result, pandasai_error = query_pandasai(query, thread_id)
        
        if pandasai_result and evaluate_result(query, pandasai_result, "PandasAI"):
            print("PandasAI result is satisfactory")
            final_response = generate_final_response(query, None, pandasai_result, "pandasai", thread_id)
            
            # Update thread context
            update_thread_context(thread_id, query, final_response, "PandasAI")
            
            return jsonify({
                "success": True,
                "result": final_response,
                "agent_used": "PandasAI",
                "query": query,
                "thread_id": thread_id
            })
        
        # Step 3: If both fail, return the best available result or error
        if langchain_result:
            final_response = generate_final_response(query, langchain_result, None, "langchain", thread_id)
            update_thread_context(thread_id, query, final_response, "LangChain (fallback)")
            
            return jsonify({
                "success": True,
                "result": final_response,
                "agent_used": "LangChain (fallback)",
                "query": query,
                "thread_id": thread_id,
                "warning": "Result may not be optimal"
            })
        elif pandasai_result:
            final_response = generate_final_response(query, None, pandasai_result, "pandasai", thread_id)
            update_thread_context(thread_id, query, final_response, "PandasAI (fallback)")
            
            return jsonify({
                "success": True,
                "result": final_response,
                "agent_used": "PandasAI (fallback)",
                "query": query,
                "thread_id": thread_id,
                "warning": "Result may not be optimal"
            })
        else:
            return jsonify({
                "success": False,
                "error": "Both agents failed to process the query",
                "langchain_error": langchain_error,
                "pandasai_error": pandasai_error,
                "query": query,
                "thread_id": thread_id
            }), 500
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "query": data.get('query', '') if 'data' in locals() else '',
            "thread_id": data.get('thread_id', '') if 'data' in locals() else ''
        }), 500

@app.route('/threads', methods=['GET'])
def get_threads():
    """Get list of active threads"""
    with thread_lock:
        threads = []
        for thread_id, context in thread_conversations.items():
            threads.append({
                'thread_id': thread_id,
                'created_at': context['created_at'].isoformat(),
                'last_activity': context['last_activity'].isoformat(),
                'message_count': len(context['messages'])
            })
        return jsonify({
            "success": True,
            "threads": threads
        })

@app.route('/threads/<thread_id>', methods=['DELETE'])
def delete_thread(thread_id):
    """Delete a specific thread"""
    with thread_lock:
        if thread_id in thread_conversations:
            del thread_conversations[thread_id]
            return jsonify({
                "success": True,
                "message": f"Thread {thread_id} deleted"
            })
        else:
            return jsonify({
                "success": False,
                "error": "Thread not found"
            }), 404

def cleanup_old_threads():
    """Clean up threads older than 24 hours"""
    def cleanup():
        while True:
            try:
                current_time = datetime.now()
                with thread_lock:
                    threads_to_delete = []
                    for thread_id, context in thread_conversations.items():
                        # Delete threads inactive for more than 24 hours
                        if (current_time - context['last_activity']).total_seconds() > 86400:
                            threads_to_delete.append(thread_id)
                    
                    for thread_id in threads_to_delete:
                        del thread_conversations[thread_id]
                        print(f"Cleaned up inactive thread: {thread_id}")
                
                # Sleep for 1 hour before next cleanup
                time.sleep(3600)
            except Exception as e:
                print(f"Error in thread cleanup: {e}")
                time.sleep(3600)
    
    cleanup_thread = threading.Thread(target=cleanup, daemon=True)
    cleanup_thread.start()

if __name__ == '__main__':
    # Initialize both agents when server starts
    print("Initializing LangChain and PandasAI agents...")
    initialize_agents()
    
    # Start thread cleanup process
    cleanup_old_threads()
    
    print("Server starting on http://localhost:5000")
    print("Frontend will be available at http://localhost:5000")
    
    # Start Flask server with threading enabled
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
