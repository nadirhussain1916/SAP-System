from flask import Flask, request, jsonify, render_template
import psycopg2
import json
from openai import OpenAI
from datetime import datetime
import pandas as pd
import os
from typing import Dict, List, Any
import logging
from flask_cors import CORS
import re
from dotenv import load_dotenv
load_dotenv()
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("Please set OPENAI_API_KEY environment variable")

client = OpenAI(api_key=OPENAI_API_KEY)

# Database configuration
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_URI = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
DB_CONFIG = {
    "dbname": DB_NAME,
    "user": DB_USER,
    "password": DB_PASSWORD,
    "host": DB_HOST,
    "port": int(DB_PORT)
}

# Conversation context storage (in production, use Redis or database)
conversation_contexts = {}

# Enhanced System prompt for query generation
QUERY_GENERATION_PROMPT = """You are a SQL query generator for SAP system analysis. Your task is to generate ONLY SQL queries based on user questions.

DATABASE SCHEMA - Table: logs_fixed
Columns:
- started (time without time zone): When the process started
- server (text): Server name/identifier  
- program (text): SAP program name
- work_process (integer): Work process ID
- user (text): SAP user
- response_time_in_ms (integer): Total response time in milliseconds
- time_in_work_process_ms (integer): Time spent in work process
- wait_time_ms (integer): Wait time in milliseconds
- cpu_time_ms (integer): CPU time consumed
- db_request_time_ms (integer): Database request time
- vmc_elapsed_ms (integer): VMC elapsed time
- enqueue_time_ms (integer): Enqueue time
- enqueues (integer): Number of enqueues
- program_load_time_ms (integer): Program load time
- screen_load_time_ms (integer): Screen load time
- load_time (integer): Total load time
- roll_ins (integer): Number of roll-ins
- roll_outs (integer): Number of roll-outs
- roll_in_time_ms (integer): Roll-in time
- roll_out_time_ms (integer): Roll-out time
- roll_wait_time_ms (integer): Roll wait time
- number_of_roundtrips (integer): Database round trips
- direct_read_requests (integer): Direct read requests count
- direct_read_database_rows (integer): Direct read database rows
- direct_read_buffer_requests (integer): Direct read buffer requests
- direct_read_request_time_ms (integer): Direct read request time
- direct_read_average_time_rows_ms (integer): Average direct read time per row
- sequential_read_request (integer): Sequential read requests
- sequential_read_database_rows (integer): Sequential read database rows
- sequential_read_buffer_request (integer): Sequential read buffer requests
- read_pysical_database_calls (integer): Physical database read calls
- sequential_read_request_time_ms (integer): Sequential read request time
- sequential_read_average_time_row_ms (integer): Average sequential read time per row
- update_requests (integer): Update requests count
- update_database_rows (integer): Updated database rows
- update_pysical_database_calls (integer): Physical database update calls
- update_request_time_ms (integer): Update request time
- update__average_time_rows (integer): Average update time per row
- delete_requests (integer): Delete requests count
- delete_database_rows (integer): Deleted database rows
- delete_physical_database_calls (integer): Physical database delete calls
- delete_request_time_ms (integer): Delete request time
- delete__average_time_row_ms (integer): Average delete time per row
- insert_requests (integer): Insert requests count
- insert_database_rows (integer): Inserted database rows
- insert_pysical_database_calls (integer): Physical database insert calls
- insert_request_times_ms (integer): Insert request time
- insert__time_row_ms (integer): Average insert time per row
- work_process_number (integer): Work process number
- maximum_memory_roll_kb (integer): Maximum memory roll in KB
- total_allocated_page_memory_kb (integer): Total allocated page memory in KB
- maximum_extended_memory_in_task_kb (integer): Maximum extended memory in task
- maximum_extended_memory_in_step_kb (integer): Maximum extended memory in step
- extended_memory_in_use_kb (integer): Extended memory in use
- privilege_memory_in_use_kb (integer): Privilege memory in use
- work_progress_in_privilege_mode (boolean): Whether work process is in privilege mode

RULES:
1. Generate ONLY SQL queries, no explanations
2. Use proper PostgreSQL syntax
3. Always use "logs_fixed" as the table name
4. For user questions about names/lists, use DISTINCT to avoid duplicates
5. Limit results to reasonable numbers (use LIMIT when appropriate)
6. Use double quotes around "user" column since it's a PostgreSQL reserved word

EXAMPLE QUERIES:
- For "show me server names": SELECT DISTINCT server FROM logs_fixed;
- For "list all users": SELECT DISTINCT "user" FROM logs_fixed;
- For "what programs are running": SELECT DISTINCT program FROM logs_fixed;
- For "top 10 slowest responses": SELECT server, program, "user", response_time_in_ms FROM logs_fixed ORDER BY response_time_in_ms DESC LIMIT 10;

Generate SQL query for the user's question:"""

# Response generation prompt
RESPONSE_GENERATION_PROMPT = """You are an advanced SAP System Assistant. You help users analyze SAP system logs and performance metrics by interpreting database query results.

IMPORTANT GUIDELINES:
1. NEVER show SQL queries in your responses
2. Provide direct, clear answers based on the actual data results provided
3. Be conversational and helpful - interpret the data meaningfully
4. Present the actual data from the database results
5. Provide insights, trends, and actionable recommendations when relevant
6. Use natural language to explain findings
7. If data shows specific names/values, list them exactly as they appear in the database
8. Add context and business meaning to numbers
9. Be concise but comprehensive in explanations
10. Focus entirely on what the actual data shows

When presenting lists of items (servers, users, programs), show the actual names from the database, not generic placeholders.

Provide a helpful, business-focused response based on the query results."""

class SAPAssistant:
    def __init__(self):
        self.db_config = DB_CONFIG
        
    def get_db_connection(self):
        """Establish database connection"""
        try:
            conn = psycopg2.connect(**self.db_config)
            return conn
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise
    
    def execute_query(self, query: str) -> Dict[str, Any]:
        """Execute SQL query and return results"""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute(query)
            
            # Fetch results
            if cursor.description:
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                
                # Convert to list of dictionaries for JSON serialization
                result_data = []
                for row in rows:
                    row_dict = {}
                    for i, value in enumerate(row):
                        if isinstance(value, datetime):
                            row_dict[columns[i]] = value.isoformat()
                        else:
                            row_dict[columns[i]] = value
                    result_data.append(row_dict)
                
                conn.close()
                return {
                    "success": True,
                    "data": result_data,
                    "columns": columns,
                    "row_count": len(result_data)
                }
            else:
                conn.commit()
                conn.close()
                return {
                    "success": True,
                    "message": "Query executed successfully",
                    "affected_rows": cursor.rowcount
                }
                
        except Exception as e:
            logger.error(f"Query execution error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_table_summary(self) -> Dict[str, Any]:
        """Get basic table statistics"""
        query = """
        SELECT 
            COUNT(*) as total_records,
            MIN(started) as earliest_record,
            MAX(started) as latest_record,
            COUNT(DISTINCT server) as unique_servers,
            COUNT(DISTINCT program) as unique_programs,
            COUNT(DISTINCT "user") as unique_users,
            AVG(response_time_in_ms) as avg_response_time,
            MAX(response_time_in_ms) as max_response_time,
            AVG(cpu_time_ms) as avg_cpu_time,
            AVG(db_request_time_ms) as avg_db_time
        FROM logs_fixed;
        """
        return self.execute_query(query)

def get_conversation_context(thread_id: str) -> List[Dict]:
    """Get conversation context for thread"""
    return conversation_contexts.get(thread_id, [])

def update_conversation_context(thread_id: str, role: str, content: str):
    """Update conversation context, keeping last 10 messages"""
    if thread_id not in conversation_contexts:
        conversation_contexts[thread_id] = []
    
    conversation_contexts[thread_id].append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    })
    
    # Keep only last 10 messages (5 exchanges)
    if len(conversation_contexts[thread_id]) > 10:
        conversation_contexts[thread_id] = conversation_contexts[thread_id][-10:]

def generate_sql_query(user_message: str, context: List[Dict]) -> str:
    """Generate SQL query based on user message"""
    try:
        messages = [{"role": "system", "content": QUERY_GENERATION_PROMPT}]
        
        # Add recent context (last 4 messages only to keep focus)
        recent_context = context[-4:] if len(context) > 4 else context
        for msg in recent_context:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        messages.append({
            "role": "user", 
            "content": user_message
        })
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.1,
            max_tokens=500
        )
        
        sql_query = response.choices[0].message.content.strip()
        
        # Clean up the response to extract only the SQL query
        if "```sql" in sql_query:
            sql_query = sql_query.split("```sql")[1].split("```")[0].strip()
        elif "```" in sql_query:
            sql_query = sql_query.split("```")[1].strip()
        
        # Remove any additional text, keep only the SQL
        lines = sql_query.split('\n')
        sql_lines = []
        for line in lines:
            line = line.strip()
            if line and (line.upper().startswith(('SELECT', 'WITH', 'FROM', 'WHERE', 'GROUP', 'ORDER', 'HAVING', 'LIMIT', 'DISTINCT')) or 
                        any(keyword in line.upper() for keyword in ['FROM', 'WHERE', 'AND', 'OR', 'JOIN', 'ON', 'GROUP BY', 'ORDER BY', 'HAVING', 'LIMIT'])):
                sql_lines.append(line)
        
        final_query = ' '.join(sql_lines) if sql_lines else sql_query
        
        logger.info(f"Generated SQL query: {final_query}")
        return final_query
        
    except Exception as e:
        logger.error(f"SQL generation error: {e}")
        return ""

def generate_response_from_data(user_message: str, context: List[Dict], query_results: List[Dict],sql_query: str,  ) -> str:
    """Generate response based on actual database results"""
    try:
        messages = [{"role": "system", "content": RESPONSE_GENERATION_PROMPT}]
        
        # Add conversation context
        for msg in context:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Format the actual data results for AI interpretation
        results_text = f"User Question: {user_message}\n\nGenerated SQL: {sql_query}\n\nDatabase Query Results:\n\n"
        
        for i, query_result in enumerate(query_results):
            if query_result["result"]["success"]:
                data = query_result["result"]["data"]
                results_text += f"Query {i+1} Results ({len(data)} rows):\n"
                
                if data:
                    # Format data in a readable way
                    if len(data) <= 50:  # Show all data if reasonable amount
                        results_text += json.dumps(data, indent=2) + "\n\n"
                    else:  # Show sample + summary for large datasets
                        results_text += f"Sample of first 10 records:\n{json.dumps(data[:10], indent=2)}\n"
                        results_text += f"Total records: {len(data)}\n\n"
                else:
                    results_text += "No data found.\n\n"
            else:
                results_text += f"Query {i+1} Error: {query_result['result']['error']}\n\n"
        
        results_text += "\nPlease provide a clear, helpful response based on this actual data. Do not show SQL queries. Focus on interpreting the results and providing insights. Present the actual data from the database results."
        logger.info(f"Results text for response generation: {results_text}")
        messages.append({
            "role": "user", 
            "content": results_text
        })
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.1,
            max_tokens=2000
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Response generation error: {e}")
        return f"I apologize, but I encountered an error while processing your request: {str(e)}"

def generate_fallback_response(user_message: str, context: List[Dict]) -> str:
    """Generate fallback response when no query can be generated"""
    try:
        messages = [
            {"role": "system", "content": "You are a helpful SAP assistant. The user asked a question but no database query could be generated. Provide a helpful response explaining what information you can help with regarding SAP system analysis, or ask for clarification."},
        ]
        
        for msg in context:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        messages.append({
            "role": "user", 
            "content": user_message
        })
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.3,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Fallback response error: {e}")
        return "I can help you analyze your SAP system data. You can ask me about servers, users, programs, performance metrics, response times, and more. What would you like to know?"

# Initialize assistant
assistant = SAPAssistant()

@app.route('/assistant', methods=['POST'])
def sap_assistant_endpoint():
    """Main endpoint for SAP assistant interactions"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        user_message = data.get('message', '').strip()
        thread_id = data.get('thread_id', f'thread_{datetime.now().timestamp()}')
        
        if not user_message:
            return jsonify({"error": "Message is required"}), 400
        
        # Get conversation context
        context = get_conversation_context(thread_id)
        
        # Step 1: Generate SQL query based on user message
        sql_query = generate_sql_query(user_message, context)
        
        if sql_query and sql_query.strip():
            # Step 2: Execute the generated query
            logger.info(f"Executing query: {sql_query}")
            query_result = assistant.execute_query(sql_query)
            logger.info(f"Query result: {query_result}")
            if query_result["success"]:
                # Step 3: Generate response based on actual data
                query_results = [{"query": sql_query, "result": query_result}]
                final_response = generate_response_from_data(user_message, context, query_results,sql_query)
            else:
                # Query failed, provide error context
                final_response = f"I encountered an issue retrieving the data: {query_result.get('error', 'Unknown error')}. Please try rephrasing your question or ask me something else about your SAP system."
        else:
            # No query generated, provide fallback response
            final_response = generate_fallback_response(user_message, context)
        
        # Update conversation context
        update_conversation_context(thread_id, "user", user_message)
        update_conversation_context(thread_id, "assistant", final_response)
        
        # Prepare response
        response_data = {
            "thread_id": thread_id,
            "response": final_response,
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Endpoint error: {e}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        conn = assistant.get_db_connection()
        conn.close()
        
        return jsonify({
            "status": "healthy",
            "database": "connected",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/table-summary', methods=['GET'])
def get_table_summary():
    """Get basic table statistics"""
    try:
        summary = assistant.get_table_summary()
        return jsonify(summary)
    except Exception as e:
        return jsonify({
            "error": "Failed to get table summary",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    # Set your OpenAI API key as environment variable before running
    # export OPENAI_API_KEY="your-api-key-here"
    app.run(debug=True, host='0.0.0.0', port=5000)