import os
from flask import Flask, request, jsonify
from agent_rag import create_embeddings_from_pdfs, chat_with_agent, setup_database
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

@app.route('/webhook/create_source_embeddings', methods=['GET'])
def create_embeddings():
    """
    Endpoint to create embeddings from PDF files in the shared directory
    This replicates the n8n webhook for creating embeddings
    """
    try:
        create_embeddings_from_pdfs()
        return jsonify({"response": "All embeddings are created successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/webhook/invoke_n8n_agent', methods=['POST'])
def invoke_agent():
    """
    Endpoint to invoke the agent with a chat input
    This replicates the n8n webhook for chatting with the agent
    """
    try:
        data = request.json
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        chat_input = data.get('chatInput')
        session_id = data.get('sessionId')
        
        if not chat_input:
            return jsonify({"error": "No chat input provided"}), 400
        
        result = chat_with_agent(chat_input, session_id)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/setup_db', methods=['GET'])
def setup_db():
    """
    Endpoint to set up the database
    """
    try:
        success = setup_database()
        if success:
            return jsonify({"response": "Database setup completed successfully"})
        else:
            return jsonify({"error": "Database setup failed"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Set up the database before starting the server
    setup_database()
    
    # Run the API server
    port = int(os.environ.get('PORT', 5678))  # Match n8n port in the original implementation
    app.run(host='0.0.0.0', port=port, debug=True) 