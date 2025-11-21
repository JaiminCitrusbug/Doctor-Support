import os
import json
from openai import OpenAI
from retriever import retrieve_similar_chunks

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def is_greeting(text):
    """Check if the user input is a greeting."""
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", 
                 "greetings", "howdy", "hi there", "hello there"]
    text_lower = text.lower().strip()
    # Remove punctuation for matching
    text_clean = ''.join(c for c in text_lower if c.isalnum() or c.isspace())
    words = text_clean.split()
    return any(greeting in text_lower for greeting in greetings) or (len(words) <= 3 and any(g in words for g in ["hi", "hello", "hey"]))

def has_question(text):
    """Check if the text contains a question or request (beyond just a greeting)."""
    question_words = ["what", "which", "when", "where", "who", "why", "how", "about", "tell", "give", "show", "explain", "describe", "assist", "help", "need", "want", "can you", "could you", "please"]
    text_lower = text.lower().strip()
    words = text_lower.split()
    
    # If it's just a greeting (1-2 words like "hi", "hello", "hey"), no question
    if len(words) <= 2:
        return False
    
    # Check if there are question words, question marks, or request words
    has_question_mark = "?" in text
    has_question_word = any(qw in text_lower for qw in question_words)
    
    # If it has more than 2 words (beyond just greeting), it likely has a request/question
    # Or if it has a question mark or question/request words, it's a question
    return has_question_mark or has_question_word or len(words) > 2

def get_medicine_list():
    """Get list of all medicines from the JSON file."""
    json_file = os.getenv("INPUT_JSON", "new_data.json")
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            products = json.load(f)
        # Extract product names
        medicine_names = [product.get("product_name", "") for product in products if product.get("product_name")]
        return medicine_names
    except Exception as e:
        print(f"⚠️  Error loading medicine list: {e}")
        return []

def get_greeting_message():
    """Generate the standard greeting message with medicine list."""
    medicine_list = get_medicine_list()
    medicine_list_str = ", ".join(medicine_list) if medicine_list else "No medicines available"
    source_url = "https://www.wockhardt.com/about-us/products/quality-generics/"
    return f"Hello! How are you? I am a POC prototype bot and for the basic demo purpose I have the mock data from the source: {source_url}\n\nYou may ask me any relevant stuffs."

def create_system_prompt(context_text, is_first_message=False):
    """Create the system prompt for the medical assistant."""
    medicine_list = get_medicine_list()
    medicine_list_str = ", ".join(medicine_list) if medicine_list else "No medicines available"
    source_url = "https://www.wockhardt.com/about-us/products/quality-generics/"
    greeting_message = f"Hello! How are you? I am a POC prototype bot and for the basic demo purpose I have the mock data for now as follows: {medicine_list_str}.\n\nWe have taken data from the source: {source_url}\n\nYou may ask me any relevant stuffs."
    
    return {
        "role": "system",
        "content": (
            "You are a reliable and context-aware medical assistant that supports doctors in emergencies. "
            "Use the entire conversation history and the provided context to give accurate, concise, and confident answers. "
            "Never guess or invent information—if uncertain, say so clearly.\n\n"

            "GREETING MESSAGE FORMAT (Use when no information is available):\n"
            f"- When you cannot find information in the context, respond with: \"{greeting_message}\"\n"
            "- This greeting message should be used when the user asks about something not in your knowledge base.\n"
            "- If the user asks a question and you have the answer in context, provide the answer directly - do NOT use the greeting message.\n"
            "- If the user asks a question and you DON'T have the answer, use the greeting message above.\n\n"

            "CRITICAL: Response Style - Be CONCISE and DIRECT:\n"
            "- Provide ONLY the essential information needed to answer the question.\n"
            "- Do NOT add unnecessary explanations, additional context, or verbose elaborations.\n"
            "- Do NOT list multiple medication classes or provide extensive background unless specifically asked.\n"
            "- Give confident, direct answers based on the context provided.\n"
            "- Example: For 'What should be given in case of cardiovascular?', answer with the specific medication from context (e.g., 'ATORITIC (Atorvastatin) is commonly used for cardiovascular conditions to manage cholesterol levels and reduce the risk of heart disease.') - NOT a long list of medication classes.\n\n"

            "CRITICAL: Query Type Detection and Response Strategy:\n"
            "- DISTINGUISH between two types of queries:\n"
            "  1. CONDITION/THERAPEUTIC CLASS QUERIES (e.g., 'cardiovascular', 'diabetes', 'antibiotics', 'heart condition'):\n"
            "     - If context contains relevant medicines for that condition/class, PROVIDE DIRECT INFORMATION about those medicines.\n"
            "     - DO NOT ask 'Did you mean' for condition queries - provide the answer directly.\n"
            "     - Example: User asks 'cardiovascular' and context has ATORITIC (CARDIAC class) → Respond: 'ATORITIC (Atorvastatin) is commonly used for cardiovascular conditions...'\n"
            "  2. PRODUCT/DRUG NAME QUERIES with potential misspellings (e.g., 'antrovast', 'kiprteb', 'ciprteb'):\n"
            "     - These are likely misspellings of product/drug names.\n"
            "     - Use 'Did you mean' format ONLY for these misspelling cases.\n\n"

            "CRITICAL: Enhanced Fuzzy Matching and Correction Behavior:\n"
            "- ALWAYS analyze the user's query to determine if it's a condition query or a product name query.\n"
            "- For CONDITION queries (cardiovascular, diabetes, antibiotics, etc.):\n"
            "  - Check if context contains medicines in matching therapeutic classes (CARDIAC, DIABETES, ANTIBIOTICS, etc.).\n"
            "  - If found, provide direct information about those medicines.\n"
            "  - Map conditions: 'cardiovascular'/'heart' → CARDIAC, 'diabetes' → DIABETES, 'antibiotic' → ANTIBIOTICS.\n"
            "- For PRODUCT/DRUG NAME queries with potential misspellings:\n"
            "  - Consider phonetic similarity (e.g., 'antrovast' sounds like 'atorvastatin', 'antrovastic' → 'atorvastatin').\n"
            "  - Consider partial matches (e.g., 'antrovast' contains 'atorvast' which is close to 'atorvastatin').\n"
            "  - Consider common misspellings: letter swaps, dropped letters, added letters, similar-sounding words.\n"
            "  - Look for product names (e.g., 'ATORITIC') AND active ingredients (e.g., 'Atorvastatin') in the context.\n"
            "  - If you identify a reasonable match for a misspelling, use the 'Did you mean' format.\n\n"

            "Correction Message Format (MANDATORY - ONLY for misspellings):\n"
            "- Use 'Did you mean' ONLY when the user query appears to be a misspelling of a product/drug name.\n"
            "- When suggesting a correction for a misspelling, ALWAYS use this EXACT format: \"Did you mean '[suggested_match]'?\"\n"
            "- Example: User says \"antrovast\" (misspelling) → Respond: \"Did you mean 'ATORITIC'?\" or \"Did you mean 'Atorvastatin'?\"\n"
            "- Example: User says \"kiprteb\" (misspelling) → Respond: \"Did you mean 'CIPROTAB'?\"\n"
            "- Do NOT use 'Did you mean' for condition queries - provide direct answers instead.\n"
            "- Do NOT include any other statements before or after. Use ONLY the format: \"Did you mean '[suggested_match]'?\"\n\n"

            "Matching Strategy:\n"
            "1. First, determine if the query is about a condition/therapeutic class or a product/drug name.\n"
            "2. For CONDITION queries:\n"
            "   a. Check if context contains medicines in matching therapeutic classes.\n"
            "   b. If found, provide direct information about those medicines.\n"
            "   c. If not found, use the greeting message format provided above.\n"
            "3. For PRODUCT/DRUG NAME queries:\n"
            "   a. Check if the user query exactly matches any product name, brand name, or active ingredient in the context.\n"
            "   b. If exact match, provide information directly.\n"
            "   c. If no exact match, look for phonetic similarities (how it sounds).\n"
            "   d. If no phonetic match, look for partial string matches (substrings).\n"
            "   e. If you find a reasonable match for a misspelling, use 'Did you mean' format.\n"
            "   f. If no match found, use the greeting message format provided above.\n"
            "4. IMPORTANT: When the user asks a question (even in a greeting like 'hello! ciprotab?'), if you have the answer in context, provide it directly. Only use the greeting message when you truly don't have the information.\n\n"

            "On user confirmation (yes, yeah, correct, exactly, right, that's it, etc.):\n"
            "- Immediately proceed using the last suggested match from the assistant's clarification. "
            "- Provide complete information about that entity from the context. "
            "- Never respond with 'I couldn't find that' after a confirmed match unless the term truly does not exist in context.\n\n"

            "If the user follows up with a related or generalized term (e.g., 'antibiotic', 'cardiac drugs', 'painkillers'), "
            "interpret it contextually. Dynamically identify related entities within the same therapeutic class or purpose. "
            "Provide meaningful information or examples from the data that align with that category.\n\n"

            "Dynamic context memory rules:\n"
            "- Remember the last clarification and the confirmed term in the conversation. "
            "- Use that stored term when the user later refers with pronouns ('it', 'that', 'this') or follow-ups ('What about it?'). "
            "- Keep the flow natural and consistent—never lose or reset confirmed context within the same session.\n\n"

            "Matching and reasoning priorities:\n"
            "1. Exact match → respond directly with information.\n"
            "2. Fuzzy / misspelled match → suggest correction using the format: \"Did you mean '[match]'?\"\n"
            "3. Related term match → provide connected examples or explain related items.\n"
            "4. No match → politely say: \"I couldn't find that in my current data. Could you please rephrase or clarify what you meant?\"\n\n"

            "Maintain a professional, empathetic tone suitable for medical professionals. "
            "Be concise, factual, and medically safe in all responses.\n\n"

            f"Context from knowledge base:\n{context_text}\n\n"
            "Remember: Be PROACTIVE in suggesting corrections. Even if the match isn't perfect, if it's reasonable, suggest it! "
            "And always provide CONCISE, DIRECT answers without unnecessary elaboration."
        ),
    }

def determine_retrieval_query(user_query, history):
    """
    Use LLM to dynamically determine the best query for retrieval based on conversation context.
    The LLM analyzes the conversation and determines what should be searched in the vector database.
    """
    # Convert history to chat format
    chat_history = [{"role": m["role"], "content": m["content"]} for m in history]
    
    # Create a prompt for the LLM to determine the retrieval query
    query_determination_prompt = {
        "role": "system",
        "content": """You are a query analyzer for a medical information retrieval system. 
Your task is to analyze the conversation and determine the BEST search query to use for retrieving relevant information from the knowledge base.

CRITICAL RULES:
1. If the user is confirming a correction (e.g., responding "yes" to "Did you mean CIPROTAB?"), extract the CORRECTED TERM (e.g., "CIPROTAB") from the conversation history.
2. If the user is asking about a medical condition or therapeutic class:
   - Map conditions to therapeutic classes: "cardiovascular" → "CARDIAC", "heart" → "CARDIAC", "diabetes" → "DIABETES", "antibiotic" → "ANTIBIOTICS"
   - Use the therapeutic class term for retrieval
3. If the user is asking a new question, analyze it for potential misspellings and try to normalize it:
   - If you recognize a misspelling (e.g., "antrovast" → likely "atorvastatin"), use the corrected version
   - If you recognize a partial match (e.g., "antrovast" contains "atorvast"), expand to full term
   - Consider phonetic similarities and common drug/product name patterns
4. If the user is referring to something mentioned earlier (using pronouns like "it", "that", "this"), extract the actual entity name from the conversation history.
5. Always return ONLY the search query term(s) - no explanations, no questions, just the query string.
6. Focus on extracting the specific product name, drug name, medical condition, or therapeutic class that should be searched.

Examples:
- User: "I need details on Ciprteb" → Return: "CIPROTAB" (corrected spelling)
- User: "antrovast" → Return: "atorvastatin" or "ATORITIC" (normalized/corrected)
- User: "antrovastic" → Return: "atorvastatin" (corrected)
- User: "Tell me about antibiotics" → Return: "ANTIBIOTICS"
- User: "cardiovascular" or "cardiovascular condition" → Return: "CARDIAC" or "cardiovascular CARDIAC"
- User: "What about it?" (after discussing CIPROTAB) → Return: "CIPROTAB"
- User: "I need details on Ciprotab" → Return: "CIPROTAB"

IMPORTANT: Be proactive in correcting obvious misspellings. If you recognize a drug/product name even with typos, use the corrected version.
IMPORTANT: Map medical conditions to their therapeutic classes for better retrieval.

Return ONLY the search query, nothing else."""
    }
    
    # Build messages for query determination
    messages = [query_determination_prompt] + chat_history + [
        {"role": "user", "content": f"Given the conversation above, what should be the search query for the current user message: '{user_query}'?\n\nReturn ONLY the search query:"}
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.1,  # Low temperature for consistent extraction
            max_tokens=50,  # Short response - just the query
        )
        
        retrieval_query = response.choices[0].message.content.strip()
        
        # Clean up the response - remove quotes if present
        retrieval_query = retrieval_query.strip('"').strip("'").strip()
        
        # If LLM returns something like "The search query should be: X", extract X
        if ":" in retrieval_query and len(retrieval_query.split(":")) > 1:
            retrieval_query = retrieval_query.split(":")[-1].strip()
        
        # Fallback: if LLM returns something unexpected, use original query
        if not retrieval_query or len(retrieval_query) < 2:
            retrieval_query = user_query
        
        print(f"🤖 LLM determined retrieval query: '{retrieval_query}' (from user: '{user_query}')")
        return retrieval_query
        
    except Exception as e:
        print(f"⚠️  Error in LLM query determination: {e}. Using original query.")
        return user_query

def generate_answer(user_query, history):
    """Generate context-aware answer using chat history and RAG."""
    # Check if this is a first-time greeting
    # History includes the current user message, so check if it's the only message and is a greeting
    is_first_message = len(history) == 1 and history[0].get("role") == "user"
    
    if is_first_message and is_greeting(user_query):
        # Check if the greeting also contains a question or request
        if has_question(user_query):
            # User asked a question/request along with greeting - let LLM handle everything
            retrieval_query = determine_retrieval_query(user_query, history)
            retrieved_chunks = retrieve_similar_chunks(retrieval_query, top_k=5)
            context_text = "\n\n".join([chunk["text"] for chunk in retrieved_chunks])
            
            # Let LLM decide everything - it knows the greeting message format from system prompt
            chat_history = [{"role": m["role"], "content": m["content"]} for m in history]
            system_prompt = create_system_prompt(context_text, is_first_message=True)
            messages = [system_prompt] + chat_history + [{"role": "user", "content": user_query}]

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.2,
            )

            answer = response.choices[0].message.content.strip()
            # Return LLM's response directly - it will decide what to say
            return answer
        
        else:
            # Just a simple greeting (1-2 words like "hi", "hello") - show greeting + medicine list
            return get_greeting_message()
    
    # Use LLM to dynamically determine the best query for retrieval
    retrieval_query = determine_retrieval_query(user_query, history)
    
    # Retrieve more chunks to have better candidates for fuzzy matching
    # This helps when user misspells - we get more potential matches to analyze
    retrieved_chunks = retrieve_similar_chunks(retrieval_query, top_k=5)
    context_text = "\n\n".join([chunk["text"] for chunk in retrieved_chunks])

    # Always let LLM process the query - it will decide if it can answer
    # Convert Streamlit history to OpenAI chat format
    chat_history = [{"role": m["role"], "content": m["content"]} for m in history]
    system_prompt = create_system_prompt(context_text)

    messages = [system_prompt] + chat_history + [{"role": "user", "content": user_query}]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2,
    )

    answer = response.choices[0].message.content.strip()
    # Return LLM's response directly - it will decide when to use greeting message
    return answer

