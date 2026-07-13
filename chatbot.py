# =====================================================================
# PROJECT 1: RULE-BASED AI CHATBOT (DETERMINISTIC LOGIC ENGINE)
# Powered by DecodeLabs [Batch: 2026]
# =====================================================================

# 1. KNOWLEDGE BASE: Dictionary with 5+ unique intents (O(1) Constant Time Lookup)
responses = {
    "hello": "Hi there! I am your deterministic rule-based AI assistant. How can I help you today?",
    "hi": "Hello! Hope you are doing great. What's on your mind?",
    "help": "I can assist you with predefined commands. Try asking about my 'name', 'status', or 'architecture'.",
    "name": "I am a White-Box Guardrail Chatbot, built as a control layer for AI systems.",
    "status": "All systems operational. Memory allocation: Optimal. Hallucination risk: 0%.",
    "architecture": "My architecture is based on the IPO Model (Input-Process-Output) using a Hash Map lookup.",
}

print("==========================================================")
print("🤖 DETECTIVE CHATBOT INITIALIZED (Type 'exit' to shutdown)")
print("==========================================================")

# 2. INPUT LOOP: The Infinite Cycle (Keeps the organism alive)
while True:
    # 3. INPUT & SANITIZATION (Cleaning raw feed: removing spaces & forcing lowercase)
    raw_input = input("\nYou: ")
    clean_input = raw_input.lower().strip()

    # 4. EXIT STRATEGEY (The Kill Command)
    if clean_input == "exit" or clean_input == "bye":
        print("Bot: Terminating process. Goodbye! 👋")
        break  # Loop se bahar nikalne k liye

    # 5. PROCESS & OUTPUT: Atomic Lookup + Fallback
    # .get() handles matching and fallback in a single atomic operation
    reply = responses.get(clean_input, "Bot: I am sorry, I do not understand that command. Please try again.")
    
    print(reply)