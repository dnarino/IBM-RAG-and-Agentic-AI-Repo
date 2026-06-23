from langchain_classic.memory import ConversationSummaryMemory
from langchain_classic.chains import ConversationChain

# Import our modularized client!
from core.watson_client import get_watsonx_chat

# 1. Setup the LLM
model_id='meta-llama/llama-3-3-70b-instruct'
parameters = {
    "max_new_tokens": 256,
    "temperature": 0.2,
    "top_k": 2,
    "top_p": 0.1
}
llama_LLM = get_watsonx_chat(model_id=model_id, params=parameters)

# 2. Setup the Summary Memory
# We MUST pass the LLM into the memory object because it needs a brain to write the summaries!
print("Initializing Summary Memory (This requires an LLM!)...")
memory = ConversationSummaryMemory(llm=llama_LLM)

# 3. Create a massive fake call-center conversation (20 interactions!)
call_center_script = [
    {"input": "Hello, my internet has been down for 3 days!", "output": "I am so sorry to hear that. Let me look up your account. Can I have your name?"},
    {"input": "My name is David.", "output": "Thank you, David. I see your account. Are the lights on your router blinking?"},
    {"input": "Yes, the DSL light is blinking red.", "output": "A red blinking DSL light means the router cannot connect to the street line. Have you tried rebooting it?"},
    {"input": "Yes, I unplugged it for 10 minutes and plugged it back in.", "output": "Okay. Since you already rebooted it, I will need to run a line test from my end. Please hold for 2 minutes."},
    {"input": "Okay, I will hold.", "output": "Thank you for holding. The line test failed. It looks like the cable outside your house was damaged in the recent storm."},
    {"input": "Are you kidding me? How long will it take to fix?", "output": "I can dispatch a technician tomorrow morning between 8 AM and 12 PM. Will you be home?"},
    {"input": "Yes, I will be home. But I work from home and this is costing me money!", "output": "I completely understand the frustration. I can apply a $50 credit to your next bill for the inconvenience."},
    {"input": "A $50 credit? That's it? I want a free month of service!", "output": "I am not authorized to give a free month, but I can escalate this to my manager if you'd like."},
    {"input": "Yes, escalate it. I've been a loyal customer for 5 years.", "output": "I have submitted the escalation ticket. My manager will call you within 24 hours. The technician is still scheduled for tomorrow morning."},
    {"input": "Fine. Just make sure the technician actually shows up.", "output": "I have added a priority flag to the dispatch. Is there anything else I can help you with today?"}
]

# 4. Inject the massive script into the memory
print("\nInjecting 20 messages into the memory buffer... The LLM is actively summarizing this in the background!")
for interaction in call_center_script:
    # We save each back-and-forth into the memory bank
    memory.save_context({"input": interaction["input"]}, {"output": interaction["output"]})

# 5. Let's see what the memory actually stored!
print("\n" + "="*50)
print("🧠 FINAL MEMORY BUFFER (Condensed Summary):")
print("="*50)
print(memory.buffer)
print("="*50 + "\n")

# 6. Test the memory with a live question!
print("Creating the Conversation Chain to test the memory...")
conversation = ConversationChain(
    llm=llama_LLM,
    memory=memory,
    verbose=True
)

print("Asking the AI a question about the past conversation...")
response = conversation.invoke(input="Wait, what time is the technician coming tomorrow again, and what compensation did you offer me?")

print("\n🤖 AI Response:")
print(response["response"])
