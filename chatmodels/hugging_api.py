
# from dotenv import load_dotenv
# from huggingface_hub import InferenceClient

# # Silence warnings
# os.environ["TRANSFORMERS_VERBOSITY"] = "error"
# warnings.filterwarnings("ignore", category=UserWarning)

# load_dotenv()
# token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# # Use a model currently supported on the FREE Inference Providers tier
# # Gemma 2 9B is very stable right now.
# client = InferenceClient(api_key=token)

# try:
#     print("Sending request via InferenceClient...")
    
#     # This uses the OpenAI-compatible format which is the new standard
#     response = client.chat.completions.create(
#         model="google/gemma-2-9b-it", 
#         messages=[{"role": "user", "content": "Hello! Can you hear me?"}],
#         max_tokens=100,
#         temperature=0.5
#     )
    
#     print(f"\nAI: {response.choices[0].message.content}")

# except Exception as e:
#     print(f"\nError: {e}")

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()

llm= HuggingFaceEndpoint(
    repo_id="google/gemma-2-9b-it",
    task="text-generation",
    temperature=0
)
model = ChatHuggingFace(llm=llm)
result = model.invoke("Write a 5 linepoem on cricket")
print(result.content)