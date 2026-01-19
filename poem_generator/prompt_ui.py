from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import load_prompt



load_dotenv()

# Initialize the HuggingFace LLM and Chat Model
llm= HuggingFaceEndpoint(
    repo_id="google/gemma-2-9b-it",
    task="text-generation",
    temperature=0
)
model = ChatHuggingFace(llm=llm)



# st.header("Research Tool")
# # this is static propmt
# user_input = st.text_input("Enter your query here")

# if st.button("Summary"):
#     result = model.invoke(user_input)
    
#     st.write(result.content)

# dynamic prompt
st.header("Poem Generator")

# user inputs for dynamic prompt
user_input = st.selectbox(
    "Choose a topic for your poem:",
    ["Nature", "Love", "Friendship", "Adventure", "Customs"]
)

if user_input == "Customs":
    Customs_input = st.text_input("Enter your custom topic here:")
    user_input = Customs_input


style_input = st.selectbox(
    "Choose a style for your poem:",
    ["Romantic", "Modern", "Classic", "Humorous", "Customs"]
)
if style_input == "Customs":
    Customs_style = st.text_input("Enter your custom style here:")
    style_input = Customs_style

length_input = st.selectbox(
    "Choose the length of your poem:",  
    ["Short (4-6 lines)", "Medium (7-10 lines)", "Long (11-15 lines)", "Customs"]  
)
if length_input == "Customs":
    Customs_length = st.text_input("Enter your custom length here:")
    length_input = Customs_length

# template for dynamic prompt

template = load_prompt("template.json")

# formatting the prompt with user inputs
prompt = template.format(
    user_input=user_input,  
    style_input=style_input,
    length_input=length_input
)

# Button to generate poem and invoking the model
if st.button("Generate Poem"):
    result = model.invoke(prompt)
    st.write(result.content)