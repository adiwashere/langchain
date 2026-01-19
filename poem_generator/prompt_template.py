from langchain_core.prompts import PromptTemplate

# template dictionary for dynamic prompt
template = PromptTemplate(
    input_variables=["user_input", "style_input", "length_input"],
    template="""
You are a creative poet.

Write a {length_input} poem about **{user_input}** in a **{style_input}** style.
Guidelines:
- Use vivid imagery and expressive language
- Maintain a consistent tone throughout
- Make the poem emotionally engaging
- Do NOT explain the poem
- Output ONLY the poem text
- Make it user friendly and easy to read

Poem:
""" 
)

template.save("template.json")
