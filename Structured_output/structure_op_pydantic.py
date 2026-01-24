## we get review -> put in llm -> get summary, get sentiment as output

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal, Optional, List
from pydantic import BaseModel, Field, EmailStr


load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

#schema for structured output
class review(BaseModel):
    key_theme :list[str]=Field(description="The main theme or topic of the review")
    summmary: str=Field(description="A brief summary of the review") # by using annotated we can guide the model
    sentiment: Literal["pos", "neg"]=Field(description="The overall sentiment of the review, e.g., positive, negative, neutral")
    pros : Optional[list[str]]=Field(default=None,description="List of positive aspects mentioned in the review")      
    cons : Optional[list[str]]=Field(default=None,description="List of negative aspects mentioned in the review")  

structured_model = model.with_structured_output(review)

#write a review 
result = structured_model.invoke("I recently purchased the XYZ smartphone, and I must say, it has exceeded my expectations in many ways. The build quality is excellent, with a sleek design that feels premium in hand. The display is vibrant and sharp, making media consumption a delight. Performance-wise, the phone handles multitasking and gaming with ease, thanks to its powerful processor and ample RAM.")

print(result )
print(result.summmary)
print(result.sentiment)
print(result.pros)
print(result.cons)
