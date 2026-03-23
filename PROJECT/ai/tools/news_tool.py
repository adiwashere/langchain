from langchain_community.tools import DuckDuckGoSearchRun
from ai.model import model
from ai.prompts import location_extract_prompt
from langchain_core.output_parsers import StrOutputParser

search = DuckDuckGoSearchRun()

location_chain = location_extract_prompt | model | StrOutputParser()

def news_tool(user_input,session):

    location = location_chain.invoke({"input": user_input}).strip()

    query = f"Latest news today in {location}"

    results = search.run(query)

    if not results:
        return f"Couldn't fetch news for {location}"

    summary = model.invoke(
        f"Summarize this news in bullet points:\n{results}"
    )

    return summary.content