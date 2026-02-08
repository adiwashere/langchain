from langchain_community.document_loaders import WebPageLoader

loader = WebPageLoader("https://www.google.com/")
documents = loader.load()
print(documents)