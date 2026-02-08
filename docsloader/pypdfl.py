from langchain_community.document_loaders import pypdf_loader

loader = pypdf_loader("sample.pdf")
documents = loader.load()
print(documents) # you will get same no of docs in list as no of pages in pdf

print(len(documents))

print(documents[0].page_content)

print(documents[0].metadata)
