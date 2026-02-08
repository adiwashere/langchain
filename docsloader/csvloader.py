from langchain_community.document_loaders import csv_loader

loader = csv_loader("sample.csv")
documents = loader.load() # for every row you will get a single document obj
print(documents) 