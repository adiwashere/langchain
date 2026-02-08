from langchain_community.document_loaders import DirectoryLoader, pypdf_loader

loader = DirectoryLoader(
    path="folder ka naam",
    glob = '*.pdf', # iss book ke ander jitni v pdf files hai sabko load kro
    loader_cls = pypdf_loader #type of loader
)

documents = loader.load()
print(len(documents))

documents = loader.lazy_load()

for doc in documents:
    print(doc)  