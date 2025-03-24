#!/usr/bin/env python
# coding: utf-8

# In[ ]:
pip install faiss-cpu langchain langchain-community transformers sentence-transformers

pip install pypdf

# In[3]:


from langchain.document_loaders import PyPDFLoader, GoogleSpeechToTextLoader, TextLoader


# In[4]:


file_path = "Medical_book.pdf"
if file_path.endswith(".pdf"):
  data = PyPDFLoader(file_path)
else:
  data = TextLoader(file_path)
document = data.load()
document


# In[5]:


from langchain.text_splitter import RecursiveCharacterTextSplitter
text_split = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
documents = text_split.split_documents(document)
documents


# In[6]:


from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-miniLM-L6-v2")
# vector_store = FAISS.from_documents(documents,embedding_model)
if not documents:
    print("Error: The 'documents' list is empty. Ensure documents are properly loaded.")
else:
    vector_store = FAISS.from_documents(documents, embedding_model)


# In[7]:


retriever = vector_store.as_retriever(search_kwargs={"k":1})


# In[ ]:


# get_ipython().system('pip install groq')


# In[9]:

pip install groq
from groq import Groq
groq_client = Groq(api_key="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")


# In[10]:


def response(query):
  responses = retriever.get_relevant_documents(query)
  ans = responses[0].page_content

  groq_response = groq_client.chat.completions.create(model = "llama-3.3-70b-versatile",
                                                      messages = [{"role":"system","content":ans},
                                                                  {"role":"user","content":query}])
  return groq_response.choices[0].message.content

while True:
  query = input("Enter a query or type 'exit' to quit: ")
  final_response = response(query)
  print(f"Yogita: {final_response}")
  if query == "exit":
    print("Thank You!")
    break

#get_response("What is Computer Network? Explain Its Advantages, Disadvantages and Use of Computer Network.")


# ##extra

# In[11]:


# def response(query):
#   responses = retriever.get_relevant_documents(query)
#   ans = responses[0].page_content

#   groq_response = groq_client.chat.completions.create(model = "llama-3.3-70b-versatile",
#                                                       messages = [{"role":"system","content":ans},
#                                                                   {"role":"user","content":query}])
#   return groq_response.choices[0].message.content

# while True:
#   query = input("Enter a query or type 'exit' to quit: ")
#   final_response = response(query)
#   print(f"Yogita: {final_response}")
#   if query == "exit":
#     print("Thank You!")
#     break


# In[12]:


# while True:
#   query = input('Enter a query or type "exit" to quit: ')
#   if query == "exit":
#     print("Thank You!")
#     break
#   else:
#     ans = get_response(query)
#     print("Bot: ", ans)

