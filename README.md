## Design and Implementation of a Multidocument Retrieval Agent Using LlamaIndex
## AIM:
To design and implement a multidocument retrieval agent using LlamaIndex (formerly known as GPT Index), which extracts and synthesizes information from multiple research articles and to evaluate its performance by testing it with diverse queries, analyzing its ability to deliver concise, relevant, and accurate responses.

## PROBLEM STATEMENT:
The objective is to create an agent that can handle multiple research articles or documents and retrieve relevant information based on user queries. By leveraging LlamaIndex, we aim to build an efficient retrieval system that can access multiple documents, extract the necessary data, and synthesize it into meaningful responses, improving the speed and accuracy of information retrieval.

## DESIGN STEPS:
### STEP 1: Set Up the Environment
 - Install the necessary Python packages: LlamaIndex (GPT Index), openai, and any other dependencies (e.g., requests, pdfplumber for PDF parsing).
 - Obtain the OpenAI API key (or other model keys for retrieval).
```bash
%pip install llama-index-agent-openai
%pip install llama-index-embeddings-openai
%pip install llama-index-llms-openai
!pip install llama-index
```
### STEP 2: Data Ingestion and Indexing
 - Collect the research articles or documents in a structured format (PDF, Word, or plain text).
 - Use LlamaIndex to create an index for the documents, which will allow for efficient retrieval of information.
 - If the documents are in PDF format, use a PDF parsing library (e.g., pdfplumber or PyMuPDF) to extract text.
### STEP 3: Build the Retrieval Agent
 - Implement the agent that will:
 - Load and index the documents using LlamaIndex.
 - Accept a user query.
 - Retrieve the most relevant sections from the documents based on the query.
 - Synthesize the responses and generate a concise answer.
### STEP 4: Testing and Evaluation
 - Create various queries that test the agent's ability to extract and synthesize information from multiple documents.
 - Measure the performance by evaluating the relevance, accuracy, and conciseness of the retrieved answers.

## PROGRAM:
```python
from llama_index.core import (
    VectorStoreIndex,
    SimpleKeywordTableIndex,
    SimpleDirectoryReader,
)
from llama_index.core import SummaryIndex
from llama_index.core.schema import IndexNode
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.openai import OpenAI
from llama_index.core.callbacks import CallbackManager
from pathlib import Path
import requests
for title in wiki_titles:
    response = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "extracts",
            # 'exintro': True,
            "explaintext": True,
        },
    ).json()
    page = next(iter(response["query"]["pages"].values()))
    wiki_text = page["extract"]

    data_path = Path("data")
    if not data_path.exists():
        Path.mkdir(data_path)

    with open(data_path / f"{title}.txt", "w") as fp:
        fp.write(wiki_text)

# Load all wiki documents
city_docs = {}
for wiki_title in wiki_titles:
    city_docs[wiki_title] = SimpleDirectoryReader(
        input_files=[f"data/{wiki_title}.txt"]
    ).load_data()

import os
os.environ["OPENAI_API_KEY"] = "your-api-key"

from llama_index.agent.openai import OpenAIAgent

top_agent = OpenAIAgent.from_tools(
    tool_retriever=obj_index.as_retriever(similarity_top_k=3),
    system_prompt=""" \
You are an agent designed to answer queries about a set of given cities.
Please always use the tools provided to answer a question. Do not rely on prior knowledge.\

""",
    verbose=True,
)
# should use Boston agent -> vector tool
response = top_agent.query("Tell me about the arts and culture in Boston")
```

## OUTPUT:
![alt text](Image.png)

## RESULT:
Prompt Handling: The program constructs a query dynamically and feeds it into the LlamaIndex.
Document Indexing: LlamaIndex efficiently indexed multiple documents (research articles) to retrieve the relevant context.
Query Response: The system retrieved concise, relevant, and accurate responses by synthesizing the content from the indexed documents.
