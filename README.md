## Design and Implementation of a Multidocument Retrieval Agent Using LlamaIndex
## AIM:
To design and implement a multidocument retrieval agent using LlamaIndex (formerly known as GPT Index), which extracts and synthesizes information from multiple research articles and to evaluate its performance by testing it with diverse queries, analyzing its ability to deliver concise, relevant, and accurate responses.

## PROBLEM STATEMENT:
The objective is to create an agent that can handle multiple research articles or documents and retrieve relevant information based on user queries. By leveraging LlamaIndex, we aim to build an efficient retrieval system that can access multiple documents, extract the necessary data, and synthesize it into meaningful responses, improving the speed and accuracy of information retrieval.

## DESIGN STEPS:
### **Algorithm for PDF Analysis and Query Processing Using Agent Tools**
### **1. Input Initialization**
- **Inputs**:
  - `urls`: A list of URLs pointing to the research papers (optional for download).
  - `papers`: Local filenames of the PDF files.
- **Output**:
  - Tools for each paper (`vector_tool` and `summary_tool`).

### **2. Set Up Document Processing Tools**
1. Import the `get_doc_tools` function for generating document-specific tools.
2. Iterate over the `papers` list:
   - Print the current paper being processed.
   - For each paper, call `get_doc_tools` with:
     - The paper file path.
     - The stem of the file path (used as an identifier).
   - Store the generated tools (`vector_tool` and `summary_tool`) in a dictionary, mapping to the paper.

### **3. Initialize Tools**
1. Combine all tools from the dictionary into a single list (`initial_tools`).

### **4. Set Up the LLM**
1. Import and initialize the `OpenAI` LLM:
   - Use the `gpt-4` model.

### **5. Configure the Agent Worker**
1. Import `FunctionCallingAgentWorker` and `AgentRunner` to manage agent functionalities.
2. Create an agent worker using `FunctionCallingAgentWorker.from_tools`:
   - Pass the combined `initial_tools` list.
   - Set the LLM (`llm`).
   - Enable verbose output for detailed logging.

### **6. Run the Agent Query**
1. Create an `AgentRunner` instance with the configured worker.
2. Query the agent with a question:
   - Include specific queries about datasets and results from the `LongLoRA` paper.

### **7. Output the Result**
1. Retrieve and display the response from the agent.

## PROGRAM:
```python
urls = [
    "https://openreview.net/pdf?id=VtmBAGCN7o",
    "https://openreview.net/pdf?id=6PmJoRfdaK",
    "https://openreview.net/pdf?id=hSyW5go0v8",
]

papers = [
    "metagpt.pdf",
    "longlora.pdf",
    "selfrag.pdf",
]

from utils import get_doc_tools
from pathlib import Path

paper_to_tools_dict = {}
for paper in papers:
    print(f"Getting tools for paper: {paper}")
    vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]

initial_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]

from llama_index.llms.openai import OpenAI
llm = OpenAI(model="gpt-4")

from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner

agent_worker = FunctionCallingAgentWorker.from_tools(
    initial_tools, 
    llm=llm, 
    verbose=True
)

agent = AgentRunner(agent_worker)
response = agent.query(
    "Tell me about the evaluation dataset used in LongLoRA, "
    "and then tell me about the evaluation results"
)
```

## OUTPUT:
![alt text](Output.png)

## RESULT:
Prompt Handling: The program constructs a query dynamically and feeds it into the LlamaIndex.
Document Indexing: LlamaIndex efficiently indexed multiple documents (research articles) to retrieve the relevant context.
Query Response: The system retrieved concise, relevant, and accurate responses by synthesizing the content from the indexed documents.
