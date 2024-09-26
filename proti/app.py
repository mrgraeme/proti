from flask import Flask, render_template, request
from dotenv import load_dotenv
import os
import textwrap

# Load your RAG setup (Langchain + Neo4j)
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv('./.env')
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Initialize Neo4j graph
graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)

# Your Cypher generation template
CYPHER_GENERATION_TEMPLATE = """
Task:Generate Cypher statement to 
query a graph database.
Instructions:
Use only the provided relationship types and properties in the 
schema. Do not use any other relationship types or properties that 
are not provided.
Schema:
{schema}

Do not respond to any questions that might construct a Cypher statement that use keywords DELETE, CREATE or MERGE
Where you provide a list of proteins include a link to: https://protkg.com/protein_list/?protein_list=ARID1A+ [ list of proteins ]

Examples: Here are a few examples of generated Cypher statements for particular questions:

# What proteins are related to ARID1A?
MATCH (p:Protein)-[r]->(p2:Protein)
WHERE p.name = 'ARID1A'
RETURN p2.name, type(r) as relationship_type

# Describe the proteins are related to ARID1A?
MATCH (p:Protein)-[r]->(p2:Protein)<-[:DESCRIBES]-(other_desc:Description)
WHERE p.name = 'ARID1A' AND other_desc='ncbi'
RETURN other_desc.text

# Describe ARID1A
MATCH (p:Protein)<-[:Describes]-(description:Description)
WHERE p.name = 'ARID1A' AND description.source='ncbi'
RETURN description.text

# What functions are associated with ARID1A
MATCH (p:Protein)<-[]-(func:Function)
WHERE p.name = 'ARID1A'
RETURN func.name

# What proteins are associated with Homologous recombination
MATCH (p:Protein)<-[]-(func:Function)
WHERE func.name = 'Homologous recombination'
RETURN func.name

# How are MUS81, RBBP8, RAD54B, related?
MATCH (p:Protein)-[r]->(p2:Protein)
WHERE p.name IN ['MUS81', 'RBBP8', 'RAD54B'] AND p2.name IN ['MUS81', 'RBBP8', 'RAD54B']
RETURN p.name, type(r), p2.name




The question is:
{question}"""

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], 
    template=CYPHER_GENERATION_TEMPLATE
)

# Set up the GraphCypherQAChain
cypherChain = GraphCypherQAChain.from_llm(
    ChatOpenAI(model="gpt-4", temperature=0),
    graph=graph,
    verbose=True,
    cypher_prompt=CYPHER_GENERATION_PROMPT,
    allow_dangerous_requests=True
)

# Flask app setup
app = Flask(__name__)

# Function to handle the question and return the formatted response
def prettyCypherChain(question: str) -> str:
    response = cypherChain.run(question)
    return textwrap.fill(response, 60)

# Define the main route for the app
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        question = request.form["question"]
        # Run the question through the cypher chain
        answer = prettyCypherChain(question)
        return render_template("index.html", question=question, answer=answer)
    return render_template("index.html")

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
