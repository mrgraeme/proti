
from dotenv import load_dotenv
import os

# Common data processing
import json
import textwrap

# Langchain
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_openai import ChatOpenAI

# Warning control
# import warnings
# warnings.filterwarnings("ignore")

# Note the code below is unique to this course environment, and not a 
# standard part of Neo4j's integration with OpenAI. Remove if running 
# in your own environment.

load_dotenv('./.env')
OPENAI_ENDPOINT = 'https://api.openai.com/v1/embeddings'
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)


# Global constants
# VECTOR_INDEX_NAME = 'description_index'
# VECTOR_NODE_LABEL = 'Description'
# VECTOR_SOURCE_PROPERTY = 'text'
# VECTOR_EMBEDDING_PROPERTY = 'textEmbedding'

### Indexing - jsut once

graph.query("""
         CREATE VECTOR INDEX `description_index` IF NOT EXISTS
          FOR (d:Description) ON (d.textEmbedding) 
          OPTIONS { indexConfig: {
            `vector.dimensions`: 1536,
            `vector.similarity_function`: 'cosine'    
         }}
""")

# graph.refresh_schema()
print(graph.schema)
graph.query("SHOW INDEXES")


# graph.query("""
# MATCH (d:Description) 
# REMOVE d.textEmbedding
# """)


tick= graph.query("""
    MATCH (description:Description) 
    WHERE description.textEmbedding IS NOT NULL
    RETURN COUNT(description) as count
    """)[0]['count']


while True:
    result = graph.query("""
        MATCH (description:Description) 
        WHERE description.textEmbedding IS NULL
        WITH description 
        LIMIT 50
        WITH description, genai.vector.encode(
          description.text, 
          "OpenAI", 
          {
            token: $openAiApiKey, 
            endpoint: $openAiEndpoint
          }) AS vector
        CALL db.create.setNodeVectorProperty(description, "textEmbedding", vector)
        RETURN count(description) as processed
        """, 
        params={"openAiApiKey": OPENAI_API_KEY, "openAiEndpoint": OPENAI_ENDPOINT})
    
    # Break the loop if no more nodes were processed
    if result[0]['processed'] == 0:
        break
    tick = tick+50
    print(tick)



graph.query("""
    MATCH (description:Description) 
    WHERE description.textEmbedding IS NULL
    RETURN COUNT(description) AS count
    """)[0]['count']


def neo4j_vector_search(question):
  """Search for similar nodes using the Neo4j vector index"""
  vector_search_query = """
    WITH genai.vector.encode(
      $question, 
      "OpenAI", 
      {
        token: $openAiApiKey,
        endpoint: $openAiEndpoint
      }) AS question_embedding
    CALL db.index.vector.queryNodes($index_name, $top_k, question_embedding) yield node, score
    RETURN score, node.text AS text
  """
  similar = graph.query(vector_search_query, 
                     params={
                      'question': question, 
                      'openAiApiKey':OPENAI_API_KEY,
                      'openAiEndpoint': OPENAI_ENDPOINT,
                      'index_name':'description_index', 
                      'top_k': 10})
  return similar





search_results = neo4j_vector_search(
    'In a single sentence, tell me about SWI/SNF.'
)


search_results




retrieval_query_window = graph.query("""
MATCH window=
    (:Description)-[:DESCRIBES*0..1]->(node)-[:DESCRIBES*0..1]->(:Description)
WITH node, score, window as longestWindow 
  ORDER BY length(window) DESC LIMIT 1
WITH nodes(longestWindow) as descriptionList, node, score
  UNWIND descriptionList as descriptionRows
WITH collect(descriptionRows.text) as textList, node, score
RETURN apoc.text.join(textList, " \n ") as text,
    score,
    node {.source} AS metadata
""")

retrieval_query_window





vector_store_window = Neo4jVector.from_existing_index(
    embedding=OpenAIEmbeddings(),
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    index_name='description_index',
    node_label='Description',
    text_node_property=['text'],
    # embedding_node_property='textEmbedding',
    retrieval_query=retrieval_query_window,
)

# Create a retriever from the vector store
retriever_window = vector_store_window.as_retriever()

# Create a chatbot Question & Answer chain from the retriever
chain_window = RetrievalQAWithSourcesChain.from_chain_type(
    ChatOpenAI(temperature=0.1), 
    chain_type="stuff", 
    retriever=retriever_window
)

question = "tell me about ARID1A."

answer = chain_window(
    {"question": question},
    return_only_outputs=True,
)
print(textwrap.fill(answer["answer"]))




   
graph.query("""
    MATCH (description:Description)-[:DESCRIBES]->(protein:Protein),
            (protein)<-[:DESCRIBES]-(other_descriptions:Description),
            (protein)-[protein_protein]-(related_proteins:Protein),
            (related_proteins)<-[:DESCRIBES]-(other_protein_descriptions:Description)
    WHERE protein.name='ARID1A' AND protein_protein.pkg_score > 0.7
    WITH DISTINCT related_proteins, description, protein, other_descriptions, protein_protein, other_protein_descriptions
    WITH COLLECT (
            protein.name + ' is described by the following:  ' + description.text +
            ' and further described by ' + other_descriptions.text
            ) AS protein_summary, protein, related_proteins, protein_protein, other_protein_descriptions
            
    WITH COLLECT (
            protein.name + ' is related to ' + related_proteins.name + '. ' +
            related_proteins.name + ' has the following description ' + other_protein_descriptions.text
            ) AS related_protein_summary, protein_summary
    
    WITH COLLECT (apoc.text.join(protein_summary, "\n") + apoc.text.join(related_protein_summary, "\n")) as protein_summary
    
    RETURN apoc.text.join(protein_summary, "\n")
    """)



####################

protein_retrieval_query = ("""
    MATCH (node)-[:DESCRIBES]->(protein:Protein),
            (protein)<-[:DESCRIBES]-(other_descriptions:Description),
            (protein)-[protein_protein]-(related_proteins:Protein),
            (related_proteins)<-[:DESCRIBES]-(other_protein_descriptions:Description)
    WHERE protein_protein.pkg_score > 0.9
    WITH DISTINCT related_proteins, node, score, protein, other_descriptions, protein_protein, other_protein_descriptions
    WITH COLLECT (
            protein.name + ' is further described by ' + other_descriptions.text
            ) AS protein_summary, node, score, protein, related_proteins, protein_protein, other_protein_descriptions
    WITH COLLECT (
            protein.name + ' is related to ' + related_proteins.name + '. '
            ) AS related_protein_summary, protein_summary, node, score
    
    WITH COLLECT (apoc.text.join(protein_summary, "\n") + apoc.text.join(related_protein_summary, "\n")) AS protein_summary, node, score
    
    RETURN apoc.text.join(protein_summary, "\n") + 
    "\n" + node.text AS text,
    score,
    { 
      source: node.source
    } as metadata
    """)


vector_store_with_protein_relations = Neo4jVector.from_existing_index(
    OpenAIEmbeddings(),
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    database="neo4j",
    index_name='description_index',
    text_node_property=['text'],
    retrieval_query=protein_retrieval_query,
)


# Create a retriever from the vector store
retriever_with_protein_relations = vector_store_with_protein_relations.as_retriever()

# Create a chatbot Question & Answer chain from the retriever
protein_chain = RetrievalQAWithSourcesChain.from_chain_type(
    ChatOpenAI(temperature=0.1), 
    chain_type="stuff", 
    retriever=retriever_with_protein_relations
)

question = "tell me about SWI/SNF and its protein members"

protein_chain(
    {"question": question},
    return_only_outputs=True,
)