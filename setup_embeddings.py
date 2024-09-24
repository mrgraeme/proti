
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
OPENAI_ENDPOINT = 'https://api.openai.com/v1/embeddings'
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

load_dotenv('./.env')
graph = Neo4jGraph(url=os.getenv("NEO4J_URI"), username=os.getenv("NEO4J_USERNAME"), password=os.getenv("NEO4J_PASSWORD"))


# Global constants
# VECTOR_INDEX_NAME = 'form_10k_chunks'
# VECTOR_NODE_LABEL = 'Chunk'
# VECTOR_SOURCE_PROPERTY = 'text'
# VECTOR_EMBEDDING_PROPERTY = 'textEmbedding'

### Indexing - jsut once

# graph.query("""
#          CREATE VECTOR INDEX `description_index` IF NOT EXISTS
#           FOR (d:Description) ON (d.textEmbedding) 
#           OPTIONS { indexConfig: {
#             `vector.dimensions`: 1536,
#             `vector.similarity_function`: 'cosine'    
#          }}
# """)

# graph.refresh_schema()
# print(graph.schema)
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