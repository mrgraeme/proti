

from langchain_community.graphs import Neo4jGraph
import neo4j
import os
from dotenv import load_dotenv

load_dotenv('./.env')
graph = Neo4jGraph(url=os.getenv("NEO4J_URI"), username=os.getenv("NEO4J_USERNAME"), password=os.getenv("NEO4J_PASSWORD"))

QUERY = """
        MATCH (p:Protein) 
        RETURN p.name as name
        """
graph.query(QUERY)


with open('protein_desc.json', 'r') as json_file:
    protein_descriptions = json.load(json_file)
descriptions

for p in descriptions:
    print(p)
    for d in descriptions[p]:
        print(d)
        print(descriptions[p][d])

        QUERY = """
            MERGE (p1:Protein {name:$protein})
            MERGE (d1:Description {source:$source, text:$text})
            MERGE (d1)-[c:DESCRIBES]-(p1)
        """

        graph.query(QUERY, {"protein":p,
                            "source":d,
                            "text":descriptions[p][d]}, )


QUERY = """
    MATCH p = (p1:Protein {name:$protein})-[r:DESCRIBES]-(d)
    RETURN d.source, d.text
"""

graph.query(QUERY, {"protein":"ARID1A"} )