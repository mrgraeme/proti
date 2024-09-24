
import requests
from bs4 import BeautifulSoup
import json

import pandas as pd
import re

from openai import OpenAI
import os
from dotenv import load_dotenv

# Load your API key from an environment variable or secret management service
load_dotenv('./.env')

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)



protein_list = pd.read_csv('~/work/data/global/ccle/proteomics/CCLE_imputed_all.csv')[['Gene']]
protein_list
id_map = pd.read_csv('~/work/data/global/biomart/ncbi_id_map.txt')
id_map.columns = ['protein_id', 'protein', 'ncbi']
id_map = id_map[['protein', 'ncbi']]
protein_list = protein_list.merge(id_map, how='left', left_on='Gene', right_on='protein')
protein_list = protein_list.dropna().drop_duplicates()
protein_list['ncbi'] = protein_list['ncbi'].astype(int)
protein_list = protein_list[['protein', 'ncbi']]

protein_data = {}

def get_ncbi(protein, ncbi):
    ### Get NCBI description text
    URL = "https://www.ncbi.nlm.nih.gov/gene/%s" % (ncbi)
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, "html.parser")
    try: 
        target = soup.find('dt',string='Summary')
        for sib in target.find_next_siblings():
            if sib.name=="dt":
                break
            else:
                return(sib.text)
    except:
        print('no ncbi data for %s' % protein)
        return(False)
    
def get_wiki(protein):
    URL = 'https://en.wikipedia.org/wiki/%s' % row['protein']
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, "html.parser")
    try:
        pattern = re.compile(r'Function')
        h2_element = soup.find('h2', string = pattern)

        # Find the next <p> tag after the <h2> tag
        p_element = h2_element.find_next('p')
        p_element = ' '.join(p_element.stripped_strings)
        return(p_element)
    except:
        print('no wiki data for %s' % row['protein'])
        return(False)
    

def get_chatgpt_summary(protein):
    chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Please describe the protein %s and it's mechanisms and functions" % protein,
        }
    ],
    model="gpt-3.5-turbo",
    )
    return chat_completion.choices[0].message.content

for index, row in protein_list.iterrows():
    print(row['protein'], row['ncbi'])

    descriptions = {}

    ncbi_desc = get_ncbi(protein = row['protein'], ncbi = row['ncbi'])
    if ncbi_desc:
        descriptions['ncbi'] = ncbi_desc
    wiki_desc = get_wiki(protein = row['protein'])
    if wiki_desc:
        descriptions['wiki'] = wiki_desc
    descriptions['proti_summary'] = get_chatgpt_summary(row['protein'])
    
    protein_data[row['protein']] = descriptions

with open('protein_desc.json', 'w') as json_file:
    json.dump(protein_data, json_file)