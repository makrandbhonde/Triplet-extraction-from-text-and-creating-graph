from flask import Flask, request, render_template

import re
import pandas as pd
import bs4
import requests
import spacy
from spacy import displacy
nlp = spacy.load('en_core_web_sm')

from spacy.matcher import Matcher 
from spacy.tokens import Span 

import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tqdm import tqdm


app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        data = request.form.get("input_text")


        from csv import writer

        text_field = ""
        text_field = data


        sentences = []
        sentences = text_field.split(".")
        for i in sentences:
        
            with open('sentences.csv', 'a', newline='') as f_object:  
            
                writer_object = writer(f_object)

                writer_object.writerow([i])  

                f_object.close()


        candidate_sentences = pd.read_csv("sentences.csv")
        candidate_sentences.shape




        def get_entities(sent):
         
            ent1 = ""
            ent2 = ""

            prv_tok_dep = ""    # dependency tag of previous token in the sentence
            prv_tok_text = ""   # previous token in the sentence

            prefix = ""
            modifier = ""

                               
            for tok in nlp(sent):
                # chunk 2
                # if token is a punctuation mark then move on to the next token
                if tok.dep_ != "punct":
                # check: token is a compound word or not
                    if tok.dep_ == "compound":
                        prefix = tok.text
                        # if the previous word was also a 'compound' then add the current word to it
                        if prv_tok_dep == "compound":
                            prefix = prv_tok_text + " "+ tok.text
                
                # check: token is a modifier or not
                if tok.dep_.endswith("mod") == True:
                    modifier = tok.text
                    # if the previous word was also a 'compound' then add the current word to it
                    if prv_tok_dep == "compound":
                        modifier = prv_tok_text + " "+ tok.text
                
                # chunk 3
                if tok.dep_.find("subj") == True:
                    ent1 = modifier +" "+ prefix + " "+ tok.text
                    prefix = ""
                    modifier = ""
                    prv_tok_dep = ""
                    prv_tok_text = ""      

                # chunk 4
                if tok.dep_.find("obj") == True:
                    ent2 = modifier +" "+ prefix +" "+ tok.text
                    
                # chunk 5  
                # update variables
                prv_tok_dep = tok.dep_
                prv_tok_text = tok.text
            

            return [ent1.strip(), ent2.strip()]



      

        entity_pairs = []

        for i in tqdm(candidate_sentences["sentence"]):
            entity_pairs.append(get_entities(i))

        entity_pairs[10:20]




        def get_relation(sent):

            doc = nlp(sent)

            # Matcher class object 
            matcher = Matcher(nlp.vocab)

            #define the pattern 
            pattern = [{'DEP':'ROOT'}, 
                        {'DEP':'prep','OP':"?"},
                        {'DEP':'agent','OP':"?"},  
                        {'POS':'ADJ','OP':"?"}] 

            matcher.add("matching_1", [pattern]) 

            matches = matcher(doc)
            k = len(matches) - 1

            span = doc[matches[k][1]:matches[k][2]] 

            return(span.text)



     

        relations = [get_relation(i) for i in tqdm(candidate_sentences['sentence'])]

        print(relations)

        pd.Series(relations).value_counts()[:50]

        source = [i[0] for i in entity_pairs]

        
        target = [i[1] for i in entity_pairs]

        kg_df = pd.DataFrame({'source':source, 'target':target, 'edge':relations})
        
        
        #clearing the csv for next input
        df = candidate_sentences 
        df = df.head(0)
        df.to_csv('sentences.csv',index=False, header=True)

        
        # creating graph from a dataframe
        G=nx.from_pandas_edgelist(kg_df, "source", "target", 
                                edge_attr=True, create_using=nx.MultiDiGraph())



        plt.figure(figsize=(12,12))

        pos = nx.spring_layout(G)
        plt.ioff()
        
        # make a variable containing not positions
        gpos = nx.circular_layout(G) 

        # draw the graph                                                     
        nx.draw_networkx(G,pos=gpos) 

        #adding labels on edges of graph (RELATIONS)
        nx.draw_networkx_edge_labels(G,pos=gpos,edge_labels={(u,v):w for u,v,w in G.edges(data='edge')}) 

        plt.close(plt.figure())
        plt.savefig('static/plot.png')
        plt.close(plt.figure())
        
        return render_template("output.html")



















    else:
        return render_template("form.html")


if __name__=='__main__':
    app.run()







