import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import cohere
import os
from dotenv import load_dotenv
load_dotenv()

co = cohere.Client(os.environ["CO_API_KEY"])
df = pd.read_parquet('embeddings_shein.parquet')



#  columns in data: name, description,price, cover_image

def display_item_card(item):
    st.markdown(
        f"""
        <div style="background-color:#f0f0f0; padding: 10px; border-radius: 10px; box-shadow: 0px 0px 10px 0px rgba(0,0,0,0.1); flex: 0 0 20%; margin: 10px;">
            <h2 style="color:#333333; text-align:center; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{item['name']}</h2>
            <img src="{item['cover_image']}" style="display:block; margin:auto; width:150px; border-radius:5px;">
            <p style="color:#666666; text-align:center; font-style: italic;">{item['description']}</p>
            <p style="color:#333333; text-align:center; font-weight:bold;">Price: {item['price']}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

def get_embeddings(texts,model='embed-english-v3.0', input_type = 'search_query'):
  output = co.embed(
                model=model,
                texts=texts,
                input_type = input_type
                )
                
  return output.embeddings


def get_similarity(target,candidates):
  # Turn list into array
  candidates = np.array(candidates)
  target = np.expand_dims(np.array(target),axis=0)

  # Calculate cosine similarity
  sim = cosine_similarity(target,candidates)
  sim = np.squeeze(sim).tolist()
  sort_index = np.argsort(sim)[::-1]
  sort_score = [sim[i] for i in sort_index]
  similarity_scores = zip(sort_index,sort_score)

  # Return similarity scores
  return similarity_scores

def search(new_query):
  # Get embeddings of the new query
  new_query_embeds = get_embeddings([new_query])[0]
  top_recommendations = list(get_similarity(new_query_embeds, df.query_embeds.tolist()))[:10]
  print(top_recommendations)
  returned_listings = [ df.iloc[i[0]] for i in top_recommendations ]
  return pd.DataFrame(returned_listings)


# Streamlit UI
col1, col2, col3= st.columns(3)
with col1:
    st.image('assets/searchy_logo.png',  width = 100)
with col2:
    st.markdown("## " + 'Searchy')

# Search functionality
search_query = st.text_input("Text Input", placeholder = "Search for items...", label_visibility = 'hidden')

# Filter items based on search query
filtered_df = search(search_query) if search_query else df

# Display card for each item
for index, item in filtered_df.iloc[:20].iterrows():
    display_item_card(item)
