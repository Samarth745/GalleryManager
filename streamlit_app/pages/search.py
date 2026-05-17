import streamlit as st
from image_features.search_query import search_by_text
from utils import safe_image_path
from build_faiss_index import build_faiss_from_db
from database import init_db
st.set_page_config(page_title="Search", page_icon="🔎")

st.title("Text Search")
st.write("Search indexed media using a text query. Results are returned using the FAISS index.")

query = st.text_input("Search query", value="sunset")
threshold = st.slider("Similarity threshold", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
top_k = st.slider("Top results", min_value=5, max_value=50, value=10, step=5)

DB_PATH = "databases/media_database.db"
build_faiss_from_db(init_db(DB_PATH))
if st.button("Run search"):
    if not query.strip():
        st.warning("Please enter a search query.")
    else:
        try:
            results = search_by_text(query, similarity_threshold=threshold, top_k=top_k)
            if not results:
                st.info("No matching media found.")
            else:
                st.success(f"Found {len(results)} results")
                for item in results:
                    st.write(f"**{item['file_path']}**")
                    st.write(f"Similarity: {item['similarity']:.4f}")
                    if safe_image_path(item['file_path']):
                        st.image(safe_image_path(item['file_path']), use_column_width=True)
        except Exception as exc:
            st.error(f"Search failed: {exc}")
