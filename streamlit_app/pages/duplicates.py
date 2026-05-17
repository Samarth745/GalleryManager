import streamlit as st
from image_features.duplicate_detection import get_duplicates_clusters
from streamlit_app.utils import safe_image_path

st.set_page_config(page_title="Duplicates", page_icon="🧩")

st.title("Duplicate Finder")
st.write(
    "Find near-duplicate media in your collection using the existing embeddings. "
    "Select a file from a duplicate group to inspect similar originals."
)

try:
    duplicate_clusters = get_duplicates_clusters()
    if not duplicate_clusters:
        st.info("No duplicate groups found.")
    else:
        st.success(f"Found {len(duplicate_clusters)} duplicate groups.")

        group_selection = st.selectbox(
            "Choose a duplicate group to explore",
            list(range(len(duplicate_clusters)))
        )

        selected = duplicate_clusters[group_selection]
        st.write(f"Group size: {len(selected)}")

        for item in selected:
            st.write(f"**{item['file_path']}** — similarity: {item['similarity']:.4f}")
            if safe_image_path(item['file_path']):
                st.image(safe_image_path(item['file_path']), use_column_width=True)
except Exception as exc:
    st.error(f"Duplicate detection failed: {exc}")
