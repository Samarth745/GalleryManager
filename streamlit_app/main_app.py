import streamlit as st
from streamlit_app.utils import get_media_stats, get_sample_media

st.set_page_config(page_title="Media Library", page_icon="📚", layout="wide")

st.title("Media Library Dashboard")
st.write(
    "Use the sidebar pages to browse the library, run clustering, find duplicates, and search your media collection."
)

stats = get_media_stats()
cols = st.columns(4)
cols[0].metric("Total media", stats.get("total", 0))
cols[1].metric("Images", stats.get("image", 0))
cols[2].metric("Videos", stats.get("video", 0))
cols[3].metric("Embeddings", stats.get("embeddings", 0))

st.markdown("---")
st.header("Quick start")

st.write(
    "1. Go to the **Library** page to browse media files and thumbnails.\n"
    "2. Go to **Cluster** to inspect session clusters built from existing embeddings.\n"
    "3. Use **Duplicates** to find groups of near-duplicate media.\n"
    "4. Use **Search** to run text-based similarity search over your indexed media."
)

sample_media = get_sample_media()
if sample_media:
    st.subheader("Sample media from your collection")
    grid = st.columns(4)
    for idx, item in enumerate(sample_media):
        with grid[idx % 4]:
            st.image(item["file_path"], caption=item["file_name"], use_container_width=True)
else:
    st.info("No media samples are available yet. Run data ingestion first.")
