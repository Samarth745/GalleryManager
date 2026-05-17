import streamlit as st
from database import init_db, load_image_embeddings_for_clustering
from image_processing.cluster_trip_processor import find_sessions, build_similarity
from streamlit_app.utils import safe_image_path

st.set_page_config(page_title="Cluster", page_icon="🧠")

st.title("Image Clustering")
st.write(
    "Use existing image embeddings to group similar photos and sessions. "
    "This page shows clusters based on both visual similarity and time proximity."
)

run_clustering = st.button("Run clustering")

if run_clustering:
    try:
        conn = init_db()
        df = load_image_embeddings_for_clustering(conn)
        if df.empty:
            st.warning("No image embeddings are available yet. Please generate embeddings first.")
        else:
            with st.spinner("Computing clusters..."):
                df = find_sessions(df)
                labels = build_similarity(df)
                df["cluster_label"] = labels

            st.success("Clustering complete.")
            valid_labels = [label for label in set(labels) if label != -1]
            st.write(f"Found {len(valid_labels)} clusters")

            cluster_options = sorted(valid_labels)
            selected_cluster = st.selectbox("Choose a cluster to preview", [None] + cluster_options)

            if selected_cluster is not None:
                cluster_rows = df[df["cluster_label"] == selected_cluster]
                st.write(cluster_rows[["file_id", "date_taken", "session_id"]])
                st.markdown("---")
                st.write("Showing up to 10 cluster members")
                for idx, row in cluster_rows.head(10).iterrows():
                    st.write(f"- File ID: {row['file_id']}  |  Date: {row['date_taken']}  |  Session: {row['session_id']}")
    except Exception as exc:
        st.error(f"Clustering failed: {exc}")
