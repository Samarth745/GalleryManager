import streamlit as st
from streamlit_app.utils import get_full_media_df, safe_image_path

st.set_page_config(page_title="Library", page_icon="🖼️")

st.title("Media Library")
st.write("Browse the media files that are currently indexed in the database.")

try:
    df = get_full_media_df()
except Exception as exc:
    st.error(f"Could not load library table: {exc}")
    st.stop()

st.markdown(f"**Files indexed:** {len(df)}")

media_type = st.selectbox("Filter by media type", ["all", "image", "video"])
if media_type != "all":
    df = df[df["type"] == media_type]

if df.empty:
    st.warning("No media files to display.")
    st.stop()

st.dataframe(df[["file_name", "file_path", "type", "date_taken", "filesize"]].head(50))

st.markdown("---")
st.subheader("Preview thumbnails")

preview_rows = df[df["type"] == "image"].head(12)
cols = st.columns(4)
for idx, row in preview_rows.iterrows():
    with cols[idx % 4]:
        path = safe_image_path(row["file_path"])
        if path:
            st.image(path, caption=row["file_name"], use_column_width=True)
        else:
            st.write(row["file_name"])
