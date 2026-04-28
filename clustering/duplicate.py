import streamlit as st
import pandas as pd
import os
from PIL import Image

from image_features.duplicate_detection import cluster_duplicate_images


def load_sample_data(path = r"databases\data.pkl"):
    df = pd.read_pickle(path)
    return cluster_duplicate_images(df)[["filename", "filepath", "duplicated"]]

@st.cache_data
def load_image(filepath):
    return Image.open(filepath)

def main():
    st.title("Duplicated Images Viewer and Deleter")

    df = load_sample_data()

    # Dropdown to select one duplicated group at a time
    duplicated_groups = df['duplicated'].unique()
    selected_group = st.selectbox("Select Duplicated Group to View", duplicated_groups)

    # Filter dataframe by selected duplicated group
    group_df = df[df['duplicated'] == selected_group]

    # Initialize session state set to track selected images
    if 'selected_images' not in st.session_state:
        st.session_state.selected_images = set()

    st.write(f"Showing {len(group_df)} images in duplicated group '{selected_group}':")

    cols = st.columns(len(group_df))
    for idx, (_, row) in enumerate(group_df.iterrows()):
        with cols[idx]:
            if os.path.exists(row['filepath']):
                img = load_image(row['filepath'])
                st.image(img, use_column_width=True)
            else:
                st.write("Image file not found")

            key_checkbox = f"select_{row['filename']}"
            checked = st.checkbox("Select", key=key_checkbox)
            if checked:
                st.session_state.selected_images.add(row['filename'])
            else:
                st.session_state.selected_images.discard(row['filename'])
            # Open image link in new tab
            #st.markdown(f"[Open in new tab]({row['filepath']}){{:target=\"_blank\"}}", unsafe_allow_html=True)
    st.write(f"SELECTED - {st.session_state.selected_images}")
    # Delete selected images button
    if st.button("Delete Selected Images"):
        if not st.session_state.selected_images:
            st.warning("No images selected for deletion.")
        else:
            confirm = st.checkbox("Confirm deletion of selected images")
            if confirm:
                deleted_files = []
                for filename in list(st.session_state.selected_images):
                    rows = df[df['filename'] == filename]
                    for _, row in rows.iterrows():
                        filepath = row['filepath']
                        # Delete physical file if exists
                        st.write(filepath)
                        if os.path.exists(filepath):
                            try:
                                os.remove(filepath)
                                deleted_files.append(filepath)
                            except Exception as e:
                                st.error(f"Error deleting {filepath}: {e}")
                        # Remove from dataframe in memory
                        df.drop(df[df['filename'] == filename].index, inplace=True)
                    st.session_state.selected_images.discard(filename)
                st.success(f"Deleted files: {deleted_files}")
                st.rerun()

if __name__ == "__main__":
    main()
