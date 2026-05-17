# Streamlit Media Dashboard

This folder contains a simple multipage Streamlit app for browsing media, clustering images, detecting duplicates, and searching by text.

## Run the app

From the project root:

```bash
streamlit run streamlit_app/main_app.py
```

## Pages

- `main_app.py` - app home page with summary stats and sample thumbnails
- `pages/library.py` - browse indexed media files and preview images
- `pages/cluster.py` - run clustering on embedded images
- `pages/duplicates.py` - inspect duplicate groups
- `pages/search.py` - search media by text query

## Notes

- The app uses the existing database at `databases/media_database.db`.
- Existing embeddings are required for clustering, duplicate detection, and text search.
- Install packages from `streamlit_app/requirements.txt` if needed.
