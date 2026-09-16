# data/

Source documents for ingestion (`run_ingestion()` globs `**/*.pdf` in this folder).

The ISO 15189:2022 standard itself isn't checked into this repo (licensed document — you need your own copy). Drop the PDF here, then run the backend's ingestion function to build the local Chroma + BM25 indexes.
