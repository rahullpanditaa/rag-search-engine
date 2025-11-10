from pathlib import Path

DATA_DIR_PATH = (Path(__file__).resolve().parent.parent.parent).resolve() / "data" 
MOVIES_DATA_PATH = DATA_DIR_PATH / "movies.json"
STOPWORDS_FILE_PATH = DATA_DIR_PATH / "stopwords.txt"

CACHE_DIR_PATH = Path(__file__).resolve().parent.parent.parent / "cache"
# Inverted Index 
INDEX_FILE_PATH = CACHE_DIR_PATH / "index.pkl"
DOCMAP_FILE_PATH = CACHE_DIR_PATH / "docmap.pkl"
TERM_FREQUENCIES_FILE_PATH = CACHE_DIR_PATH / "term_frequencies.pkl"
DOC_LENGTHS_PATH = CACHE_DIR_PATH / "doc_lengths.pkl"

# Tunable Parameters for calculating BM25 score
BM25_K1 = 1.5
BM25_B = 0.75

# Semantic Search
MOVIE_EMBEDDINGS_PATH = CACHE_DIR_PATH / "movie_embeddings.npy"

# Chunked Semantic Search
CHUNK_EMBEDDINGS_PATH = CACHE_DIR_PATH / "chunk_embeddings.npy"
CHUNK_METADATA_PATH = CACHE_DIR_PATH / "chunk_metadata.json"

DEFAULT_CHUNK_SIZE = 200
DEFAULT_CHUNK_OVERLAP = 1
DEFAULT_SEMANTIC_CHUNK_SIZE = 4
SCORE_PRECISION = 3