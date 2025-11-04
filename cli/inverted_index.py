from helpers import tokenize, get_movie_data_from_file
from pathlib import Path
from pickle import dump

class InvertedIndex:
    index: dict[str, set[int]]
    docmap: dict[int, dict[str, object]]

    def __init__(self):
        self.index = {}
        self.docmap = {}

    def __add_document(self, doc_id: int, text: str):
        # tokenize input text
        tokens = tokenize(text=text)
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
            
    
    def get_documents(self, term: str) -> list[int]:
        # term is given token
        # get the set of doc ids
        doc_ids = self.index.get(term.lower())
        if not doc_ids:
            return []
        return sorted(doc_ids)
    
    def build(self):
        movies_list = get_movie_data_from_file()
        for movie in movies_list:
            self.__add_document(movie["id"], f'{movie["title"]} {movie["description"]}')
            self.docmap[movie["id"]] = movie

    def save(self):
        # if a cache directory does not exist,
        # create it
        cache_dir_path = Path(__file__).resolve().parent.parent / "cache"
        if not cache_dir_path.exists():
            cache_dir_path.mkdir()
        
       
        # dir exists write files into it
        index_file_path = cache_dir_path / "index.pkl"
        docmap_file_path = cache_dir_path / "docmap.pkl"
        # index_file_path.touch()
        # docmap_file_path.touch()
        with index_file_path.open("wb") as index_dump, docmap_file_path.open("wb") as docmap_dump:
            dump(self.index, index_dump)
            dump(self.docmap, docmap_dump)
        
        
