from helpers import tokenize, get_movie_data_from_file
from pathlib import Path
from pickle import dump

class InvertedIndex:
    def __init__(self, index, docmap):
        self.index = index
        self.docmap = docmap

    def __add_document(self, doc_id: int, text: str):
        # tokenize input text
        tokens = tokenize(text=text)
        for token in tokens:
            if token in self.index:
                self.index[token].add(doc_id)
                continue
            else:
                self.index[token] = set()
                self.index[token].add(doc_id)
    
    def get_document(self, term):
        # term is given token
        # get the set of doc ids
        result_doc_ids = []
        doc_ids = self.index.get(term.lower())
        for id in sorted(doc_ids):
            result_doc_ids.append(id)
        return result_doc_ids
    
    def build(self):
        movies_list = get_movie_data_from_file()
        for movie in movies_list:
            self.__add_document(movie["id"], f"{movie["title"]} {movie["description"]}")
            if movie["id"] not in self.docmap:
                self.docmap[movie["id"]] = movie
                continue
            else:
                raise ValueError("movie ids collide")

    def save(self):
        # if a cache directory does not exist,
        # create it
        cache_dir_path = Path(__file__).resolve().parent.parent / "cache"
        if not cache_dir_path.exists():
            Path.mkdir(cache_dir_path)
        
        if cache_dir_path.exists and cache_dir_path.is_dir:
            # dir exists write files into it
            index_file_path = cache_dir_path / "index.pkl"
            docmap_file_path = cache_dir_path / "docmap.pkl"
            with index_file_path.open() as index_dump, docmap_file_path.open() as docmap_dump:
                dump(self.index, index_dump)
                dump(self.docmap, docmap_dump)
        
        
