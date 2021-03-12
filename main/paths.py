from pathlib import Path


class Path_Handler:
    def __init__(self):
        self.dict = {}

    def fill_dict(self):
        root = Path(__file__).resolve().parent.parent
        path_dict = {}
        path_dict["root"] = root

        path_dict["data"] = root / "data"
        path_dict["config"] = root / "config"

        path_dict["eval"] = root / "files" / "eval"
        path_dict["checkpoints"] = root / "files" / "checkpoints"
        path_dict["images"] = root / "files" / "images"

        path_dict["hist"] = root / "files" / "eval" / "hist"
        path_dict["fake"] = root / "files" / "images" / "fake"
        path_dict["recon"] = root / "files" / "images" / "recon"
        path_dict["gaug"] = root / "files" / "images" / "gaug"

        self.dict = path_dict

    def create_paths(self):
        for path in self.dict.values():
            if not Path.exists(path):
                Path.mkdir(path)

    def _dict(self):
        self.fill_dict()
        self.create_paths()
        return self.dict
