import json

def process_data(*paths):
    datadict = {}
    for path in list(paths):
        assert type(path) == type("str"), "Path not right!"
        with open(path, "r") as f:
            json_data = json.load(f)
            images_list = json_data["images"]
            annotations_list = json_data["annotations"]   

        for key in images_list:
            datadict[key["id"]] = {"file_name": key["file_name"], "captions": []}
        for anno in annotations_list:
            datadict[anno["image_id"]]["captions"].append(anno["caption"])
    return len(datadict)