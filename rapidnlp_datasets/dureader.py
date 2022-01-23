import json


def read_dureader_robust(input_files, **kwargs):
    if isinstance(input_files, str):
        input_files = [input_files]
    for input_file in input_files:
        with open(input_file, mode="rt", encoding="utf-8") as fin:
            info = json.load(fin)
        for d in info["data"]:
            paragraphs = d["paragraphs"]
            for p in paragraphs:
                context = p["context"]
                for qa in p["qas"]:
                    question = qa["question"]
                    _id = qa["id"]
                    context = " ".join(str(context).split())
                    question = " ".join(str(question).split())
                    instance = {
                        "id": _id,
                        "context": context,
                        "question": question,
                    }
                    if "answers" in qa:
                        _answer = qa["answers"][0]
                        answer = _answer["text"]
                        answer = " ".join(str(answer).split())
                        instance["answer"] = answer
                        if "answer_start" in _answer:
                            instance["answer_start"] = int(_answer["answer_start"])
                            instance["answer_end"] = int(_answer["answer_start"]) + len(answer)

                    yield instance


def read_dureader_checklist(input_files, **kwargs):
    if isinstance(input_files, str):
        input_files = [input_files]
    for input_file in input_files:
        with open(input_file, mode="rt", encoding="utf-8") as fin:
            info = json.load(fin)
        for data in info["data"]:
            for p in data["paragraphs"]:
                title = p["title"]
                context = p["context"]
                title = " ".join(str(title).split())
                context = " ".join(str(context).split())
                for qa in p["qas"]:
                    if qa["is_impossible"]:
                        continue
                    question = qa["question"]
                    question = " ".join(str(question).split())
                    instance = {
                        "id": qa["id"],
                        "context": title + context,
                        "question": question,
                    }
                    if "answers" in qa:
                        _answer = qa["answers"][0]
                        answer = _answer["text"]
                        answer = " ".join(str(answer).split())
                        instance["answer"] = answer
                        if "answer_start" in _answer:
                            instance["answer_start"] = int(_answer["answer_start"])
                            instance["answer_end"] = int(_answer["answer_start"]) + len(answer)

                    yield instance
