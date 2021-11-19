import json


def read_dureader_rubost(input_files, **kwargs):
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
                    if not qa["answers"]:
                        continue
                    answer = qa["answers"][0]["text"]
                    instance = {"context": context, "question": question, "answer": answer, "id": _id}
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
                for qa in p["qas"]:
                    if qa["is_impossible"]:
                        continue
                    question = qa["question"]
                    answer = qa["answers"][0]["text"]
                    instance = {"context": title + context, "question": question, "answer": answer, "id": qa["id"]}
                    yield instance


def read_jsonl_files(input_files, context_key="context", question_key="question", answers_key="answer", id_key="id", **kwargs):
    if isinstance(input_files, str):
        input_files = [input_files]
    _id = 0
    for input_file in input_files:
        with open(input_file, mode="rt", encoding="utf-8") as fin:
            for line in fin:
                data = json.loads(line)
                answers = data[answers_key]
                if not answers:
                    continue
                if not isinstance(answers, list):
                    answers = [answers]
                answer = answers[0]
                instance_id = data.get(id_key, _id)
                _id += 1
                instance = {"context": data[context_key], "question": data[question_key], "answer": answer, "id": instance_id}
                yield instance
