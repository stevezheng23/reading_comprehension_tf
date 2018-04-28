import argparse
import codecs
import json
import os.path

def add_arguments(parser):
    parser.add_argument("--format", help="format to generate", required=True)
    parser.add_argument("--input_file", help="path to input file", required=True)
    parser.add_argument("--output_file", help="path to output file", required=True)

def get_word_span(context, text, start):
    end = start + len(text)
    span_length = len(text.split(" "))
    span_end = len(context[:end].split(" ")) - 1
    span_start = span_end - span_length + 1
    
    return span_start, span_end

def preprocess(file_name):
    if not os.path.exists(file_name):
        raise FileNotFoundError("file not found")
    
    flatten_data_list = []
    with open(file_name, "r") as file:
        json_content = json.load(file)
        for article in json_content["data"]:
            for paragraph in article["paragraphs"]:
                context = paragraph["context"]
                for qa in paragraph["qas"]:
                    qa_id = qa["id"]
                    question = qa["question"]
                    for answer in qa["answers"]:
                        answer_text = answer["text"]
                        answer_start = answer["answer_start"]
                        answer_word_start, answer_word_end = get_word_span(context,
                            answer_text, answer_start)
                        
                        flatten_data = {
                            "id": qa_id,
                            "question": question,
                            "context": context,
                            "answer_text": answer_text,
                            "answer_span": {
                                "start": answer_word_start,
                                "end": answer_word_end
                            }
                        }
                        flatten_data_list.append(flatten_data)
    
    return flatten_data_list

def output_to_json(data_list, file_name):
    with open(file_name, "w") as file:
        data_json = json.dumps(data_list, indent=4)
        file.write(data_json)

def output_to_plain(data_list, file_name):
    with open(file_name, "wb") as file:
        for data in data_list:
            data_plain = "{0}\t{1}\t{2}\t{3}\t{4}|{5}\r\n".format(data["id"], data["question"], data["context"].replace("\n", " "),
                data["answer_text"], data["answer_span"]["start"], data["answer_span"]["end"])
            file.write(data_plain.encode("utf-8"))

def output_to_split(data_list, file_prefix):
    with open("{0}.question".format(file_prefix), "wb") as q_file, open("{0}.context".format(file_prefix), "wb") as c_file, open("{0}.answer_text".format(file_prefix), "wb") as at_file, open("{0}.answer_span".format(file_prefix), "wb") as as_file:
        for data in data_list:
            q_data_plain = "{0}\r\n".format(data["question"])
            q_file.write(q_data_plain.encode("utf-8"))
            c_data_plain = "{0}\r\n".format(data["context"].replace("\n", " "))
            c_file.write(c_data_plain.encode("utf-8"))
            at_data_plain = "{0}\r\n".format(data["answer_text"])
            at_file.write(at_data_plain.encode("utf-8"))
            as_data_plain = "{0}|{1}\r\n".format(data["answer_span"]["start"], data["answer_span"]["end"])
            as_file.write(as_data_plain.encode("utf-8"))

def main(args):
    flatten_data = preprocess(args.input_file)
    if (args.format == 'json'):
        output_to_json(flatten_data, args.output_file)
    elif (args.format == 'plain'):
        output_to_plain(flatten_data, args.output_file)
    elif (args.format == 'split'):
        output_to_split(flatten_data, args.output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    main(args)