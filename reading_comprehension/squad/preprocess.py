import argparse
import codecs
import json
import os.path
import string
import re
import nltk
import spacy

spacy_nlp = spacy.load('en')

def add_arguments(parser):
    parser.add_argument("--format", help="format to generate", required=True)
    parser.add_argument("--input_file", help="path to input file", required=True)
    parser.add_argument("--output_file", help="path to output file", required=True)

def nltk_tokenize(text, lower_case=False, remove_punc=False):
    def process_token(tokens):
        special = ("-", "£", "€", "¥", "¢", "₹", "\u2212", "\u2014", "\u2013",
                   "/", "~", '"', "'", "\ud01C", "\u201C", "\u2019", "\u201D", "\u2018", "\u00B0")
        pattern = "([{}])".format("".join(special))
        processed_tokens = []
        for token in tokens:
            token = token.replace("''", '" ').replace("``", '" ')
            processed_tokens.extend(re.split(pattern, token))
        
        return processed_tokens
    
    def remove_punctuation(tokens):
        exclude = set(string.punctuation)
        return [token for token in tokens if token not in exclude]
    
    def fix_white_space(tokens):
        return [token for token in tokens if token and not token.isspace()]
    
    sents = nltk.sent_tokenize(text)
    norm_sents = []
    for sent in sents:
        words = nltk.word_tokenize(sent)
        words = process_token(words)
        if remove_punc:
            words = remove_punctuation(words)
        
        words = fix_white_space(words)
        norm_sents.append(' '.join(words))
    
    norm_text = ' '.join(norm_sents)
    if lower_case:
        norm_text = norm_text.lower()
    
    return norm_text

def spacy_tokenize(text, lower_case=False, remove_punc=False):
    def process_token(tokens):
        special = ("-", "£", "€", "¥", "¢", "₹", "\u2212", "\u2014", "\u2013",
                   "/", "~", '"', "'", "\ud01C", "\u201C", "\u2019", "\u201D", "\u2018", "\u00B0")
        pattern = "([{}])".format("".join(special))
        processed_tokens = []
        for token in tokens:
            token = token.replace("''", '" ').replace("``", '" ')
            processed_tokens.extend(re.split(pattern, token))
        
        return processed_tokens
    
    def remove_punctuation(tokens):
        exclude = set(string.punctuation)
        return [token for token in tokens if token not in exclude]
    
    def fix_white_space(tokens):
        return [token for token in tokens if token and not token.isspace()]
    
    word_docs = spacy_nlp(text)
    words = [word.text for word in word_docs]
    words = process_token(words)
    if remove_punc:
        words = remove_punctuation(words)

    words = fix_white_space(words)
    
    norm_text = ' '.join(words)
    if lower_case:
        norm_text = norm_text.lower()
    
    return norm_text

def get_word_span(context, text, start):
    end = start + len(text)
    span_length = len(text.split(" "))
    span_end = len(context[:end].split(" ")) - 1
    span_start = span_end - span_length + 1
    
    return span_start, span_end

def preprocess(file_name):
    if not os.path.exists(file_name):
        raise FileNotFoundError("file not found")
    
    processed_data_list = []
    with open(file_name, "r") as file:
        json_content = json.load(file)
        for article in json_content["data"]:
            for paragraph in article["paragraphs"]:
                context = paragraph["context"].strip()
                norm_context = spacy_tokenize(context)
                for qa in paragraph["qas"]:
                    qa_id = qa["id"]
                    question = qa["question"].strip()
                    norm_question = spacy_tokenize(question)
                    
                    processed_data = {
                        "id": qa_id,
                        "question": norm_question,
                        "context": norm_context,
                        "answers": []
                    }
                    
                    for answer in qa["answers"]:
                        answer_text = answer["text"].strip()
                        norm_answer_text = spacy_tokenize(answer_text)
                        
                        answer_start = answer["answer_start"]
                        norm_answer_start = len(spacy_tokenize(context[:answer_start]))
                        norm_answer_start = norm_answer_start + 1 if norm_answer_start > 0 else norm_answer_start 
                        
                        norm_answer_word_start, norm_answer_word_end = get_word_span(norm_context,
                            norm_answer_text, norm_answer_start)
                        
                        processed_data["answers"].append({
                            "text": norm_answer_text,
                            "start": norm_answer_word_start,
                            "end": norm_answer_word_end
                        })
                    
                    processed_data_list.append(processed_data)
    
    return processed_data_list

def output_to_json(data_list, file_name):
    with open(file_name, "w") as file:
        data_json = json.dumps(data_list, indent=4)
        file.write(data_json)

def output_to_plain(data_list, file_name):
    with open(file_name, "wb") as file:
        for data in data_list:
            for answer in data["answers"]:
                data_plain = "{0}\t{1}\t{2}\t{3}\t{4}|{5}\r\n".format(data["id"], data["question"],
                    data["context"].replace("\n", " "), answer["text"], answer["start"], answer["end"])
                file.write(data_plain.encode("utf-8"))

def output_to_split(data_list, file_prefix):
    with open("{0}.question".format(file_prefix), "wb") as q_file, open("{0}.context".format(file_prefix), "wb") as c_file, open("{0}.answer_text".format(file_prefix), "wb") as at_file, open("{0}.answer_span".format(file_prefix), "wb") as as_file:
        for data in data_list:
            for answer in data["answers"]:
                q_data_plain = "{0}\r\n".format(data["question"])
                q_file.write(q_data_plain.encode("utf-8"))
                c_data_plain = "{0}\r\n".format(data["context"].replace("\n", " "))
                c_file.write(c_data_plain.encode("utf-8"))
                at_data_plain = "{0}\r\n".format(answer["text"])
                at_file.write(at_data_plain.encode("utf-8"))
                as_data_plain = "{0}|{1}\r\n".format(answer["start"], answer["end"])
                as_file.write(as_data_plain.encode("utf-8"))

def main(args):
    processed_data = preprocess(args.input_file)
    if (args.format == 'json'):
        output_to_json(processed_data, args.output_file)
    elif (args.format == 'plain'):
        output_to_plain(processed_data, args.output_file)
    elif (args.format == 'split'):
        output_to_split(processed_data, args.output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    main(args)