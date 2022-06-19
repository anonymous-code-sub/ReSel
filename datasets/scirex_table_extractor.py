# %%
import os.path
from xml.etree.ElementTree import ElementTree

import TexSoup
import numpy as np
import pandas as pd
import semanticscholar as sch
import unicodeit
import arxiv
import patoolib
import glob
import re
import camelot
import json
from tqdm import tqdm
import time
import urllib.request as libreq
import shutil
import urllib.request
import chardet
import matplotlib.pyplot as plt
import tabula
import PyPDF2
import fitz

start_time = time.time()
pd.set_option('display.max_columns', None)  # display entire df

# %%
# start_time = time.time()
# pd.set_option('display.max_columns', None)  # display entire df

# %%

# ===============================================================================
# # MAIN PROGRAM
# ===============================================================================

paper_folder = 'paper_0831_scirex_withPDF/'
owd = os.getcwd()


# %%
# TEST id for checking reference

# Extract all s2id as a list from SciRex data
def s2id_extraction(json_file):
    id_list = []
    inaccessible_dict = dict()
    with open(json_file) as f:
        # each line in json file is one instance with one s2id
        for line in f:
            data = json.loads(line)
            id_list.append(data['doc_id'])
            if data['doc_id'] == '0c278ecf472f42ec1140ca2f1a0a3dd60cbe5c48' or data['doc_id'] == '1a6b67622d04df8e245575bf8fb2066fb6729720':
                # extract the first 25 words as search query
                inaccessible_dict[data['doc_id']] = ' '.join(data['words'][:25])

    return id_list, inaccessible_dict


# Check if the current data has "Table [reference]" of "Figure [reference]"
def list_id_with_ref(json_file):
    ref_id_list = []
    # inaccessible_dict = dict()
    with open(json_file) as f:
        # each line in json file is one instance with one s2id
        for line in f:
            data = json.loads(line)
            text = data['words']
            for i in range(len(text) -3):
                if text[i] == "Table" or text[i] == "Figure":
                    if text[i+1 : i+4] == ["[", "reference", "]"]:
                        ref_id_list.append(data['doc_id'])
                        break

    return ref_id_list

# Check if the current data has "Table [reference]" of "Figure [reference]" and extract its context
def ref_dict_from_json(json_file):

    table_words = ['Table', 'Tables', 'table', 'tables']
    figure_words = ['Figure', 'Figures', 'Fig', 'figure', 'figures', 'fig'] # 'Fig', '.'
    section_words = ['Section', 'Sec'] # 'Sec', '.'
    equation_words = ['Equation', 'Eq']

    ref_id_dict = dict()
    # inaccessible_dict = dict()
    with open(json_file) as f:
        # each line in json file is one instance with one s2id
        for line in f:
            data = json.loads(line)
            curr_id = data['doc_id']
            # ref_id_dict[curr_id] = {"Table": [], "Figure": [], "Paper": []}
            ref_id_dict[curr_id] = []
            text = data['words']

            for i in range(len(text) -1):
                if text[i] == "[" and text[i+1] == "reference" and text[i+2] == "]":
                    if i == 0:
                        ref_id_dict[curr_id].append(("Paper", "Unknown"))
                    elif text[i-1] in table_words:
                        context = text[i+3 : i+8] if i+8 <= len(text) else text[i+3 :]
                        context = ''.join(context)
                        ref_id_dict[curr_id].append(("Table", context))
                    elif text[i - 1] in figure_words:
                        context = text[i+3 : i+8] if i+8 <= len(text) else text[i+3 :]
                        context = ''.join(context)
                        ref_id_dict[curr_id].append(("Figure", context))
                    elif text[i - 1] == ".":
                        if i>1:
                            if text[i - 2] in figure_words:
                                context = text[i + 3: i + 8] if i + 8 <= len(text) else text[i + 3:]
                                context = ''.join(context)
                                ref_id_dict[curr_id].append(("Figure", context))
                            if text[i-1] in table_words:
                                context = text[i+3 : i+8] if i+8 <= len(text) else text[i+3 :]
                                context = ''.join(context)
                                ref_id_dict[curr_id].append(("Table", context))
                    else:
                        # elif text[i - 1] not in table_words and text[i - 1] not in figure_words:
                        ref_id_dict[curr_id].append(("Paper", "Unknown"))

    return ref_id_dict


# This method is used when sometimes the S2 API fails to retrieve source file even the s2id is valid
def arxivid_from_title(title):
    # Example:
    # title = 'Image Restoration Using Very Deep Convolutional Encoder-Decoder Networks with Symmetric Skip Connections'
    # None: any non-alphanumeric char need to be replaced
    title = re.sub("[^0-9a-zA-Z]+", "+", title)

    # a sample query:
    # 'http://export.arxiv.org/api/query?search_query=all:electron&start=0&max_results=1'
    arxiv_url = 'http://export.arxiv.org/api/query?search_query=all:' + title + '&start=0&max_results=1'
    with libreq.urlopen(arxiv_url) as url:
        xml_str = url.read()

    if xml_str != '':
        items = xml_str.decode("utf-8").split('\n')
        for item in items:
            if '<id>' in item and '</id>' in item and 'http://arxiv.org/abs/' in item:
                arxiv_id = item.replace('<id>', '').replace('</id>', '').replace('http://arxiv.org/abs/', '').replace(
                    ' ', '')
                return arxiv_id


def isRomanNum(numeral):
    """Controls that the userinput only contains valid roman numerals"""
    numeral = numeral.upper()
    validRomanNumerals = ["M", "D", "C", "L", "X", "V", "I", "(", ")"]
    valid = True
    for letters in numeral:
        if letters not in validRomanNumerals:
            print("Sorry that is not a valid roman numeral")
            print("INPUT IS: ", numeral)
            valid = False
            break
    return valid

def romanToInt(s):
    roman = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000,'IV':4,'IX':9,'XL':40,'XC':90,'CD':400,'CM':900}
    k = 0
    num = 0
    while k < len(s):
        if k+1 < len(s) and s[k:k+2] in roman:
            num += roman[s[k:k+2]]
            k += 2
        else:
            num += roman[s[k]]
            k += 1
        return num



# Input the path of pdf, return the string of all_text in pdf (without spaces)
def pdf2text(filename):
    pdffileobj = open(filename, 'rb')
    pdfreader = PyPDF2.PdfFileReader(pdffileobj, strict = False)
    num_pages = pdfreader.numPages

    all_text = ""
    for i in range(num_pages):
        pageobj = pdfreader.getPage(i)
        text = pageobj.extractText()
        all_text += text
    all_text = ''.join(all_text.split('\n'))
    return all_text


# Get the reference number using context (following 5 words or less)
def label_from_context(text, context, typename):
    sub_context = context
    for i in range(len(context)):
        temp = context[ : len(context)-i]
        subcontext_matches = re.finditer(re.escape(temp), text)
        subcontext_positions = [match.start() for match in subcontext_matches] # idx of the start of description
        # if and only if we find the ONLY match for context
        if len(subcontext_positions) > 0:
            sub_context = temp
            break

    context_matches = re.finditer(re.escape(sub_context), text)
    context_positions = [match.start() for match in context_matches]
    if not len(context_positions):
        reference_num = "Unknown"
        return reference_num
    context_position = context_positions[-1]

    typename_matches = re.finditer(typename, text[:context_position])
    typename_positions = [match.start() for match in typename_matches]
    if not len(typename_positions):
        reference_num = "Unknown"
        return reference_num
    start_typename = typename_positions[-1]  # idx of the start of description
    start_num = start_typename + len(typename)
    if not len(typename_positions) or (context_position - start_num) >= 6: # sometimes Table will become Ta-ble
        reference_num = "Unknown"
        # try to fin last letter 'le' for 'Table' or 're' for 'Figure'
        for i in range(len(text[:context_position])):
            if typename == 'Table' and text[context_position-i] == 'e' and text[context_position-i-1] == 'l':
                reference_num = text[context_position-i+1 : context_position]
            if typename == 'Figure' and text[context_position-i] == 'e' and text[context_position-i-1] == 'r':
                reference_num = text[context_position-i+1 : context_position]
            if i == 5:
                break
    else:
        # check the reference_num length is shorter than 6, otherwise it may be wrong
        reference_num = text[start_num : context_position]
        if len(reference_num) >= 6:
            reference_num = "Unknown"

    # if reference is clean and not "Unknown", we convert Roman numbers to Arabic numbers, and minus 1
    if reference_num != "Unknown" and not reference_num.isnumeric():
        if isRomanNum(reference_num):
            reference_num = str(romanToInt(reference_num.upper()))
        else:
            reference_num = "Unknown"

    # Subtract number by 1 to match index
    if reference_num.isnumeric():
        reference_num = str(int(reference_num) - 1)

    return reference_num


ref_list_dev = list_id_with_ref('./dev.jsonl')
ref_list_test = list_id_with_ref('./test.jsonl')
ref_list_train = list_id_with_ref('./train.jsonl')
ref_id_list = ref_list_dev + ref_list_test + ref_list_train
print("There are ", len(ref_id_list), " IDs that actually have refs")

ref_dict_dev = ref_dict_from_json('./dev.jsonl')
ref_list_test = ref_dict_from_json('./test.jsonl')
ref_list_train = ref_dict_from_json('./train.jsonl')
ref_dict_temp = ref_dict_dev | ref_list_test
ref_dict = ref_dict_temp | ref_list_train


# %%
# Separate each id into different file, for better debugging the num of reference

individual_folder = 'individual_scirex/'

for json_file in ['./dev.jsonl', './test.jsonl', './train.jsonl']:
    with open(json_file) as f:
        # each line in json file is one instance with one s2id
        for line in f:
            data = json.loads(line)
            curr_id = data['doc_id']
            output_filename = individual_folder + str(curr_id) + '/' + str(curr_id) + '.json'
            os.makedirs(os.path.dirname(output_filename), exist_ok=True)
            output = dict()
            count = 0
            for i in range(len(data['words'])-2):
                if data['words'][i] == '[' and data['words'][i+1] == 'reference' and data['words'][i+2] == ']':
                    count += 1
            output['count'] = count
            output['words'] = data['words']

            with open(output_filename, "w") as out:
                json.dump(output, out)

# %%
inac_dict = dict()
s2id_list_test, inac_dict_test = s2id_extraction('./test.jsonl') #66
s2id_list_dev, inac_dict_dev = s2id_extraction('./dev.jsonl') #66
s2id_list_train, inac_dict_train = s2id_extraction('./train.jsonl') #306
s2id_list = s2id_list_test + s2id_list_dev + s2id_list_train

inac_dict_dev.update(inac_dict_test)
inac_dict_dev.update(inac_dict_train)
inac_dict = inac_dict_dev


paper_folder = 'paper_0831_scirex_withPDF/'
NUM_UNKNOWN = {"Table": 0, "Figure":0}
with open("ref_dict.json", "w") as ref_dict_outfile:
    for s2id in tqdm(s2id_list[:]):  # [33:34]
        if s2id in ref_id_list:
            print("==================================================================")
            print("==================================================================")
            print("CURRENTLY UPDATING ID: ", s2id)
            print("==================================================================")
            print("==================================================================")
            unpack_path = paper_folder + str(s2id) + '/unpack/'
            # 1. download pdf again
            paper = sch.paper(s2id, timeout=1000)
            pdf_list = []

            # 1a. first try to download the source pdf directtly
            if paper and 'arxivId' in paper and paper['arxivId']:
                # example: 'https://arxiv.org/pdf/1810.04805.pdf'
                source_url = 'https://arxiv.org/pdf/' + str(paper['arxivId']) + '.pdf'
                source_pdf_filename = paper_folder + str(s2id) + '/unpack/' + str(paper['arxivId']) + '.pdf'
                response = urllib.request.urlopen(source_url)
                # No folders for IDs:
                # 2393447b8b0b79046afea1c88a8ed3949338949e
                # 1. Count number of Unknown, try if extend the context, if it helps
                # 2. Check if the context contain [reference], and replace
                source_pdf_file = open(source_pdf_filename, 'wb')
                source_pdf_file.write(response.read())
                source_pdf_file.close()
                pdf_path = unpack_path + '*.pdf'
                pdf_list = glob.glob(pdf_path)  # call this again, because now we may get an extra source pdf file

            # 1b. if cant get the pdf, we try download from zip
            if not len(pdf_list):
                arxiv_id = None
                if not paper:
                    arxiv_id = arxivid_from_title(inac_dict[s2id])
                elif 'arxivId' in paper or 'title' in paper:
                    arxiv_id = paper['arxivId'] if 'arxivId' in paper.keys() and paper['arxivId'] else arxivid_from_title(
                        paper['title'])
                else:
                    # print("\n =============== INACCESSIBLE ID: ", s2id, " ===================")
                    continue
                if not arxiv_id:
                    continue
                arxiv_paper = next(arxiv.Search(id_list=[arxiv_id]).get())  # '1603.09056v2'
                gz_filename = s2id + '.tar'
                tar_path = paper_folder + str(s2id) + '/tar/'
                os.makedirs(tar_path, exist_ok=True)

                arxiv_paper.download_source(dirpath=tar_path, filename=gz_filename)
                os.makedirs(unpack_path, exist_ok=True)
                patoolib.extract_archive(tar_path + gz_filename, outdir=unpack_path, verbosity=-1)

            pdf_path = unpack_path + '*.pdf'
            pdf_list = glob.glob(pdf_path)  # the length could be zero, no latex file.
            print("TOTAL PDF FILE FOUND: ", len(pdf_list))


            # 2. convert all pdfs to text
            doc_text = ""
            unpack_path = paper_folder + str(s2id) + '/unpack/'
            pdf_path = unpack_path + '*.pdf'
            pdf_list = glob.glob(pdf_path)
            if len(pdf_list):
                for pdf in pdf_list:
                    pdf = pdf.replace("\\", "/")
                    try:
                        doc_text += pdf2text(pdf)
                    except:
                        pass
                # 3. get the reference number
                # curr_context_table = ref_dict[s2id]["Table"]
                # curr_context_figure = ref_dict[s2id]["Figure"]

                # print("============= doc_text ==================")
                # print(doc_text)
                # print("============= doc_text ends ==================")

                curr_ref_dict = []
                for item in ref_dict[s2id]:
                    if item[0] == "Table":
                        label_result = label_from_context(doc_text, item[1], "Table")
                        item = (item[0], label_result)
                        curr_ref_dict.append(item)
                        if label_result == "Unknown":
                            NUM_UNKNOWN["Table"] += 1
                    if item[0] == "Figure":
                        label_result = label_from_context(doc_text, item[1], "Figure")
                        item = (item[0], label_result)
                        curr_ref_dict.append(item)
                        if label_result == "Unknown":
                            NUM_UNKNOWN["Figure"] += 1

            print(curr_ref_dict)
            print("Total ref: ", len(curr_ref_dict))
            ref_curr_id_json = json.dumps({s2id: curr_ref_dict})
            ref_dict_outfile.write(ref_curr_id_json + '\n')

print("===================================")
print("==============RESULT===============")
print("===================================")
# print(ref_dict)
print(NUM_UNKNOWN)



# %%

# paper_folder = 'paper_0831_scirex_withPDF/'
# owd = os.getcwd()
#
# # Delete unnecessary files and subfolders, only keeping the csv folder
# print("Scanning files to delete: ", len(s2id_list))
# retain = ["csv", "extracted_img", "caption.txt"]
# for s2id in tqdm(s2id_list):
#     print("Current ID: ", s2id)
#     rel_directory = paper_folder + str(s2id) + '/unpack'
#     os.chdir(owd)
#     os.chdir(rel_directory)
#     # Loop through everything in folder in current working directory
#     for item in os.listdir(os.getcwd()):
#         if item not in retain:  # If it isn't in the list for retaining
#             if os.path.isdir(item):
#                 shutil.rmtree(item, ignore_errors=True)
#             if os.path.isfile(item):
#                 os.remove(item)  # Remove the item


# %%
# # NO csv Folder
# paper_folder = 'paper_0831_scirex_withPDF/'
# list_id_no_csv = []
# NUM_NO_CSV = 0
#
# os.chdir(owd)
# print("Checking ID with csv missing: ", len(s2id_list))
# for s2id in s2id_list:
#     doc_path = paper_folder + str(s2id) + '/unpack/csv'
#     if not os.path.exists(doc_path):
#         list_id_no_csv.append(s2id)
#         NUM_NO_CSV += 1
# print('NUM_NO_CSV= ', NUM_NO_CSV)
# print(list_id_no_csv)
