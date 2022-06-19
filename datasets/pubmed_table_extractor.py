"""
by Yingjun Mou ymou32@gatech.edu, Feb.2022
"""
import os.path
import numpy as np
import pandas as pd
import semanticscholar as sch
import glob
import re
import camelot
from tqdm import tqdm
import time
import urllib.request as libreq
import urllib.request
import tabula
from bs4 import BeautifulSoup
from pymed import PubMed
import requests
import os
import json


def download_pdf(download_url, save_dir, filename):
    response = urllib.request.urlopen(download_url)
    os.makedirs(save_dir, exist_ok=True)
    file = open(save_dir + filename + ".pdf", 'wb')
    file.write(response.read())
    file.close()


def download_pdf_pubmed(pubmed, pmid, save_dir):
    try:
        art = list(pubmed._getArticles(article_ids= pmid))[0].toDict()
    except:
        print("Inaccessible: ", pmid)
        return

    # Given DOI, we can access pdf using S2 API
    paper = sch.paper(str(art['doi']), timeout=99999)
    if 'url' in paper.keys():
        url = paper['url']
        print('URL FOR THIS PAPER: ', url)
    # Exit as there is no url
    else:
        return
    if 'paperId' in paper.keys():
        paper_id_link ='https://pdfs.semanticscholar.org/' + paper['paperId'][:4] + '/' + paper['paperId'][4:] + '.pdf'
        print('EXPECTED PAPER_ID LINK FOR THIS PAPER: ', paper_id_link)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36',
    }
    res = requests.get(url, headers=headers)
    res.raise_for_status()
    bs = BeautifulSoup(res.text, 'html.parser')
    # ****************************
    # IMPORTANT FOR PDF QUALITY
    # ****************************
    results = bs.findAll("a", {"data-heap-id": "paper_link_target", "data-heap-direct-pdf-link": "true"})
    href = ''
    href1 = ''
    if len(results) == 0:
        results = bs.findAll("a", {"data-heap-id": "paper_link_target", "data-heap-unpaywall-link": "true"})
    if len(results)>0:
        href = results[0].attrs.get("href")
        print('HREF FOR THIS PAPER: ', href)

    # Otherwise, we need to look into the secondary source
    else:
        results2 = bs.findAll("a", {"data-heap-id": "paper_link_target", "data-heap-primary-link": "true"})
        if len(results2)>0:
            href1 = results2[0].attrs.get("href")
            print('1st HREF FOR THIS PAPER: ', href1)
            res2 = requests.get(href1, headers=headers)
            res2.raise_for_status()
            bs2 = BeautifulSoup(res2.text, 'html.parser')
            results3 = bs2.findAll("a", {"data-article-pdf": "true"})
            if len(results3)>0:
                href = results3[0].attrs.get("href")
            # Example:
            # 15150562 : 'https://www.nature.com/articles/6601819.pdf' and '/articles/6601819.pdf'
            # however, it has redirect so host name is not nature...
            if '.com' not in href:
                href = href1.split('/')[0] + '/' + '/'.join(href1.split('/')[1:]) + href
            print('2nd HREF FOR THIS PAPER: ', href)



    # *****************************************************************************
    # Another possible link is 'https://pdfs.semanticscholar.org/' + paper['paperId'][:4] + '/' + paper['paperId'][4:] + '.pdf'
    # *****************************************************************************
    pubmed_pdf_path = save_dir + pmid + '/'
    try:
        download_pdf(href, pubmed_pdf_path, pmid)
    except:
        if 'paperId' in paper.keys():
            try:
                if href1 != '':
                    try:
                        download_pdf(href1 + '.pdf', pubmed_pdf_path, pmid)
                        print('TRY APPENDING .PDF FROM THE LANDING PAGE: ', href1 + '.pdf')
                    except:
                        download_pdf(paper_id_link, pubmed_pdf_path, pmid)
                        print('TRY SPECULATED PDF LINK: ', paper_id_link)
                else:
                    download_pdf(paper_id_link, pubmed_pdf_path, pmid)
                    print('TRY SPECULATED PDF LINK: ', paper_id_link)
            except:
                print("Inaccessible: ", pmid)


def prune_df(df, perc=70.0):
    df = df.replace('', np.nan)
    finish_pruning = False
    prev_shape = df.shape
    while not finish_pruning:
        # Delete row with more than 50% nan  # how="all"
        df = df.dropna(axis=0, thresh=int(((100 - perc) / 100) * df.shape[1] + 1))
        # Delete col with more than 50% nan  # how="all"
        df = df.dropna(axis=1, thresh=int(((100 - perc) / 100) * df.shape[0] + 1))
        if prev_shape == df.shape:
            finish_pruning = True
        prev_shape = df.shape

    df.reset_index(drop=True, inplace=True)
    df.columns = [str(i) for i in range(df.shape[1])]

    if df.empty:
        nan_perc = None
    else:
        count_nan = sum(df.isna().sum())
        nan_perc = float(count_nan) / float(len(df))  # 0.5 as threshold

    return df, nan_perc


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


# This method saves the tabular data from a list of pdfs to csv, return the number of csv files
def pdf2csv(pdf_list, paper_folder, s2id):
    successuful_csv_from_pdf = 0
    if len(pdf_list) > 0:
        csv_id = 0
        for pdf_filename in pdf_list:
            # First fix the pdf script using MuPDF - mutool
            temp_name = pdf_filename.replace('.pdf', '_temp.pdf')
            fix_pdf_command = 'gs -o ' + temp_name + ' -sDEVICE=pdfwrite -dPDFSETTINGS=/prepress -dQUIET ' + pdf_filename  # new name then old name
            os.system(fix_pdf_command)
            os.replace(temp_name, pdf_filename)
            with open(pdf_filename, 'rb') as f: # open pdf as binary
                table_dfs = []
                try:
                    tables_stream = camelot.read_pdf(f, pages='all', flavor='stream', strip_text=' .\n') # , suppress_stdout=True row_tol=2, column_tol=11 table_areas=[x0,y0,x1,y1] table_regions
                    dfs1 = [table.df for table in tables_stream]
                    table_dfs += dfs1
                except:
                    pass

                try:
                    tables_lattice = camelot.read_pdf(f, pages='all', flavor='lattice', strip_text=' .\n')
                    dfs2 = [table.df for table in tables_lattice]
                    table_dfs += dfs2
                except:
                    pass

                try:
                    tables_tabula_dfs = tabula.read_pdf(f, pages='all', guess=True, multiple_tables=True, pandas_options={'header': None}) # encoding="utf-8", area=tables[0]._bbox
                    table_dfs += tables_tabula_dfs
                except:
                    pass

                if len(table_dfs):
                    csv_path = paper_folder + str(s2id) + '/unpack/csv/'
                    os.makedirs(csv_path, exist_ok=True)

                    valid_dfs = filter_df(table_dfs)
                    for df in valid_dfs:
                        print("=============================================")
                        print(df)
                        print("=============================================")
                        filename = str(csv_id) + '.csv'
                        df.to_csv(csv_path + filename, index=False)
                        csv_id += 1
                        successuful_csv_from_pdf += 1

    return successuful_csv_from_pdf


# This function take a list of df as input and
# (1) fix the malformated dataframe by deleting some of the abnormal rows and col (2) delete overall abnormal df
# (3) delete duplicate dfs
def filter_df(lst_df):
    candidate = []
    for df in lst_df:
        # print debugging
        # print("============ BEFORE FILTERED=================")
        # print(df)
        # print("============ BEFORE FILTERED=================")

        # 1. check not empty and percentage of nan cell
        if df is None:
            continue
        df, nan_perc = prune_df(df, 70.0)
        # if not df.empty and nan_perc <= 0.5:
        if df.empty or nan_perc > 0.5:
            continue

        # 2. check the proportion of cells with exceeding length(>30 chars) of content
        bad_cell = 0
        for rowIndex, row in df.iterrows():  # iterate over rows
            count = sum(1 for columnIndex, value in row.items() if len(str(value)) > 30)
            bad_cell += count
            # for columnIndex, value in row.items():
            #     cell_string = str(value)
            #     cell_words = cell_string.split(' ')
            #     count = sum(1 for w in cell_words if len(w) > 30)
            #     bad_cell += count
        if float(bad_cell) / float(df.shape[0] * df.shape[1]) > 0.3:
            continue

        # 3. check there is only one col or one row
        if df.shape[0] < 2 or df.shape[1] < 2:
            continue

        if not any(df.equals(d) for d in candidate):
            candidate.append(df)

    return candidate


# ===============================================================================
# # MAIN PROGRAM
# ===============================================================================

start_time = time.time()
pd.set_option('display.max_columns', None)  # display entire df


# 1. Use PubMed ID to search and download pdf
paper_folder = 'paper_0811/'

nopdf_id = []
NUM_PDF = 0
NUM_CSV_PDF = 0
curr_NUM_CSV_PDF = 0
owd = os.getcwd()
pubmed = PubMed(tool="PubMedSearcher", email="ymou32@gatech.edu")
# inaccessible_pmid = dict()
# missing_list = ['26306257', '26604863', '26605007', '2883711', '3001045', '3111215', '4049179', '4070608']

# Load PubMed IDs from file
pmid_file = open('../2_Ref/N-ary_RE/naacl2019/data/pmid_lists/init_pmid_list.txt', 'r')
pmid_list = pmid_file.readlines()
pmid_list = [pmid.strip('\n') for pmid in pmid_list]
print("Total number ofo pmid in list: ", len(pmid_list))
print("Total number ofo pmid in set: ", len(set(pmid_list)))


for pmid in tqdm(pmid_list[:]):
    # if pmid in missing_list:
    print('\n' + "=============================================")
    print("================ A NEW ID ===================")
    print("=============================================")
    print("CURRENTLY PROCESSING ID: ", pmid)  # ADD ONE MORE TRY EXCEPT FOR CONN FAILED!
    #######################################
    # DOWNLOAD PDF FOR EACH ID
    #######################################

    download_pdf_pubmed(pubmed, pmid, paper_folder)

    pdf_path = paper_folder + str(pmid) + '/' + '*.pdf'
    unpack_path = paper_folder + str(pmid) + '/unpack/'
    os.makedirs(unpack_path, exist_ok=True)

    pdf_list = glob.glob(pdf_path)  # the length could be zero, no latex file.
    print("TOTAL PDF FILE FOUND: ", len(pdf_list))

    #######################################
    # IF THE PDF IS FOUND
    #######################################

    if len(pdf_list) > 0:
        print('================ START TO EXTRACT TABLE ===================')
        curr_NUM_CSV_PDF = pdf2csv(pdf_list, paper_folder, pmid)
        print('================ MANAGED TO EXTRACT ' + str(curr_NUM_CSV_PDF) + ' TABLES ===================')
        NUM_CSV_PDF += curr_NUM_CSV_PDF
        NUM_PDF += len(pdf_list)
    ## DEBUG - Finding the PDF-related s2id
    if curr_NUM_CSV_PDF <0:
        nopdf_id.append(pmid_list)


# ==============================================================================================================
# Manually record those inaccessible ids from the log output during the execution of download_pdf_pubmed
# ==============================================================================================================

# with open("inaccessible_read.txt", 'r') as f:
#     lines = f.readlines()
#     for l in lines:
#         missing_list.append(l.split(':')[0])
#
# for missing_idx in tqdm(missing_list):
#     pmid = pmid_list[int(missing_idx)]
#
# for pmid in tqdm(pmid_list[:]): # if there are more than 3, it will raise error for too many requests
#     download_pdf_pubmed(pmid, paper_dir)

# print('Total number of inaccessible IDs: ', len(inaccessible_pmid))
# with open("inaccessible.txt", 'w') as f:
#     for key, value in inaccessible_pmid.items():
#         f.write('%s:%s\n' % (key, value))


print('Total number of IDs, including those inaccessible, that don\'t have pdfs: ', len(nopdf_id))
with open("nopdf_id.txt", 'w') as f:
    for id in nopdf_id:
        f.write('%s\n' % id)


print('**********************************')
print('************ DONE! ***************')
print('NUM OF ID FROM PUBMED DATA: ', len(pmid_list))
print('NUM OF INACCESSIBLE ID: ', len(nopdf_id))
print('NUM OF PDF FILES PROCESSED: ', NUM_PDF)
print('NUM OF CSV FILES SUCCESSFULLY CREATED FROM PDF: ', NUM_CSV_PDF)
print('TOTAL TIME: ' + (time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))))
print('**********************************')


# ===============================================================================
# # Delete unnecessary files and subfolders, only keeping the csv folder
# ===============================================================================

# print("Scanning files to delete: ", len(s2id_list))
# retain = ["csv", "caption.txt"]
# for s2id in tqdm(s2id_list):
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


# ===============================================================================
# Check the number of folder without csv
# ===============================================================================
owd = os.getcwd() # D:\2_Academics\3_RESEARCH\Research_Multi-Modal\3_Code

# Load PubMed IDs from file
pmid_file = open('../2_Ref/N-ary_RE/naacl2019/data/pmid_lists/init_pmid_list.txt', 'r')
pmid_list = pmid_file.readlines()
pmid_list = [pmid.strip('\n') for pmid in pmid_list]
print("Total number ofo pmid in list: ", len(pmid_list))
print("Total number ofo pmid in set: ", len(set(pmid_list)))

paper_folder = 'paper_0811/'
list_id_no_csv = []
NUM_NO_CSV = 0
NUM_NO_PDF = 0
NUM_CSV = 0

os.chdir(owd)
for pmid in tqdm(pmid_list):
    # example: 23308140
    pdf_path = paper_folder + str(pmid) + '/*.pdf' # don't forget the slash!!!
    pdf_list = glob.glob(pdf_path)
    if len(pdf_list)==0:
        print("NO PDF: ", pmid)
        NUM_NO_PDF += 1
    doc_path = paper_folder + str(pmid) + '/unpack/csv'
    csv_path = doc_path + '/*.csv'
    csv_list = glob.glob(csv_path)
    if os.path.exists(doc_path) and len(csv_list)>0:
        NUM_CSV += len(csv_list)
    else:
        NUM_NO_CSV += 1


print('TOTAL NUMBER OF ID= ', len(pmid_list))
print('NUM_HAS_PDF= ', len(pmid_list) - NUM_NO_PDF)
print('NUM_NO_PDF= ', NUM_NO_PDF)
print('NUM_NO_CSV= ', NUM_NO_CSV)
print('NUM OF SUCCESSFULLY EXTRACTED CSV= ', NUM_CSV)
print('AVG NUM OF EXTRACTED CSV FROM VALID ID= ', float(NUM_CSV)/float(len(pmid_list) - NUM_NO_PDF))


# ===============================================================================
# Count pubmed doc-level data
# ===============================================================================

# Load PubMed IDs from file
doc_dev_file = open('../2_Ref/N-ary_RE/naacl2019/data/examples/document/ds_train_dev.txt', 'r')
doc_dev_list = doc_dev_file.readlines()
print(type(doc_dev_list[0]))
doc_dev_list = [json.loads(doc)['pmid'] for doc in doc_dev_list]

NUM_ID_DOC = len(set(doc_dev_list))
NUM_ID_DOC_TABLE = 0
paper_folder = 'paper_0811/'

for pmid in tqdm(list(set(doc_dev_list))):
    doc_path = paper_folder + str(pmid) + '/unpack/csv'
    csv_path = doc_path + '/*.csv'
    csv_list = glob.glob(csv_path)
    if os.path.exists(doc_path) and len(csv_list) > 0:
        NUM_ID_DOC_TABLE += 1

print('TOTAL NUMBER OF ID WITH DOC-LEVEL DATA= ', NUM_ID_DOC)
print('TOTAL NUMBER OF DOC-LEVEL DATA INSTANCES= ', len(doc_dev_list))
print('NUM_ID_DOC_TABLE= ', NUM_ID_DOC_TABLE)