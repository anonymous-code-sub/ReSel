import os
from collections import OrderedDict
from site import removeduppaths

# Output format: {"section" : [paragraphs]}
def parse_text_to_json(text_file):
    json_file = OrderedDict()
    key = ''
    for line in text_file:
        if 'section: ' in line:
            key = line[9:-1]
            json_file[key] = []
        elif key:
            json_file[key].append(line[:-1].lower())
    return json_file

def parse_tuple_to_dict(tuple_file):
    tuple_dict = {}
    for line in tuple_file:
        line = line[:-1]
        filename, tuple_strings = line.split('\t')
        filename = os.path.splitext(filename)[0]
        tuple_string_list = tuple_strings.split('$')
        tuple_list = [item.lower().split('#') for item in tuple_string_list]
        tuple_dict[filename] = tuple_list
    return tuple_dict

def reconstruct_table(table_json):
    caption_list = []
    reconstructed_tables = []
    for table in table_json:
        caption_list.append(table['caption'])

        row = remove_duplicate(table['rows'])
        col = remove_duplicate(table['columns'])
        current_table = [['[NONE]' for _ in range(len(col))] for _ in range(len(row))]
        
        table_cells = table['numberCells']
        bolded_cells = []
        for table_cell in table_cells:
            if table_cell['isBolded']:
                bolded_cells.append(table_cell)
            else:
                for current_row in table_cell['associatedRows']:
                    for current_col in table_cell['associatedColumns']:
                        if current_row in row and current_col in col:
                            current_table[row.index(current_row)][col.index(current_col)] = table_cell['number']
        # Make sure bolded cells are included in the table
        for bolded_cell in bolded_cells:
            for current_row in bolded_cell['associatedRows']:
                    for current_col in bolded_cell['associatedColumns']:
                        if current_row in row and current_col in col:
                            current_table[row.index(current_row)][col.index(current_col)] = bolded_cell['number']
        # Add row and col to table
        for i in range(len(row)):
            current_table[i].insert(0, row[i])
        current_table.insert(0, [''] + col)
        reconstructed_tables.append(current_table)
    return reconstructed_tables, caption_list

def remove_duplicate(input_list):
    sets = set()
    result_list = []
    for item in input_list:
        if not item in sets:
            result_list.append(item)
            sets.add(item)
    return result_list

def flatten_ary(input_list):
    table_lens = []
    result_list = []
    for current_list in input_list:
        result_list.append(current_list)
    return result_list