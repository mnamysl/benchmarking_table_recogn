import pytest
import sys

from eval_pubmed import (
    load_complexity_classes,
    load_xml_files,
    print_line,
    eval_data,
    ParsingMethod
)



def evaluate(method, gt_dir, res_dir, multivariant, filenames=[]):

    eval_log = sys.stdout

    name_pattern = f"PMC*.{ParsingMethod.get_extension(method)}"

    tuples_to_use = load_complexity_classes("mavo_table_classes.csv", [0,1,2], eval_log=eval_log) 

    res_files = load_xml_files(res_dir, "PMC*.xml", is_gt=False, multivariant=multivariant, method=method, 
        tuples_to_use=tuples_to_use, eval_log=eval_log, filenames=filenames)
    
    print_line(n=100, eval_log=eval_log)
    
    gt_files = load_xml_files(gt_dir, "PMC*.xml", is_gt=True, record_overlap=False, 
        tuples_to_use=tuples_to_use, eval_log=eval_log, filenames=filenames)

    result = eval_data(gt_files, res_files, res_multivariant=True, eval_log=eval_log)
    
    return result


def test_evaltabiais():
    
    filenames = ["PMC481073", "PMC554100", "PMC1174872"]
    expected_output = ""
        
    actual_output = evaluate(ParsingMethod.ICDAR, "gt", "res/tab.iais", True, filenames)

    assert actual_output == expected_output


