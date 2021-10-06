import pytest
import sys

from eval_pubmed import (
    load_complexity_classes,
    load_xml_files,
    print_line,
    eval_data,
    ParsingMethod
)


def get_str(tp, fn, fp):
    return f"TP:{tp} FN:{fn} FP:{fp}"

def evaluate(method, gt_dir, res_dir, multivariant, filenames=[]):

    eval_log = sys.stdout

    name_pattern = f"PMC*.{ParsingMethod.get_extension(method)}"

    tuples_to_use = load_complexity_classes("mavo_table_classes.csv", [0,1,2], eval_log=eval_log) 

    res_files = load_xml_files(res_dir, "PMC*.xml", is_gt=False, multivariant=multivariant, method=method, 
        tuples_to_use=tuples_to_use, eval_log=eval_log, filenames=filenames)
    
    print_line(n=100, eval_log=eval_log)
    
    gt_files = load_xml_files(gt_dir, "PMC*.xml", is_gt=True, record_overlap=False, 
        tuples_to_use=tuples_to_use, eval_log=eval_log, filenames=filenames)

    eval_res = eval_data(gt_files, res_files, res_multivariant=True, ignore_fp=False, eval_log=eval_log)

    result = get_str(tp=eval_res['TP'], fn=eval_res['FN'], fp=eval_res['FP'])
    
    return result


def test_eval_gt():
    
    filenames = ["PMC481073", "PMC554100", "PMC1174872"]
    expected_output = "TP:909 FN:0 FP:0"
        
    actual_output = evaluate(ParsingMethod.ICDAR, "gt", "gt", True, filenames)

    assert actual_output == expected_output


def test_eval_abby():
    
    filenames = ["PMC481073", "PMC554100", "PMC1174872"]
    expected_output = "TP:479 FN:430 FP:352"
        
    actual_output = evaluate(ParsingMethod.Abbyy, "gt", "res/abbyy", True, filenames)

    assert actual_output == expected_output


def test_eval_tab_iais():
    
    filenames = ["PMC481073", "PMC554100", "PMC1174872"]
    expected_output = "TP:111 FN:798 FP:10"
        
    actual_output = evaluate(ParsingMethod.ICDAR, "gt", "res/tab.iais", True, filenames)

    assert actual_output == expected_output


def test_eval_tabby():
    
    filenames = ["PMC481073", "PMC554100", "PMC1174872"]
    expected_output = "TP:648 FN:261 FP:350"
        
    actual_output = evaluate(ParsingMethod.ICDAR, "gt", "res/tabby", True, filenames)

    assert actual_output == expected_output


def test_eval_tabula_stream():
    
    filenames = ["PMC481073", "PMC554100", "PMC1174872"]
    expected_output = "TP:0 FN:909 FP:0"
        
    actual_output = evaluate(ParsingMethod.TabulaJson, "gt", "res/tabula_stream", True, filenames)

    assert actual_output == expected_output


def test_eval_tabula_lattice():
    
    filenames = ["PMC481073", "PMC554100", "PMC1174872"]
    expected_output = "TP:0 FN:909 FP:0"
        
    actual_output = evaluate(ParsingMethod.TabulaJson, "gt", "res/tabula_lattice", True, filenames)

    assert actual_output == expected_output

