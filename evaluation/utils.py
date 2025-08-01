# evaluation/utils.py
import re
import ast
import numpy as np
import logging

def escaped_(data: str):
    if "'" in data:
        return re.sub(r"(?<=\w)(')(?=\w)", r"\"", data)
    return re.sub(r'(?<=\w)(")(?=\w)', r"\'", data)

def parse_key_values(out_str):
    regex = re.compile(r"""['"](.*?)['"]\s*:\s*['"]*(.*?)['"]*\s*[,}]""")
    return regex.findall(out_str)

def recreate(out_str):
    kvs = parse_key_values(out_str)
    return {kv[0].replace("\\", ""): kv[1] for kv in kvs}

def clean_output(id, out_str):
    try:
        if isinstance(out_str, float) and np.isnan(out_str):
            return {}
        
        out_str = str(out_str).strip().split("\n")[0]
        out_str = out_str.replace("Stop Here", "").strip()
        out_str = out_str.replace("'s", "s")
        out_str = escaped_(out_str)
        return ast.literal_eval(out_str)
        
    except Exception:
        b_str = out_str
        out_str = recreate(out_str)
        if not out_str:
            logging.warning(f"Could not parse string for id {id}: {b_str}")
        return out_str