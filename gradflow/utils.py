import csv
from typing import List, Dict

def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        return [row for row in reader]
