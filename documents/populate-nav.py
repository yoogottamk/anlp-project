"""
Populate `nav` section of mkdocs
"""

import re
import os
from typing import List, Tuple
import yaml


def key_exists(key: str, list_keys: List) -> Tuple[int, bool]:
    """
    Finds a dict which has `key` defined from a list `list_keys`

    Args:
        key: the key to find
        list_keys: list of dicts

    Returns:
        index of key if found, whether the key was found
    """
    for i, x in enumerate(list_keys):
        if key in x:
            return i, True

    return -1, False


def get_nav_links():
    """
    Generates navigation tree for mkdocs
    """
    nav_links = []

    for module in ["anlp_project"]:
        for dirp, _, files in os.walk(module):
            md_files = []
            # generate {"title": "path"} for each file
            for md_f in files:
                md_files.append({re.sub(r"\.md$", "", md_f): os.path.join(dirp, md_f)})

            # find the correct place to insert it
            curr = nav_links
            for path in dirp.split("/"):
                idx, found = key_exists(path, curr)
                if not found:
                    curr.append({path: []})
                    idx = len(curr) - 1
                curr = curr[idx][path]

            # insert the files
            curr += md_files

    return nav_links


def populate_nav():
    """
    Populates the `nav` section in mkdocs
    """
    with open("../mkdocs.yml", "r") as f:
        conf = yaml.safe_load(f.read())

    NAV_KEY = "API Reference"
    nav_links = {NAV_KEY: get_nav_links()}

    idx, found = key_exists(NAV_KEY, conf["nav"])

    if not found:
        conf["nav"].append(nav_links)
    else:
        conf["nav"][idx] = nav_links

    with open("../mkdocs.yml", "w") as f:
        yaml.safe_dump(conf, f)


if __name__ == "__main__":
    populate_nav()
