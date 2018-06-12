# -*- coding: utf8 -*-

import os
import os.path as osp
import sys
import itertools
import json

from jacinle.cli.git import get_git_revision_hash


def escape_desc_name(filename):
    basename = osp.basename(filename)
    if basename.endswith('.py'):
        basename = basename[:-3]
    name = basename.replace('.', '_')
    return name


def ensure_path(path):
    os.makedirs(path, exist_ok=True)
    return path


def format_meters(caption, meters_kv, kv_format, glue):
    log_str = [caption]
    log_str.extend(itertools.starmap(kv_format.format, sorted(meters_kv.items())))
    return glue.join(log_str)


def pretty_json_dumps(value):
    return json.dumps(value, sort_keys=True, indent=4, separators=(',', ': '))


def dump_metainfo(metainfo=None, **kwargs):
    if metainfo is None:
        metainfo = {}
    metainfo.update(kwargs)
    metainfo.setdefault('_cmd', ' '.join(sys.argv))
    metainfo.setdefault('_git', get_git_revision_hash())
    return pretty_json_dumps(metainfo)
