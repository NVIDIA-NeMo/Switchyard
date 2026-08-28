# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Vendored from aider-chat (https://github.com/Aider-AI/aider)
# Licensed under the Apache License, Version 2.0
# Original file: aider/repomap.py
#
# Adaptations for craft-bench:
# - Removed aider.dump, aider.waiting imports (replaced with logging / no-op spinner)
# - Replaced self.io.read_text() with Path.read_text()
# - Replaced self.io.tool_output/warning/error with logging
# - Replaced self.main_model.token_count() with len(text) // 4 heuristic

from __future__ import annotations

import logging
import math
import os
import shutil
import sqlite3
import time
import warnings
from collections import Counter, defaultdict, namedtuple
from importlib import resources
from pathlib import Path

from diskcache import Cache
from grep_ast import TreeContext, filename_to_lang
from pygments.lexers import guess_lexer_for_filename
from pygments.token import Token
from tqdm import tqdm
from tree_sitter import Query

from .aider_special import filter_important_files

# tree_sitter is throwing a FutureWarning
warnings.simplefilter("ignore", category=FutureWarning)
from grep_ast.tsl import USING_TSL_PACK, get_language, get_parser  # noqa: E402

logger = logging.getLogger(__name__)

Tag = namedtuple("Tag", "rel_fname fname line name kind".split())

SQLITE_ERRORS = (sqlite3.OperationalError, sqlite3.DatabaseError, OSError)

CACHE_VERSION = 3
if USING_TSL_PACK:
    CACHE_VERSION = 4

UPDATING_REPO_MAP_MESSAGE = "Updating repo map"


class Spinner:
    """No-op spinner replacement for aider.waiting.Spinner."""

    def __init__(self, text=""):
        pass

    def step(self, text=""):
        pass

    def end(self):
        pass


class RepoMap:
    TAGS_CACHE_DIR = f".aider.tags.cache.v{CACHE_VERSION}"

    warned_files = set()

    def __init__(
        self,
        map_tokens=1024,
        root=None,
        main_model=None,
        io=None,
        repo_content_prefix=None,
        verbose=False,
        max_context_window=None,
        map_mul_no_files=8,
        refresh="auto",
    ):
        self.io = io
        self.verbose = verbose
        self.refresh = refresh

        if not root:
            root = os.getcwd()
        self.root = root

        self.load_tags_cache()
        self.cache_threshold = 0.95

        self.max_map_tokens = map_tokens
        self.map_mul_no_files = map_mul_no_files
        self.max_context_window = max_context_window

        self.repo_content_prefix = repo_content_prefix

        self.main_model = main_model

        self.tree_cache = {}
        self.tree_context_cache = {}
        self.map_cache = {}
        self.map_processing_time = 0
        self.last_map = None

    def token_count(self, text):
        if self.main_model:
            len_text = len(text)
            if len_text < 200:
                return self.main_model.token_count(text)

            lines = text.splitlines(keepends=True)
            num_lines = len(lines)
            step = num_lines // 100 or 1
            lines = lines[::step]
            sample_text = "".join(lines)
            sample_tokens = self.main_model.token_count(sample_text)
            est_tokens = sample_tokens / len(sample_text) * len_text
            return est_tokens

        # Fallback: ~4 chars per token heuristic
        return len(text) // 4

    def _read_text(self, fname):
        """Read file text, replacing aider's self.io.read_text()."""
        try:
            return Path(fname).read_text(errors="replace")
        except (OSError, UnicodeDecodeError):
            return ""

    def get_rel_fname(self, fname):
        try:
            return os.path.relpath(fname, self.root)
        except ValueError:
            return fname

    def tags_cache_error(self, original_error=None):
        if original_error:
            logger.debug("Tags cache error: %s", original_error)

        if isinstance(getattr(self, "TAGS_CACHE", None), dict):
            return

        path = Path(self.root) / self.TAGS_CACHE_DIR

        try:
            if path.exists():
                shutil.rmtree(path)

            new_cache = Cache(path)

            test_key = "test"
            new_cache[test_key] = "test"
            _ = new_cache[test_key]
            del new_cache[test_key]

            self.TAGS_CACHE = new_cache
            return

        except SQLITE_ERRORS as e:
            logger.warning("Unable to use tags cache at %s, falling back to memory cache: %s", path, e)

        self.TAGS_CACHE = dict()

    def load_tags_cache(self):
        path = Path(self.root) / self.TAGS_CACHE_DIR
        try:
            self.TAGS_CACHE = Cache(path)
        except SQLITE_ERRORS as e:
            self.tags_cache_error(e)

    def save_tags_cache(self):
        pass

    def get_mtime(self, fname):
        try:
            return os.path.getmtime(fname)
        except FileNotFoundError:
            logger.warning("File not found: %s", fname)

    def get_tags(self, fname, rel_fname):
        file_mtime = self.get_mtime(fname)
        if file_mtime is None:
            return []

        cache_key = fname
        try:
            val = self.TAGS_CACHE.get(cache_key)
        except SQLITE_ERRORS as e:
            self.tags_cache_error(e)
            val = self.TAGS_CACHE.get(cache_key)

        if val is not None and val.get("mtime") == file_mtime:
            try:
                return self.TAGS_CACHE[cache_key]["data"]
            except SQLITE_ERRORS as e:
                self.tags_cache_error(e)
                return self.TAGS_CACHE[cache_key]["data"]

        # miss!
        data = list(self.get_tags_raw(fname, rel_fname))

        try:
            self.TAGS_CACHE[cache_key] = {"mtime": file_mtime, "data": data}
            self.save_tags_cache()
        except SQLITE_ERRORS as e:
            self.tags_cache_error(e)
            self.TAGS_CACHE[cache_key] = {"mtime": file_mtime, "data": data}

        return data

    def _run_captures(self, query: Query, node):
        if hasattr(query, "captures"):
            return query.captures(node)

        from tree_sitter import QueryCursor

        cursor = QueryCursor(query)
        return cursor.captures(node)

    def get_tags_raw(self, fname, rel_fname):
        lang = filename_to_lang(fname)
        if not lang:
            return

        try:
            language = get_language(lang)
            parser = get_parser(lang)
        except Exception as err:
            logger.debug("Skipping file %s: %s", fname, err)
            return

        query_scm = get_scm_fname(lang)
        if not query_scm.exists():
            return
        query_scm = query_scm.read_text()

        code = self._read_text(fname)
        if not code:
            return
        tree = parser.parse(bytes(code, "utf-8"))

        query = Query(language, query_scm)

        # Use matches to get paired (name, kind) captures from each pattern.
        # Standard tags.scm uses @name for identifiers + @definition.* or @reference.* for parent nodes.
        # Aider's bundled queries use @name.definition.* combined captures.
        # We support both formats.
        saw = set()
        yield from self._extract_tags_from_query(query, tree.root_node, fname, rel_fname, saw)

        if "ref" in saw:
            return
        if "def" not in saw:
            return

        # We saw defs without any refs — backfill refs via pygments
        try:
            lexer = guess_lexer_for_filename(fname, code)
        except Exception:
            return

        tokens = list(lexer.get_tokens(code))
        tokens = [token[1] for token in tokens if token[0] in Token.Name]

        for token in tokens:
            yield Tag(
                rel_fname=rel_fname,
                fname=fname,
                name=token,
                kind="ref",
                line=-1,
            )

    def _extract_tags_from_query(self, query, root_node, fname, rel_fname, saw):
        """Extract Tag objects from query results, supporting both standard and aider SCM formats."""
        captures = self._run_captures(query, root_node)

        # Check if captures use aider's "name.definition.*" format or standard "definition.*" + "name"
        has_aider_format = any(
            tag.startswith("name.definition.") or tag.startswith("name.reference.") for tag in captures
        )

        if has_aider_format:
            # Aider format: @name.definition.class, @name.reference.call
            for tag, nodes in captures.items():
                if tag.startswith("name.definition."):
                    kind = "def"
                elif tag.startswith("name.reference."):
                    kind = "ref"
                else:
                    continue
                saw.add(kind)
                for node in nodes:
                    yield Tag(
                        rel_fname=rel_fname,
                        fname=fname,
                        name=node.text.decode("utf-8"),
                        kind=kind,
                        line=node.start_point[0],
                    )
        else:
            # Standard format: @definition.class + @name, @reference.call + @name
            # Use matches() to get paired captures per pattern
            from tree_sitter import QueryCursor

            cursor = QueryCursor(query)
            for _pattern_idx, match_dict in cursor.matches(root_node):
                name_nodes = match_dict.get("name", [])
                if not name_nodes:
                    continue

                kind = None
                for capture_name in match_dict:
                    if capture_name.startswith("definition."):
                        kind = "def"
                        break
                    elif capture_name.startswith("reference."):
                        kind = "ref"
                        break

                if kind is None:
                    continue

                saw.add(kind)
                for node in name_nodes:
                    yield Tag(
                        rel_fname=rel_fname,
                        fname=fname,
                        name=node.text.decode("utf-8"),
                        kind=kind,
                        line=node.start_point[0],
                    )

    def get_ranked_tags(self, chat_fnames, other_fnames, mentioned_fnames, mentioned_idents, progress=None):
        import networkx as nx

        defines = defaultdict(set)
        references = defaultdict(list)
        definitions = defaultdict(set)

        personalization = dict()

        fnames = set(chat_fnames).union(set(other_fnames))
        chat_rel_fnames = set()

        fnames = sorted(fnames)

        personalize = 100 / len(fnames)

        try:
            cache_size = len(self.TAGS_CACHE)
        except SQLITE_ERRORS as e:
            self.tags_cache_error(e)
            cache_size = len(self.TAGS_CACHE)

        if len(fnames) - cache_size > 100:
            logger.info("Initial repo scan can be slow in larger repos, but only happens once.")
            fnames = tqdm(fnames, desc="Scanning repo")
            showing_bar = True
        else:
            showing_bar = False

        for fname in fnames:
            if progress and not showing_bar:
                progress(f"{UPDATING_REPO_MAP_MESSAGE}: {fname}")

            try:
                file_ok = Path(fname).is_file()
            except OSError:
                file_ok = False

            if not file_ok:
                if fname not in self.warned_files:
                    logger.warning("Repo-map can't include %s", fname)
                    self.warned_files.add(fname)
                continue

            rel_fname = self.get_rel_fname(fname)
            current_pers = 0.0

            if fname in chat_fnames:
                current_pers += personalize
                chat_rel_fnames.add(rel_fname)

            if rel_fname in mentioned_fnames:
                current_pers = max(current_pers, personalize)

            path_obj = Path(rel_fname)
            path_components = set(path_obj.parts)
            basename_with_ext = path_obj.name
            basename_without_ext, _ = os.path.splitext(basename_with_ext)
            components_to_check = path_components.union({basename_with_ext, basename_without_ext})

            matched_idents = components_to_check.intersection(mentioned_idents)
            if matched_idents:
                current_pers += personalize

            if current_pers > 0:
                personalization[rel_fname] = current_pers

            tags = list(self.get_tags(fname, rel_fname))
            if tags is None:
                continue

            for tag in tags:
                if tag.kind == "def":
                    defines[tag.name].add(rel_fname)
                    key = (rel_fname, tag.name)
                    definitions[key].add(tag)

                elif tag.kind == "ref":
                    references[tag.name].append(rel_fname)

        if not references:
            references = dict((k, list(v)) for k, v in defines.items())

        idents = set(defines.keys()).intersection(set(references.keys()))

        G = nx.MultiDiGraph()  # noqa: N806

        for ident in defines.keys():
            if ident in references:
                continue
            for definer in defines[ident]:
                G.add_edge(definer, definer, weight=0.1, ident=ident)

        for ident in idents:
            if progress:
                progress(f"{UPDATING_REPO_MAP_MESSAGE}: {ident}")

            definers = defines[ident]

            mul = 1.0

            is_snake = ("_" in ident) and any(c.isalpha() for c in ident)
            is_kebab = ("-" in ident) and any(c.isalpha() for c in ident)
            is_camel = any(c.isupper() for c in ident) and any(c.islower() for c in ident)
            if ident in mentioned_idents:
                mul *= 10
            if (is_snake or is_kebab or is_camel) and len(ident) >= 8:
                mul *= 10
            if ident.startswith("_"):
                mul *= 0.1
            if len(defines[ident]) > 5:
                mul *= 0.1

            for referencer, num_refs in Counter(references[ident]).items():
                for definer in definers:
                    use_mul = mul
                    if referencer in chat_rel_fnames:
                        use_mul *= 50

                    num_refs = math.sqrt(num_refs)

                    G.add_edge(referencer, definer, weight=use_mul * num_refs, ident=ident)

        if personalization:
            pers_args = dict(personalization=personalization, dangling=personalization)
        else:
            pers_args = dict()

        try:
            ranked = nx.pagerank(G, weight="weight", **pers_args)
        except ZeroDivisionError:
            try:
                ranked = nx.pagerank(G, weight="weight")
            except ZeroDivisionError:
                return []

        ranked_definitions = defaultdict(float)
        for src in G.nodes:
            if progress:
                progress(f"{UPDATING_REPO_MAP_MESSAGE}: {src}")

            src_rank = ranked[src]
            total_weight = sum(data["weight"] for _src, _dst, data in G.out_edges(src, data=True))
            for _src, dst, data in G.out_edges(src, data=True):
                data["rank"] = src_rank * data["weight"] / total_weight
                ident = data["ident"]
                ranked_definitions[(dst, ident)] += data["rank"]

        ranked_tags = []
        ranked_definitions = sorted(ranked_definitions.items(), reverse=True, key=lambda x: (x[1], x[0]))

        for (fname, ident), rank in ranked_definitions:
            if fname in chat_rel_fnames:
                continue
            ranked_tags += list(definitions.get((fname, ident), []))

        rel_other_fnames_without_tags = set(self.get_rel_fname(fname) for fname in other_fnames)

        fnames_already_included = set(rt[0] for rt in ranked_tags)

        top_rank = sorted([(rank, node) for (node, rank) in ranked.items()], reverse=True)
        for rank, fname in top_rank:
            if fname in rel_other_fnames_without_tags:
                rel_other_fnames_without_tags.remove(fname)
            if fname not in fnames_already_included:
                ranked_tags.append((fname,))

        for fname in rel_other_fnames_without_tags:
            ranked_tags.append((fname,))

        return ranked_tags

    def get_ranked_tags_map(
        self,
        chat_fnames,
        other_fnames=None,
        max_map_tokens=None,
        mentioned_fnames=None,
        mentioned_idents=None,
        force_refresh=False,
    ):
        cache_key = [
            tuple(sorted(chat_fnames)) if chat_fnames else None,
            tuple(sorted(other_fnames)) if other_fnames else None,
            max_map_tokens,
        ]

        if self.refresh == "auto":
            cache_key += [
                tuple(sorted(mentioned_fnames)) if mentioned_fnames else None,
                tuple(sorted(mentioned_idents)) if mentioned_idents else None,
            ]
        cache_key = tuple(cache_key)

        use_cache = False
        if not force_refresh:
            if self.refresh == "manual" and self.last_map:
                return self.last_map

            if self.refresh == "always":
                use_cache = False
            elif self.refresh == "files":
                use_cache = True
            elif self.refresh == "auto":
                use_cache = self.map_processing_time > 1.0

            if use_cache and cache_key in self.map_cache:
                return self.map_cache[cache_key]

        start_time = time.time()
        result = self.get_ranked_tags_map_uncached(
            chat_fnames, other_fnames, max_map_tokens, mentioned_fnames, mentioned_idents
        )
        end_time = time.time()
        self.map_processing_time = end_time - start_time

        self.map_cache[cache_key] = result
        self.last_map = result

        return result

    def get_ranked_tags_map_uncached(
        self,
        chat_fnames,
        other_fnames=None,
        max_map_tokens=None,
        mentioned_fnames=None,
        mentioned_idents=None,
    ):
        if not other_fnames:
            other_fnames = list()
        if not max_map_tokens:
            max_map_tokens = self.max_map_tokens
        if not mentioned_fnames:
            mentioned_fnames = set()
        if not mentioned_idents:
            mentioned_idents = set()

        spin = Spinner(UPDATING_REPO_MAP_MESSAGE)

        ranked_tags = self.get_ranked_tags(
            chat_fnames,
            other_fnames,
            mentioned_fnames,
            mentioned_idents,
            progress=spin.step,
        )

        other_rel_fnames = sorted(set(self.get_rel_fname(fname) for fname in other_fnames))
        special_fnames = filter_important_files(other_rel_fnames)
        ranked_tags_fnames = set(tag[0] for tag in ranked_tags)
        special_fnames = [fn for fn in special_fnames if fn not in ranked_tags_fnames]
        special_fnames = [(fn,) for fn in special_fnames]

        ranked_tags = special_fnames + ranked_tags

        spin.step()

        num_tags = len(ranked_tags)
        lower_bound = 0
        upper_bound = num_tags
        best_tree = None
        best_tree_tokens = 0

        chat_rel_fnames = set(self.get_rel_fname(fname) for fname in chat_fnames)

        self.tree_cache = dict()

        middle = min(int(max_map_tokens // 25), num_tags)
        while lower_bound <= upper_bound:
            tree = self.to_tree(ranked_tags[:middle], chat_rel_fnames)
            num_tokens = self.token_count(tree)

            pct_err = abs(num_tokens - max_map_tokens) / max_map_tokens
            ok_err = 0.15
            if (num_tokens <= max_map_tokens and num_tokens > best_tree_tokens) or pct_err < ok_err:
                best_tree = tree
                best_tree_tokens = num_tokens

                if pct_err < ok_err:
                    break

            if num_tokens < max_map_tokens:
                lower_bound = middle + 1
            else:
                upper_bound = middle - 1

            middle = int((lower_bound + upper_bound) // 2)

        spin.end()
        return best_tree

    tree_cache = dict()

    def render_tree(self, abs_fname, rel_fname, lois):
        mtime = self.get_mtime(abs_fname)
        key = (rel_fname, tuple(sorted(lois)), mtime)

        if key in self.tree_cache:
            return self.tree_cache[key]

        if rel_fname not in self.tree_context_cache or self.tree_context_cache[rel_fname]["mtime"] != mtime:
            code = self._read_text(abs_fname) or ""
            if not code.endswith("\n"):
                code += "\n"

            context = TreeContext(
                rel_fname,
                code,
                color=False,
                line_number=False,
                child_context=False,
                last_line=False,
                margin=0,
                mark_lois=False,
                loi_pad=0,
                show_top_of_file_parent_scope=False,
            )
            self.tree_context_cache[rel_fname] = {"context": context, "mtime": mtime}

        context = self.tree_context_cache[rel_fname]["context"]
        context.lines_of_interest = set()
        context.add_lines_of_interest(lois)
        context.add_context()
        res = context.format()
        self.tree_cache[key] = res
        return res

    def to_tree(self, tags, chat_rel_fnames):
        if not tags:
            return ""

        cur_fname = None
        cur_abs_fname = None
        lois = None
        output = ""

        dummy_tag = (None,)
        for tag in sorted(tags) + [dummy_tag]:
            this_rel_fname = tag[0]
            if this_rel_fname in chat_rel_fnames:
                continue

            if this_rel_fname != cur_fname:
                if lois is not None:
                    output += "\n"
                    output += cur_fname + ":\n"
                    output += self.render_tree(cur_abs_fname, cur_fname, lois)
                    lois = None
                elif cur_fname:
                    output += "\n" + cur_fname + "\n"
                if type(tag) is Tag:
                    lois = []
                    cur_abs_fname = tag.fname
                cur_fname = this_rel_fname

            if lois is not None:
                lois.append(tag.line)

        # truncate long lines, in case we get minified js or something else crazy
        output = "\n".join([line[:100] for line in output.splitlines()]) + "\n"

        return output


def get_scm_fname(lang):
    # Try individual tree_sitter_<lang> packages first (how tree-sitter-language-pack ships queries)
    pkg_name = f"tree_sitter_{lang.replace('-', '_')}"
    try:
        path = resources.files(pkg_name).joinpath("queries", "tags.scm")
        if path.exists():
            return path
    except (KeyError, ModuleNotFoundError):
        pass

    # Fallback: grep_ast bundled queries (older aider layout)
    if USING_TSL_PACK:
        subdir = "tree-sitter-language-pack"
        try:
            path = resources.files("grep_ast").joinpath(
                "queries",
                subdir,
                f"{lang}-tags.scm",
            )
            if path.exists():
                return path
        except KeyError:
            pass

    subdir = "tree-sitter-languages"
    try:
        return resources.files("grep_ast").joinpath(
            "queries",
            subdir,
            f"{lang}-tags.scm",
        )
    except KeyError:
        return
