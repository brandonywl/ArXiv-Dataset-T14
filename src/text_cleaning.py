import spacy
from spacy.attrs import ORTH, NORM
from src.scispacy.abbreviation import AbbreviationDetector
from tqdm import tqdm
import re

def prepare_spacy_nlp():
    # Referencing https://stackoverflow.com/questions/76255486/how-to-stop-spacy-tokenizer-from-tokenizing-words-enclosed-within-brackets
    # To add in custom tokens for eqn latex and eqn complexity.

    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("abbreviation_detector", config={"make_serializable": True})

    infixes = nlp.Defaults.infixes + [r"([\[\]])"]
    nlp.tokenizer.infix_finditer = spacy.util.compile_infix_regex(infixes).finditer

    # Define the words that will be found within the square brackets
    tags = [
        "EQN_LATEX",
        "EQN_COMPLEXITY",
    ]

    # Add the special cases to the tokenizer
    for tag in tags:
        nlp.tokenizer.add_special_case(f"[{tag}]", [{ORTH: f"[{tag}]"}])

    return nlp

def abbreviation_expansion_disambiguation(doc, trace_count=0, remove_prepend_ws=True):

    if len(doc._.abbreviations) == 0:
        text = doc.text
        if trace_count > 1:
            count = {}
        else:
            count = 0
        return text if trace_count == 0 else (text, count)

    serializable = type(doc._.abbreviations[0]) == dict

    # Set of abbreviations in doc
    if serializable:
        unique_abbv_set = set([(abbv['short_text'], abbv['long_text']) for abbv in doc._.abbreviations if (abbv['short_end'] - abbv['short_start'] == 1)])
    else:
        unique_abbv_set = set([(abbv.text, abbv._.long_form.text) for abbv in doc._.abbreviations])

    # Candidate set of abbreviations to disambiguate plural and singular abbreviations
    # I.e. Hidden Markov Models (HMMs) and HMM both -> Hidden Markov Models
    # Rules: Check if both longform and shortform end with s, and there exists no other abbreviations that have the shortform less an s.
    # (What happens if it is GPS? Should not be an issue following the rules)
    candidate_set = {
        abbv[:-1]: (lf, abbv)
        for abbv, lf in unique_abbv_set
            if abbv[:-1] not in unique_abbv_set and abbv[-1].lower() == lf[-1].lower() and abbv[-1].lower() == 's'
    }

    plural_candidate_set = {
        abbv + 's': (lf, abbv)
        for abbv, lf in unique_abbv_set
            if abbv + 's' not in unique_abbv_set and abbv[-1] != "s"
    }

    # Not abbreviated by the pipeline yet
    new_abbvs_set = candidate_set | plural_candidate_set

    abbvs_set = {sf: (lf, sf)for sf, lf in unique_abbv_set}
    abbvs_set = abbvs_set | new_abbvs_set

    # original_abbreviation_tokens = [d if not serializable else doc[d['short_start']] for d in doc._.abbreviations if (d['short_end'] - d['short_start']) == 1]

    # Does not capture multiple token acronyms
    abbrev_token_set = [token for token in doc if token.text in new_abbvs_set]
    abbrev_token_set.extend([doc[d['short_start']] for d in doc._.abbreviations if d['short_end'] - d['short_start'] == 1])

    # Set of abbreviations to drop due to abbreviation definition
    abbvs_to_drop = [abbv for abbv in abbrev_token_set if doc[abbv.i-1].text == '(' and doc[abbv.i+1].text == ')']
    # Index tuple of those abbreviations
    prepend_ws_index = int(remove_prepend_ws) + 1
    idx_abbv_to_drop = sorted([(abbv.idx-prepend_ws_index, abbv.idx+1+len(abbv.text)) for abbv in abbvs_to_drop], key=lambda x: (-x[0], x[1]))
    # Remaining tokens to be expanded
    abbvs = [abbv for abbv in abbrev_token_set if abbv not in abbvs_to_drop]
    # Sort by descending idx for starting character index. Won't mess up ordering in text
    sorted_abbvs = [(abbv.idx, abbv.idx + len(abbv.text), abbvs_set[abbv.text]) for abbv in abbvs]
    sorted_abbvs = sorted(sorted_abbvs, key=lambda x: (-x[0], x[1]))

    text = doc.text

    for abbv in sorted_abbvs:
        text = text[:abbv[0]] + abbv[2][0] + text[abbv[1]:]

    for idx_set in idx_abbv_to_drop:
        text = text[:idx_set[0]] + text[idx_set[1]:]

    if trace_count > 1:
        count = {}
        abbvs = [abbvs_set[token.text] for token in abbrev_token_set]
        for abbv in abbvs:
            if abbv in count:
                count[abbv] += 1
            else:
                count[abbv] = 1
    else:
        count = int(len(abbvs) > 0)

    return text if trace_count == 0 else (text, count)

def preprocess(text_series, trace_count=0, show_prev_per_step=False):

    counts = {}
    prev_per_step = []

    newline_to_space = "new-line-to-space"
    newline_to_space_regex = r'\n'
    newline_to_space_sub = r' '

    whitespace_strip = "strip_whitespace"
    whitespace_strip_regex = r'^\s+(\S.*\S)\s+$'
    whitespace_strip_sub = r'\1'

    latex_notation = "latex-to-placeholder"
    latex_notation_regex = r'\$.+?\$'
    latex_notation_sub = r'[EQN_LATEX]'

    complexity_notation = "complexity-to-placeholder"
    complexity_notation_regex = r'\b[A-Z]+\(.+?\)'
    complexity_notation_sub = r'[EQN_COMPLEXITY]'
    
    latex_apostrophe = "latex-apostrophy-removal"
    latex_apostrophe_regex = r'`{,2}\b(.+?)\b\'{,2}'
    latex_apostrophe_sub = r'\1'

    # Citations have seen \\cite{Hee05} \\\\cite{ Heeee123 } \\cite {123asb}. Note the space before is needed ' \\...' to replace it properly without spare whitespace.
    latex_citation = "latex-citation-removal"
    latex_citation_regex = r' \\{,2}cite\s*?{\s*\b.+?\b\s*}'
    latex_citation_sub = r''

    # Older text formatting format works like {\\ text here} or {\\it text here} or {\\\\bf text here} or even {\o}
    latex_formatting = "latex-formatting-removal"
    latex_formatting_regex = r'{\\+(\S*?\s+?)?(\b.+?\b)}'
    latex_formatting_sub = r'\2'

    # Some text formatting takes the form of \\textbf{} or \\emph{} or even \\sqrt(n). For the last case no choice but to ignore the error in parsing.
    latex_formatting_alt = "latex-formatting-alt-removal"
    latex_formatting_alt_regex = r'\\\S*?{\s*?(\S.*?)\s*?}'
    latex_formatting_alt_sub = r'\1'

    regexes = [
        (newline_to_space, newline_to_space_regex, newline_to_space_sub),
        (whitespace_strip, whitespace_strip_regex, whitespace_strip_sub),
        (latex_notation, latex_notation_regex, latex_notation_sub),
        (complexity_notation, complexity_notation_regex, complexity_notation_sub),
        (latex_apostrophe, latex_apostrophe_regex, latex_apostrophe_sub),
        (latex_citation, latex_citation_regex, latex_citation_sub),
        (latex_formatting, latex_formatting_regex, latex_formatting_sub),
        (latex_formatting_alt, latex_formatting_alt_regex, latex_formatting_alt_sub)
    ]

    for regex_name, regex, regex_sub in regexes:
        temp = text_series.str.replace(regex, regex_sub, regex=True)
        if trace_count != 0:
            count = len(text_series) - (temp == text_series).sum()
            counts[regex_name] = count

        if show_prev_per_step:
            prev_per_step.append(temp)

        text_series = temp

    if show_prev_per_step:
        return (text_series, prev_per_step) if trace_count == 0 else (text_series, counts, prev_per_step)

    return text_series if trace_count == 0 else (text_series, counts)
