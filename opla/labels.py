"""Label definitions for POS tagging and dependency parsing.

Copied from gr-nlp-toolkit configs to ensure exact compatibility.
"""

# UPOS tags + 16 morphological feature label maps.
# Each key maps to an ordered list of labels (index -> label string).
pos_labels = {
    'Abbr': ['_', 'Yes'],
    'Aspect': ['Perf', '_', 'Imp'],
    'Case': ['Dat', '_', 'Acc', 'Gen', 'Nom', 'Voc'],
    'Definite': ['Ind', 'Def', '_'],
    'Degree': ['Cmp', 'Sup', '_'],
    'Foreign': ['_', 'Yes'],
    'Gender': ['Fem', 'Masc', '_', 'Neut'],
    'Mood': ['Ind', '_', 'Imp'],
    'NumType': ['Mult', 'Card', '_', 'Ord', 'Sets'],
    'Number': ['Plur', '_', 'Sing'],
    'Person': ['3', '1', '_', '2'],
    'Poss': ['_', 'Yes'],
    'PronType': ['Ind', 'Art', '_', 'Rel', 'Dem', 'Prs', 'Ind,Rel', 'Int'],
    'Tense': ['Pres', 'Past', '_'],
    'VerbForm': ['Part', 'Conv', '_', 'Inf', 'Fin'],
    'Voice': ['Pass', 'Act', '_'],
    'upos': [
        'X', 'PROPN', 'PRON', 'ADJ', 'AUX', 'PART', 'ADV', '_',
        'DET', 'SYM', 'NUM', 'CCONJ', 'PUNCT', 'NOUN', 'SCONJ',
        'ADP', 'VERB',
    ],
}

# Valid morphological features per UPOS tag.
pos_properties = {
    'ADJ': ['Degree', 'Number', 'Gender', 'Case'],
    'ADP': ['Number', 'Gender', 'Case'],
    'ADV': ['Degree', 'Abbr'],
    'AUX': ['Mood', 'Aspect', 'Tense', 'Number', 'Person', 'VerbForm', 'Voice'],
    'CCONJ': [],
    'DET': ['Number', 'Gender', 'PronType', 'Definite', 'Case'],
    'NOUN': ['Number', 'Gender', 'Abbr', 'Case'],
    'NUM': ['NumType', 'Number', 'Gender', 'Case'],
    'PART': [],
    'PRON': ['Number', 'Gender', 'Person', 'Poss', 'PronType', 'Case'],
    'PROPN': ['Number', 'Gender', 'Case'],
    'PUNCT': [],
    'SCONJ': [],
    'SYM': [],
    'VERB': ['Mood', 'Aspect', 'Tense', 'Number', 'Gender', 'Person',
             'VerbForm', 'Voice', 'Case'],
    'X': ['Foreign'],
    '_': [],
}

# Dependency relation labels (index -> label string).
dp_labels = [
    'obl', 'obj', 'dep', 'mark', 'case', 'flat', 'nummod', 'obl:arg',
    'punct', 'cop', 'acl:relcl', 'expl', 'nsubj', 'csubj:pass', 'root',
    'advmod', 'nsubj:pass', 'ccomp', 'conj', 'amod', 'xcomp', 'aux',
    'appos', 'csubj', 'fixed', 'nmod', 'iobj', 'parataxis', 'orphan',
    'det', 'advcl', 'vocative', 'compound', 'cc', 'discourse', 'acl',
    'obl:agent',
]
