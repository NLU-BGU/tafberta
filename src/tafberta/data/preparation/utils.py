from typing import List
import pylangacq
from pylangacq import Reader

def get_children(file_paths: List[str]) -> List[str]:
    """
    Suit for BatEl, BermanLong and Levy datasets
    """
    return list(set([path.split('/')[1] for path in file_paths]))

def get_not_child_participants(dataset: pylangacq.chat.Reader) -> List[str]:
    return list(set([parti for header in dataset.headers()
                    for parti, meta in header['Participants'].items()
                    if meta['role'] != 'Target_Child']))

def get_child_participants(dataset: pylangacq.chat.Reader) -> List[str]:
    return list(set([parti for header in dataset.headers()
                    for parti, meta in header['Participants'].items()
                    if meta['role'] == 'Target_Child']))

def is_dataset_has_morphology(dataset: pylangacq.chat.Reader):
    utt = dataset.utterances(participants=get_not_child_participants(dataset))
    return '%mor' in(utt[0].tiers.keys())

def flatten_list(l):
    return [item for sublist in l for item in sublist]

def del_list_by_ids(l, id_to_del):
    if id_to_del == []:
        return l
    return [i for j, i in enumerate(l) if j not in id_to_del]

def get_eng_cleaned_sentences(reader_eng: Reader,
                              extra_lines_ids: List
                             ) -> List[str]:
    eng_utt = reader_eng.utterances()
        
    # in all the files, the extra lines are in eng files
    eng_utt_cleaned = del_list_by_ids(eng_utt, extra_lines_ids)
    eng_sentences = [' '.join([token.word for token in utt.tokens])
                     for utt in eng_utt_cleaned]
    return eng_sentences, eng_utt_cleaned