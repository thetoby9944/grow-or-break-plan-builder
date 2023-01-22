import streamlit as st


def pad_dict_list(dict_list_orig: dict, padel):
    dict_list = {key: [i for i in value]for key, value in dict_list_orig.items()}
    lmax = 0
    for lname in dict_list.keys():
        lmax = max(lmax, len(dict_list[lname]))
    for lname in dict_list.keys():
        ll = len(dict_list[lname])
        if  ll < lmax:
            dict_list[lname] += [padel] * (lmax - ll)
    return dict_list


def make_grid(rows, cols):
    return {
        row :
        st.columns(cols)
        for row in range(rows)
    }


def flatten(item, sequences=(tuple, list, set)):
    yield from map(flatten, item) if isinstance(item, sequences) else [item]
