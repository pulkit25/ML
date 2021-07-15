# Returns a labels to IDs mapping sorted by key
def labels_to_ids(labels):
    return {label: i for i, label in enumerate(sorted(labels))}