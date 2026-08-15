"""
Decision tree — debugging edition.
Open this file in PyCharm, set a breakpoint on the line marked
"# <-- BREAKPOINT 1", then press Debug (Shift+F9) and step through.

Suggested breakpoints:
  1. In get_best_split, on:  gini = gini_index(groups, class_values)   (see each candidate split)
  2. In split, on:           left, right = node['groups']              (watch the recursion in Frames)
"""
from collections import Counter


def gini_index(groups, classes):
    n_instances = float(sum([len(group) for group in groups]))
    gini = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0
        labels = [row[-1] for row in group]
        for class_val in classes:
            p = labels.count(class_val) / size   # proportion of this class in the group
            score += p * p
        gini += (1.0 - score) * (size / n_instances)
    return gini


def test_split(index, value, dataset):
    left, right = [], []
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


def get_best_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    best_index, best_value, best_score, best_groups = None, None, float('inf'), None
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)   # <-- BREAKPOINT 1 (Step Into here with F7)
            if gini < best_score:
                best_index, best_value, best_score, best_groups = index, row[index], gini, groups
    return {'index': best_index, 'value': best_value, 'groups': best_groups}


def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return Counter(outcomes).most_common(1)[0][0]


def split(node, max_depth, min_size, depth):
    left, right = node['groups']                      # <-- BREAKPOINT 2 (watch Frames stack up)
    del (node['groups'])
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_best_split(left)
        split(node['left'], max_depth, min_size, depth + 1)
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_best_split(right)
        split(node['right'], max_depth, min_size, depth + 1)


def build_tree(train, max_depth, min_size):
    root = get_best_split(train)
    split(root, max_depth, min_size, 1)
    return root


def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


if __name__ == '__main__':
    # Each row: [years_experience (X1), interview_score (X2), hired (class)]
    dataset = [
        [2.8, 1.8, 0],
        [1.5, 2.3, 0],
        [3.4, 1.0, 0],
        [2.0, 3.6, 1],
        [3.1, 3.9, 1],
        [1.3, 3.3, 1],
        [7.6, 2.8, 1],
        [8.9, 1.2, 1],
        [7.4, 3.6, 1],
        [9.2, 0.9, 1],
    ]

    tree = build_tree(dataset, max_depth=3, min_size=1)
    print("Learned tree:", tree)

    # Step Into this call (F7) to walk the tree during a prediction:
    candidate = [7.5, 1.5, None]   # 7.5 yrs experience, interview score 1.5
    print("Prediction for", candidate[:-1], "->", predict(tree, candidate))