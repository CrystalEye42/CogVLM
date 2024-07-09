"""
import json

def create_gt(data):
    def convert_bb(bbox):
        h, w = data['height'], data['width']
        temp = [bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h]
        temp = [int(t * 1000) for t in temp]
        return [temp[0], temp[1], temp[2] + temp[0], temp[3] + temp[1]]
    bboxes = []
    for bbox in data['bboxes']:
        bboxes.append(convert_bb(bbox['bbox']))
    result = []
    for rxn in data['reactions']:
        curr = {}
        for k, v in rxn.items():
            curr[k] = [[bboxes[i]] for i in v]
        result.append(curr)
    data['label'] = result
with open('/data/rsg/chemistry/wang7776/images/train/train.json', 'r') as f:
	all_files = json.load(f)['images']

max_len = 0
max_label = None

for sample in all_files:
    create_gt(sample)
    if len(str(sample['label'])) > max_len:
        max_len = len(str(sample['label']))
        max_label = sample['label']

print(max_len)
print(str(max_label))
"""


def rxnscribe_eval(pred, label):
    try:
        pred = eval(pred)
        tar = eval(label)
    except:
        return 0
    def get_iou(bb1, bb2):
        bb1 = bb1[0]
        bb2 = bb2[0]
        bb_intersect = [max(bb1[0], bb2[0]), max(bb1[1], bb2[1]), min(bb1[2], bb2[2]), min(bb1[3], bb2[3])]
        def get_area(bbox):
            return max(0, bbox[2] - bbox[0] + 1) * max(0, bbox[3] - bbox[1] + 1)
        inter_area = get_area(bb_intersect)
        return inter_area / (get_area(bb1) + get_area(bb2) - inter_area)

    def match_bbox(bbox, rxn, keys):
        max_iou = 0
        match_bb = None
        for k in keys:
            for bb in rxn[k]:
                iou = get_iou(bb, bbox)
                if iou > max_iou:
                    max_iou = iou
                    match_bb = bb
        return max_iou, match_bb
    
    soft_matched = {'prec': 0, 'rec': 0}
    hard_matched = {'prec': 0, 'rec': 0}
    soft_keys = {'reactants': ['reactants', 'conditions'], 'conditions': ['reactants', 'conditions'], 'products': ['products']}
    for rxn in pred:
        # soft match
        for k, v in rxn.items():
            soft_flag = True
            hard_flag = True
            for bbox in v:
                iou, bb = list(zip(*[match_bbox(bbox, rxn1, soft_keys[k]) for rxn1 in tar]))
                if max(iou) < 0.5:
                    soft_flag = False
                iou, bb = list(zip(*[match_bbox(bbox, rxn1, [k]) for rxn1 in tar]))
                if max(iou) < 0.5:
                    hard_flag = False
        if soft_flag:
            soft_matched['prec'] += 1
        if hard_flag:
            hard_matched['prec'] += 1
    for rxn in tar:
        # soft match
        for k, v in rxn.items():
            soft_flag = True
            hard_flag = True
            for bbox in v:
                iou, bb = list(zip(*[match_bbox(bbox, rxn1, soft_keys[k]) for rxn1 in pred]))
                if max(iou) < 0.5:
                    soft_flag = False
                iou, bb = list(zip(*[match_bbox(bbox, rxn1, [k]) for rxn1 in pred]))
                if max(iou) < 0.5:
                    hard_flag = False
        if soft_flag:
            soft_matched['rec'] += 1
        if hard_flag:
            hard_matched['rec'] += 1
    return soft_matched, hard_matched

print(rxnscribe_eval("[{'reactants': [[[179, 194, 346, 354]]], 'conditions': [[[379, 201, 487, 260]], [[379, 260, 619, 354]]], 'products': [[[651, 194, 821, 354]]]}, {'reactants': [[[179, 443, 346, 604]]], 'conditions': [[[379, 448, 619, 515]], [[379, 515, 619, 575]]], 'products': [[[651, 443, 821, 582]]]}]", 
                     "[{'reactants': [[174, 440, 347, 607]]], 'conditions': [[[380, 455, 613, 513]], [[378, 519, 544, 580]]], 'products': [[[648, 447, 821, 593]]]}, {'reactants': [[[179, 194, 346, 356]]], 'conditions': [[[380, 202, 551, 269]], [[377, 272, 559, 358]]], 'products': [[[651, 192, 821, 338]]]}]"))
print(rxnscribe_eval("[{'reactants': [[[25, 5, 249, 132]]], 'conditions': [[[268, 1, 351, 23]], [[268, 51, 351, 69]]], 'products': [[[404, 11, 595, 132]]]}, {'reactants': [[[404, 11, 595, 132]]], 'conditions': [[[638, 43, 768, 65]], [[648, 66, 765, 81]]], 'products': [[[808, 19, 939, 81]]]}]", "[{'reactants': [[[21, 2, 253, 134]]], 'conditions': [[[268, 50, 354, 72]]], 'products': [[[397, 5, 632, 138]]]}, {'reactants': [[[268, 3, 354, 20]]], 'conditions': [], 'products': [[[268, 50, 354, 72]]]}, {'reactants': [[[397, 5, 632, 138]]], 'conditions': [[[639, 35, 770, 65]]], 'products': [[[764, 18, 954, 86]]]}]"))