from rouge_score import rouge_scorer

"""Rouge-L 来计算response和reference之间的最长公共子序列 (LCS)"""
def get_rougel(generate, reference):
    """
    generate指的是模型生成的答案，类型为字符串
    reference指的是标注的标准答案，类型为字符串
    """
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generate)
    rougel = scores['rougeL'].fmeasure
    return rougel

"""通过准确率P、召回率R计算诊断结果的Micro-F1"""
def eval(reference, predict_result):
    """
    reference指的是标注的标准答案，类型为list
    predict_result指的是模型生成的答案，类型为list
    """
    predict_entity_num = len(list(set(predict_result)))
    gold_entity_num = len(list(set(reference)))
    correct_entity_num =len(list(set(reference)&set(predict_result)))

    entity_precision = correct_entity_num/predict_entity_num
    entity_recall = correct_entity_num/gold_entity_num
    triplet_f1 = 2 * entity_precision * entity_recall/(entity_precision + entity_recall)
    return entity_precision, entity_recall, triplet_f1

"""要点信息计算Macro-Recall（针对的是初步诊断依据、最终诊断依据、鉴别诊断）"""
# 该指标是一种宏观平均指标，它首先计算每个类别的召回率，然后对所有类别的召回率取平均值。

def calculate_macro_recall(score_points, student_answer):
    """
    score_points指的是标注的打分点
    student_answer指的是模型生成的答案
    """
    recalls = []
    for category, points in score_points.items():
        tp = sum(1 for point in points if point in student_answer)
        fn = len(points) - tp
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        recalls.append(recall)
    
    macro_recall = sum(recalls) / len(recalls) if recalls else 0
    return macro_recall
