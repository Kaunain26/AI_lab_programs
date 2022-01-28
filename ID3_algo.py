import pandas as pd
import math
from collections import Counter
from pprint import pprint

df = pd.read_csv('PlayTennis.csv')
print("\n Input Data Set is:\n", df)
t = df.keys()[-1]
print("Target Attribute is: ", t)
attribute_names = list(df.keys())
attribute_names.remove(t)
print('Predicting Attributes: ', attribute_names)


def entropy(probs):
    return sum([-prob * math.log(prob, 2) for prob in probs])


def entropy_of_list(ls, value):
    from collections import Counter
    cnt = Counter(x for x in ls)
    print("target attribute class count(Yes/No)=", dict(cnt))
    total_instances = len(ls)
    print("Total no of instances/records associated with {0} is: {1}".format(value, total_instances))
    probs = [x / total_instances for x in cnt.values()]
    print("Probability of Class {0} is: {1:.4f}".format(min(cnt), min(probs)))
    print("Probability of Class {0} is: {1:.4f}".format(max(cnt), max(probs)))
    return entropy(probs)


def information_gain(df, split_attribute, target_attribute, battr):
    print("\n\n----Information Gain Calculation of ", split_attribute, ".........")
    df_split = df.groupby(split_attribute)
    glist = []
    for gname, group in df_split:
        print('Grouped Attribute Values \n', group)
        glist.append(gname)

    glist.reverse()
    nobs = len(df.index) * 1.0
    df_agg1 = df_split.agg({target_attribute: lambda x: entropy_of_list(x, glist.pop())})
    df_agg2 = df_split.agg({target_attribute: lambda x: len(x) / nobs})

    df_agg1.columns = ['Entropy']
    df_agg2.columns = ['Proportion']

    new_entropy = sum(df_agg1['Entropy'] * df_agg2['Proportion'])
    if battr != 'S':
        old_entropy = entropy_of_list(df[target_attribute], 'S-' + df.iloc[0][df.columns.get_loc(battr)])
    else:
        old_entropy = entropy_of_list(df[target_attribute], battr)
    return old_entropy - new_entropy


def id3(df, target_attribute, attribute_names, default_class=None, default_attr='S'):
    from collections import Counter
    cnt = Counter(x for x in df[target_attribute])

    if len(cnt) == 1:
        return next(iter(cnt))

    elif df.empty or (not attribute_names):
        return default_class

    else:
        default_class = max(cnt.keys())
        gainz = []
        for attr in attribute_names:
            ig = information_gain(df, attr, target_attribute, default_attr)
            gainz.append(ig)
            print("Information gain of ", attr, ' is : ', ig)

        index_of_max = gainz.index(max(gainz))
        best_attr = attribute_names[index_of_max]
        print("\nAttribute with the maximum gain is :", best_attr)

        tree = {best_attr: {}}
        remaining_attribute_names = [i for i in attribute_names if i != best_attr]

        for attr_val, data_subset in df.groupby(best_attr):
            subtree = id3(data_subset, target_attribute, remaining_attribute_names, default_class, default_attr)
            tree[best_attr][attr_val] = subtree
        return tree


c = Counter(x for x in df['Play Tennis'])
len(c)

tree = id3(df, t, attribute_names)
print("\nThe Resultant Decision Tree is: ")
pprint(tree)

# Output...

#
#  Input Data Set is:
#       Outlook Temperature Humidity    Wind Play Tennis
# 0      Sunny         Hot     High    Weak          No
# 1      Sunny         Hot     High  Strong          No
# 2   Overcast         Hot     High    Weak         Yes
# 3       Rain        Mild     High    Weak         Yes
# 4       Rain        Cool   Normal    Weak         Yes
# 5       Rain        Cool   Normal  Strong          No
# 6   Overcast        Cool   Normal  Strong         Yes
# 7      Sunny        Mild     High    Weak          No
# 8      Sunny        Cool   Normal    Weak         Yes
# 9       Rain        Mild   Normal    Weak         Yes
# 10     Sunny        Mild   Normal  Strong         Yes
# 11  Overcast        Mild     High  Strong         Yes
# 12  Overcast         Hot   Normal    Weak         Yes
# 13      Rain        Mild     High  Strong          No
# Target Attribute is:  Play Tennis
# Predicting Attributes:  ['Outlook', 'Temperature', 'Humidity', 'Wind']
#
#
# ----Information Gain Calculation of  Outlook .........
# Grouped Attribute Values
#       Outlook Temperature Humidity    Wind Play Tennis
# 2   Overcast         Hot     High    Weak         Yes
# 6   Overcast        Cool   Normal  Strong         Yes
# 11  Overcast        Mild     High  Strong         Yes
# 12  Overcast         Hot   Normal    Weak         Yes
# Grouped Attribute Values
#     Outlook Temperature Humidity    Wind Play Tennis
# 3     Rain        Mild     High    Weak         Yes
# 4     Rain        Cool   Normal    Weak         Yes
# 5     Rain        Cool   Normal  Strong          No
# 9     Rain        Mild   Normal    Weak         Yes
# 13    Rain        Mild     High  Strong          No
# Grouped Attribute Values
#     Outlook Temperature Humidity    Wind Play Tennis
# 0    Sunny         Hot     High    Weak          No
# 1    Sunny         Hot     High  Strong          No
# 7    Sunny        Mild     High    Weak          No
# 8    Sunny        Cool   Normal    Weak         Yes
# 10   Sunny        Mild   Normal  Strong         Yes
# target attribute class count(Yes/No)= {'Yes': 4}
# Total no of instances/records associated with Overcast is: 4
# Probability of Class Yes is: 1.0000
# Probability of Class Yes is: 1.0000
# target attribute class count(Yes/No)= {'Yes': 3, 'No': 2}
# Total no of instances/records associated with Rain is: 5
# Probability of Class No is: 0.4000
# Probability of Class Yes is: 0.6000
# target attribute class count(Yes/No)= {'No': 3, 'Yes': 2}
# Total no of instances/records associated with Sunny is: 5
# Probability of Class No is: 0.4000
# Probability of Class Yes is: 0.6000
# target attribute class count(Yes/No)= {'No': 5, 'Yes': 9}
# Total no of instances/records associated with S is: 14
# Probability of Class No is: 0.3571
# Probability of Class Yes is: 0.6429
# Information gain of  Outlook  is :  0.2467498197744391
#
#
# ----Information Gain Calculation of  Temperature .........
# Grouped Attribute Values
#      Outlook Temperature Humidity    Wind Play Tennis
# 4      Rain        Cool   Normal    Weak         Yes
# 5      Rain        Cool   Normal  Strong          No
# 6  Overcast        Cool   Normal  Strong         Yes
# 8     Sunny        Cool   Normal    Weak         Yes
# Grouped Attribute Values
#       Outlook Temperature Humidity    Wind Play Tennis
# 0      Sunny         Hot     High    Weak          No
# 1      Sunny         Hot     High  Strong          No
# 2   Overcast         Hot     High    Weak         Yes
# 12  Overcast         Hot   Normal    Weak         Yes
# Grouped Attribute Values
#       Outlook Temperature Humidity    Wind Play Tennis
# 3       Rain        Mild     High    Weak         Yes
# 7      Sunny        Mild     High    Weak          No
# 9       Rain        Mild   Normal    Weak         Yes
# 10     Sunny        Mild   Normal  Strong         Yes
# 11  Overcast        Mild     High  Strong         Yes
# 13      Rain        Mild     High  Strong          No
# target attribute class count(Yes/No)= {'Yes': 3, 'No': 1}
# Total no of instances/records associated with Cool is: 4
# Probability of Class No is: 0.2500
# Probability of Class Yes is: 0.7500
# target attribute class count(Yes/No)= {'No': 2, 'Yes': 2}
# Total no of instances/records associated with Hot is: 4
# Probability of Class No is: 0.5000
# Probability of Class Yes is: 0.5000
# target attribute class count(Yes/No)= {'Yes': 4, 'No': 2}
# Total no of instances/records associated with Mild is: 6
# Probability of Class No is: 0.3333
# Probability of Class Yes is: 0.6667
# target attribute class count(Yes/No)= {'No': 5, 'Yes': 9}
# Total no of instances/records associated with S is: 14
# Probability of Class No is: 0.3571
# Probability of Class Yes is: 0.6429
# Information gain of  Temperature  is :  0.029222565658954647
#
#
# ----Information Gain Calculation of  Humidity .........
# Grouped Attribute Values
#       Outlook Temperature Humidity    Wind Play Tennis
# 0      Sunny         Hot     High    Weak          No
# 1      Sunny         Hot     High  Strong          No
# 2   Overcast         Hot     High    Weak         Yes
# 3       Rain        Mild     High    Weak         Yes
# 7      Sunny        Mild     High    Weak          No
# 11  Overcast        Mild     High  Strong         Yes
# 13      Rain        Mild     High  Strong          No
# Grouped Attribute Values
#       Outlook Temperature Humidity    Wind Play Tennis
# 4       Rain        Cool   Normal    Weak         Yes
# 5       Rain        Cool   Normal  Strong          No
# 6   Overcast        Cool   Normal  Strong         Yes
# 8      Sunny        Cool   Normal    Weak         Yes
# 9       Rain        Mild   Normal    Weak         Yes
# 10     Sunny        Mild   Normal  Strong         Yes
# 12  Overcast         Hot   Normal    Weak         Yes
# target attribute class count(Yes/No)= {'No': 4, 'Yes': 3}
# Total no of instances/records associated with High is: 7
# Probability of Class No is: 0.4286
# Probability of Class Yes is: 0.5714
# target attribute class count(Yes/No)= {'Yes': 6, 'No': 1}
# Total no of instances/records associated with Normal is: 7
# Probability of Class No is: 0.1429
# Probability of Class Yes is: 0.8571
# target attribute class count(Yes/No)= {'No': 5, 'Yes': 9}
# Total no of instances/records associated with S is: 14
# Probability of Class No is: 0.3571
# Probability of Class Yes is: 0.6429
# Information gain of  Humidity  is :  0.15183550136234136
#
#
# ----Information Gain Calculation of  Wind .........
# Grouped Attribute Values
#       Outlook Temperature Humidity    Wind Play Tennis
# 1      Sunny         Hot     High  Strong          No
# 5       Rain        Cool   Normal  Strong          No
# 6   Overcast        Cool   Normal  Strong         Yes
# 10     Sunny        Mild   Normal  Strong         Yes
# 11  Overcast        Mild     High  Strong         Yes
# 13      Rain        Mild     High  Strong          No
# Grouped Attribute Values
#       Outlook Temperature Humidity  Wind Play Tennis
# 0      Sunny         Hot     High  Weak          No
# 2   Overcast         Hot     High  Weak         Yes
# 3       Rain        Mild     High  Weak         Yes
# 4       Rain        Cool   Normal  Weak         Yes
# 7      Sunny        Mild     High  Weak          No
# 8      Sunny        Cool   Normal  Weak         Yes
# 9       Rain        Mild   Normal  Weak         Yes
# 12  Overcast         Hot   Normal  Weak         Yes
# target attribute class count(Yes/No)= {'No': 3, 'Yes': 3}
# Total no of instances/records associated with Strong is: 6
# Probability of Class No is: 0.5000
# Probability of Class Yes is: 0.5000
# target attribute class count(Yes/No)= {'No': 2, 'Yes': 6}
# Total no of instances/records associated with Weak is: 8
# Probability of Class No is: 0.2500
# Probability of Class Yes is: 0.7500
# target attribute class count(Yes/No)= {'No': 5, 'Yes': 9}
# Total no of instances/records associated with S is: 14
# Probability of Class No is: 0.3571
# Probability of Class Yes is: 0.6429
# Information gain of  Wind  is :  0.04812703040826927
#
# Attribute with the maximum gain is : Outlook
#
#
# ----Information Gain Calculation of  Temperature .........
# Grouped Attribute Values
#    Outlook Temperature Humidity    Wind Play Tennis
# 4    Rain        Cool   Normal    Weak         Yes
# 5    Rain        Cool   Normal  Strong          No
# Grouped Attribute Values
#     Outlook Temperature Humidity    Wind Play Tennis
# 3     Rain        Mild     High    Weak         Yes
# 9     Rain        Mild   Normal    Weak         Yes
# 13    Rain        Mild     High  Strong          No
# target attribute class count(Yes/No)= {'Yes': 1, 'No': 1}
# Total no of instances/records associated with Cool is: 2
# Probability of Class No is: 0.5000
# Probability of Class Yes is: 0.5000
# target attribute class count(Yes/No)= {'Yes': 2, 'No': 1}
# Total no of instances/records associated with Mild is: 3
# Probability of Class No is: 0.3333
# Probability of Class Yes is: 0.6667
# target attribute class count(Yes/No)= {'Yes': 3, 'No': 2}
# Total no of instances/records associated with S is: 5
# Probability of Class No is: 0.4000
# Probability of Class Yes is: 0.6000
# Information gain of  Temperature  is :  0.01997309402197489
#
#
# ----Information Gain Calculation of  Humidity .........
# Grouped Attribute Values
#     Outlook Temperature Humidity    Wind Play Tennis
# 3     Rain        Mild     High    Weak         Yes
# 13    Rain        Mild     High  Strong          No
# Grouped Attribute Values
#    Outlook Temperature Humidity    Wind Play Tennis
# 4    Rain        Cool   Normal    Weak         Yes
# 5    Rain        Cool   Normal  Strong          No
# 9    Rain        Mild   Normal    Weak         Yes
# target attribute class count(Yes/No)= {'Yes': 1, 'No': 1}
# Total no of instances/records associated with High is: 2
# Probability of Class No is: 0.5000
# Probability of Class Yes is: 0.5000
# target attribute class count(Yes/No)= {'Yes': 2, 'No': 1}
# Total no of instances/records associated with Normal is: 3
# Probability of Class No is: 0.3333
# Probability of Class Yes is: 0.6667
# target attribute class count(Yes/No)= {'Yes': 3, 'No': 2}
# Total no of instances/records associated with S is: 5
# Probability of Class No is: 0.4000
# Probability of Class Yes is: 0.6000
# Information gain of  Humidity  is :  0.01997309402197489
#
#
# ----Information Gain Calculation of  Wind .........
# Grouped Attribute Values
#     Outlook Temperature Humidity    Wind Play Tennis
# 5     Rain        Cool   Normal  Strong          No
# 13    Rain        Mild     High  Strong          No
# Grouped Attribute Values
#    Outlook Temperature Humidity  Wind Play Tennis
# 3    Rain        Mild     High  Weak         Yes
# 4    Rain        Cool   Normal  Weak         Yes
# 9    Rain        Mild   Normal  Weak         Yes
# target attribute class count(Yes/No)= {'No': 2}
# Total no of instances/records associated with Strong is: 2
# Probability of Class No is: 1.0000
# Probability of Class No is: 1.0000
# target attribute class count(Yes/No)= {'Yes': 3}
# Total no of instances/records associated with Weak is: 3
# Probability of Class Yes is: 1.0000
# Probability of Class Yes is: 1.0000
# target attribute class count(Yes/No)= {'Yes': 3, 'No': 2}
# Total no of instances/records associated with S is: 5
# Probability of Class No is: 0.4000
# Probability of Class Yes is: 0.6000
# Information gain of  Wind  is :  0.9709505944546686
#
# Attribute with the maximum gain is : Wind
#
#
# ----Information Gain Calculation of  Temperature .........
# Grouped Attribute Values
#    Outlook Temperature Humidity  Wind Play Tennis
# 8   Sunny        Cool   Normal  Weak         Yes
# Grouped Attribute Values
#    Outlook Temperature Humidity    Wind Play Tennis
# 0   Sunny         Hot     High    Weak          No
# 1   Sunny         Hot     High  Strong          No
# Grouped Attribute Values
#     Outlook Temperature Humidity    Wind Play Tennis
# 7    Sunny        Mild     High    Weak          No
# 10   Sunny        Mild   Normal  Strong         Yes
# target attribute class count(Yes/No)= {'Yes': 1}
# Total no of instances/records associated with Cool is: 1
# Probability of Class Yes is: 1.0000
# Probability of Class Yes is: 1.0000
# target attribute class count(Yes/No)= {'No': 2}
# Total no of instances/records associated with Hot is: 2
# Probability of Class No is: 1.0000
# Probability of Class No is: 1.0000
# target attribute class count(Yes/No)= {'No': 1, 'Yes': 1}
# Total no of instances/records associated with Mild is: 2
# Probability of Class No is: 0.5000
# Probability of Class Yes is: 0.5000
# target attribute class count(Yes/No)= {'No': 3, 'Yes': 2}
# Total no of instances/records associated with S is: 5
# Probability of Class No is: 0.4000
# Probability of Class Yes is: 0.6000
# Information gain of  Temperature  is :  0.5709505944546686
#
#
# ----Information Gain Calculation of  Humidity .........
# Grouped Attribute Values
#    Outlook Temperature Humidity    Wind Play Tennis
# 0   Sunny         Hot     High    Weak          No
# 1   Sunny         Hot     High  Strong          No
# 7   Sunny        Mild     High    Weak          No
# Grouped Attribute Values
#     Outlook Temperature Humidity    Wind Play Tennis
# 8    Sunny        Cool   Normal    Weak         Yes
# 10   Sunny        Mild   Normal  Strong         Yes
# target attribute class count(Yes/No)= {'No': 3}
# Total no of instances/records associated with High is: 3
# Probability of Class No is: 1.0000
# Probability of Class No is: 1.0000
# target attribute class count(Yes/No)= {'Yes': 2}
# Total no of instances/records associated with Normal is: 2
# Probability of Class Yes is: 1.0000
# Probability of Class Yes is: 1.0000
# target attribute class count(Yes/No)= {'No': 3, 'Yes': 2}
# Total no of instances/records associated with S is: 5
# Probability of Class No is: 0.4000
# Probability of Class Yes is: 0.6000
# Information gain of  Humidity  is :  0.9709505944546686
#
#
# ----Information Gain Calculation of  Wind .........
# Grouped Attribute Values
#     Outlook Temperature Humidity    Wind Play Tennis
# 1    Sunny         Hot     High  Strong          No
# 10   Sunny        Mild   Normal  Strong         Yes
# Grouped Attribute Values
#    Outlook Temperature Humidity  Wind Play Tennis
# 0   Sunny         Hot     High  Weak          No
# 7   Sunny        Mild     High  Weak          No
# 8   Sunny        Cool   Normal  Weak         Yes
# target attribute class count(Yes/No)= {'No': 1, 'Yes': 1}
# Total no of instances/records associated with Strong is: 2
# Probability of Class No is: 0.5000
# Probability of Class Yes is: 0.5000
# target attribute class count(Yes/No)= {'No': 2, 'Yes': 1}
# Total no of instances/records associated with Weak is: 3
# Probability of Class No is: 0.3333
# Probability of Class Yes is: 0.6667
# target attribute class count(Yes/No)= {'No': 3, 'Yes': 2}
# Total no of instances/records associated with S is: 5
# Probability of Class No is: 0.4000
# Probability of Class Yes is: 0.6000
# Information gain of  Wind  is :  0.01997309402197489
#
# Attribute with the maximum gain is : Humidity
#
# The Resultant Decision Tree is:
# {'Outlook': {'Overcast': 'Yes',
#              'Rain': {'Wind': {'Strong': 'No', 'Weak': 'Yes'}},
#              'Sunny': {'Humidity': {'High': 'No', 'Normal': 'Yes'}}}}
