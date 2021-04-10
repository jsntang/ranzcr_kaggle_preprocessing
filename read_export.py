#!/usr/bin/env python
# coding: utf-8

# In[3]:


import json,pprint,itertools,pprint
from collections import defaultdict
import pandas as pd


# In[4]:


data = json.load(open('mdai20200531.json')) #this file not available as it contains private information of the labellers


# In[5]:


user_map = {user['id']:user['email'] for user in data['users']}


# In[6]:


excluded_labellers = ['redacted@gmail.com']
assisted_labellers = ['redacted2@gmail.com']


# In[7]:


label_id_to_label_group = {}
label_id_to_name = {}
label_id_to_tubetype = {}
tubetype_to_label_name = {}
label_name_to_tubetype = {}
for i, label_group in enumerate(data['labelGroups']):
    for label in label_group['labels']:
        label_id_to_label_group[label['id']] = i
        label_id_to_name[label['id']] = label['name']
        label_type = label['name'].split(' ')[0].lower()
        label_id_to_tubetype[label['id']] = label_type
        label_name_to_tubetype[label['name']] = label_type
        if label_type not in tubetype_to_label_name:
            tubetype_to_label_name[label_type] = set()
        tubetype_to_label_name[label_type].add(label['name'])


# In[8]:


tubetype_to_label_name = {k:sorted(list(v)) for k,v in tubetype_to_label_name.items()}
tubetype_label_name_to_index = {k: {v:k for k,v in enumerate(label_names)} for k,label_names in tubetype_to_label_name.items()}


# In[9]:


import numpy as np
dataset = data['datasets'][0] #use the first dataset always
studyuid_to_examnum = {study['StudyInstanceUID']:study['number'] for study in dataset['studies']}
examnum_to_studyuid = {study['number']:study['StudyInstanceUID'] for study in dataset['studies']}
exam_num_to_index = {v:k for k,v in enumerate(list(set([studyuid_to_examnum[annotation['StudyInstanceUID']] for annotation in dataset['annotations']])))}
index_to_exam_num = {k:v for k,v in enumerate(list(set([studyuid_to_examnum[annotation['StudyInstanceUID']] for annotation in dataset['annotations']])))}
user_to_index = {v:k for k,v in enumerate([email for user_id,email in user_map.items() if email not in excluded_labellers])}


# In[10]:


clean_annotations = []
for annotation in dataset['annotations']:
    clean_annotations.append({
        'exam_num':studyuid_to_examnum[annotation['StudyInstanceUID']],
        'name': label_id_to_name[annotation['labelId']],
        'labeller': user_map[annotation['createdById']],
        'label_group': label_id_to_label_group[annotation['labelId']]
    })


# In[11]:


gold_annotations_dict = {'ett':defaultdict(list), 'ngt':defaultdict(list), 'cvc':defaultdict(list), 'swan':defaultdict(list), 'no':defaultdict(list)}
gold_indices = set()
for annotation in dataset['annotations']:
    if label_id_to_label_group[annotation['labelId']] == 4:        
        name = label_id_to_name[annotation['labelId']]
        tubetype = label_name_to_tubetype[name]
        examnum = studyuid_to_examnum[annotation['StudyInstanceUID']]
        gold_annotations_dict[tubetype][exam_num_to_index[examnum]].append(tubetype_label_name_to_index[tubetype][name])        
        gold_indices.add(exam_num_to_index[examnum])


# In[12]:


for tubetype in gold_annotations_dict:
    for index in gold_indices:
        if index not in gold_annotations_dict[tubetype]:
            gold_annotations_dict[tubetype][index].append(len(tubetype_label_name_to_index[tubetype]))


# In[14]:


labels_per_case_per_labeller = {}
for annotation in clean_annotations:
    if annotation['labeller'] not in excluded_labellers:
        if annotation['exam_num'] <= 30: #first 30 are demonstration cases 
            if annotation['labeller'] not in assisted_labellers:
                continue
        if annotation['exam_num'] not in labels_per_case_per_labeller:
            labels_per_case_per_labeller[annotation['exam_num']] = {}
        if annotation['label_group'] not in labels_per_case_per_labeller[annotation['exam_num']]:
            labels_per_case_per_labeller[annotation['exam_num']][annotation['label_group']] = {}
        if annotation['labeller'] not in labels_per_case_per_labeller[annotation['exam_num']][annotation['label_group']]:
            labels_per_case_per_labeller[annotation['exam_num']][annotation['label_group']][annotation['labeller']] = set()
            
        labels_per_case_per_labeller[annotation['exam_num']][annotation['label_group']][annotation['labeller']].add(annotation['name'])


# In[16]:


not_annotated = []
for i in range(1,45000):
    if i not in labels_per_case_per_labeller:
        not_annotated.append(i)
assert not len(not_annotated)


# In[18]:


for case in labels_per_case_per_labeller:
    for label_group in labels_per_case_per_labeller[case]:        
        annotators = list(labels_per_case_per_labeller[case][label_group].keys())
        annotators_not_in_assisted = [a for a in annotators if a not in assisted_labellers]
        annotators_in_assisted = [a for a in annotators if a in assisted_labellers]        
        if len(annotators_not_in_assisted) > 0 and len(annotators_in_assisted) > 0:
            chosen_annotator = annotators_not_in_assisted[0]
            for annotator in annotators_in_assisted:
                labels_per_case_per_labeller[case][label_group][chosen_annotator].update(labels_per_case_per_labeller[case][label_group][annotator])
                del labels_per_case_per_labeller[case][label_group][annotator]
        


# In[19]:


cvc_annotations = np.zeros((len(exam_num_to_index), len(user_to_index), len(tubetype_to_label_name['cvc'])+1))
ngt_annotations = np.zeros((len(exam_num_to_index), len(user_to_index), len(tubetype_to_label_name['ngt'])+1))
ett_annotations = np.zeros((len(exam_num_to_index), len(user_to_index), len(tubetype_to_label_name['ett'])+1)) #add one for labeller saw the case and did not see a line of that type
swan_annotations = np.zeros((len(exam_num_to_index), len(user_to_index), len(tubetype_to_label_name['swan'])+1)) #add one for labeller saw the case and did not see a line of that type


# In[20]:


for case in labels_per_case_per_labeller:
    for label_group in labels_per_case_per_labeller[case]:       
        for annotator in labels_per_case_per_labeller[case][label_group]:   
            cvc_found = False
            ngt_found = False
            ett_found = False
            swan_found= False
            for label_name in labels_per_case_per_labeller[case][label_group][annotator]:                
                if label_name_to_tubetype[label_name] == 'cvc':
                    cvc_annotations[exam_num_to_index[case]][user_to_index[annotator]][tubetype_label_name_to_index['cvc'][label_name]] = 1
                    cvc_found = True
                elif label_name_to_tubetype[label_name] == 'ngt':
                    ngt_annotations[exam_num_to_index[case]][user_to_index[annotator]][tubetype_label_name_to_index['ngt'][label_name]] = 1
                    ngt_found = True
                elif label_name_to_tubetype[label_name] == 'ett':
                    ett_annotations[exam_num_to_index[case]][user_to_index[annotator]][tubetype_label_name_to_index['ett'][label_name]] = 1
                    ett_found = True
                elif label_name_to_tubetype[label_name] == 'swan':
                    swan_annotations[exam_num_to_index[case]][user_to_index[annotator]][tubetype_label_name_to_index['swan']['Swan Ganz Catheter Present']] = 1
                    cvc_annotations[exam_num_to_index[case]][user_to_index[annotator]][tubetype_label_name_to_index['cvc']['CVC - Normal']] = 1
                    cvc_found = True
                    swan_found = True
            if not cvc_found:
                cvc_annotations[exam_num_to_index[case]][user_to_index[annotator]][-1] = 1
            if not ngt_found:
                ngt_annotations[exam_num_to_index[case]][user_to_index[annotator]][-1] = 1
            if not ett_found:
                ett_annotations[exam_num_to_index[case]][user_to_index[annotator]][-1] = 1
            if not swan_found:
                swan_annotations[exam_num_to_index[case]][user_to_index[annotator]][-1] = 1


# In[21]:


def add_selections(indices,num_lines,forgotten_index):
    '''
    We collect binary yes / no answers for the presence or absence of each type of line
    Since there can be more than one of each type of line present, if the number of unique indices 
    selected by the labeller is smaller than what we suspect the number of lines to be,
    then we construct all possibilities of selection by adding duplicates of the selections
    already made by that labeller, as well as accounting for the possibility that the labeller
    may have simply forgotten to label a line, by adding the forgotten index, which is 
    simply the number of unique types of lines in that category. E.g. if there are 
    Type 0, Type 1, and Type 2 lines, then the index representing the labeller forgetting a line
    is 3. 
    '''
    assert len(set(indices)) == len(list(indices)), 'must have only unique indices!!'
    selections = []
    if len(indices) == num_lines: #they selected as many lines as there were, no need to add synthetic lines
        selections.append(indices)
    else:
        difference = num_lines - len(indices)
        for added_indices in itertools.combinations(indices + [forgotten_index], difference):
            selections.append(indices+list(added_indices))
    return selections
    
    


# In[22]:


assert add_selections([0,1,2], 4, 3) == [[0, 1, 2, 0], [0, 1, 2, 1], [0, 1, 2, 2], [0, 1, 2, 3]]


# In[23]:


assert add_selections([0], 2, 3) == [[0, 0], [0, 3]]


# In[24]:


assert add_selections([0,2], 2, 3) == [[0, 2]]


# In[25]:


assert add_selections([3], 2, 3) == [[3, 3], [3, 3]]


# In[26]:


num_lines = 3
num_types_of_lines = 2
np.stack(np.indices((num_types_of_lines,)*num_lines), -1).reshape((-1, num_lines))


# In[27]:


def check_prob_of_possibility(possibility, selection, labeller_error_rate):
    return np.prod(labeller_error_rate[np.array(possibility), np.array(selection)])


# In[28]:


def find_gold_standard_for_case(labeller_selections, labeller_error_rates, preselected_gold_standard=None, verbose=False):
    '''
    labeller_indices = [[0,1,2],[1,2] ...]    
    '''
    num_types_of_lines = labeller_error_rates[0].shape[-1] - 1
    
    if preselected_gold_standard: 
        preselected_gold_standard = list(set(preselected_gold_standard))
        num_lines = max([len(indices) for indices in labeller_selections]) 
        if len(preselected_gold_standard) < num_lines:
            preselected_gold_standard = list(preselected_gold_standard)
            preselected_gold_standard.extend([num_types_of_lines]*(num_lines - len(preselected_gold_standard)))
        num_lines = len(preselected_gold_standard)
        assert len(preselected_gold_standard) >= max([len(indices) for indices in labeller_selections]) 
    else:
        num_lines = max([len(indices) for indices in labeller_selections])        
    
    
    
    augmented_labeller_selection = []
    # add in all the possible permutations to account for if someone forgot a line or if it was flipped the other way         
    for indices in labeller_selections:
        selections = add_selections(indices, num_lines, num_types_of_lines)
        permuted_selections = []
        for selection in selections:
            permuted_selections.extend(itertools.permutations(selection))
        augmented_labeller_selection.append(list(set(permuted_selections)))
    
    
    
    all_labellers_selected_no_line = True
    for indices in labeller_selections:
        for index in indices:
            if index != num_types_of_lines:
                all_labellers_selected_no_line = False 
    
    if preselected_gold_standard:
        #since gold standrad is provided, only one possibility exists
        possibilities = np.array([preselected_gold_standard])
    else:
        possibilities = list(np.stack(np.indices((num_types_of_lines,)*num_lines), -1).reshape((-1, num_lines)))
        if num_lines == 1 and all_labellers_selected_no_line:
            possibilities = [[num_types_of_lines,]]        
        possibilities = np.array(possibilities)
    
    probabilities = []
    best_selections_per_possibility = []
    for possibility in possibilities:  
        probability_of_this_possibility = 0    
        best_selection_per_labeller = []
        if verbose:
            print ('Considering possibility', possibility)
            print ('augmented labeller selection', augmented_labeller_selection)
        for i, permuted_selections in enumerate(augmented_labeller_selection):
            # assume permuted selections are all equally possible 
            labeller_error_rate = labeller_error_rates[i]
            probability_per_selection = [check_prob_of_possibility(possibility, selection, labeller_error_rate) for selection in permuted_selections]
            if verbose:
                pprint.pprint (list(zip(permuted_selections, probability_per_selection)))
            probability_of_this_possibility += np.max(probability_per_selection)
            best_selection_per_labeller.append(permuted_selections[np.argmax(probability_per_selection)])
        
        best_selections_per_possibility.append(best_selection_per_labeller)
        probabilities.append(probability_of_this_possibility+1E-10)
    
    if verbose:
        print ('Final list of probabilities per possibility')
        pprint.pprint (list(zip(possibilities,probabilities,best_selections_per_possibility)))
    chosen_index = np.argmax(probabilities)
    return possibilities[chosen_index], probabilities[chosen_index], best_selections_per_possibility[chosen_index]
        
        
    
    


# In[29]:


labeller_error_rates = [np.array([[0.9,0.0,0.1],
                                  [0.1,0.5,0.4],
                                  [0.1,0.1,0.8],
                                 ]),
                        np.array([[0.9,0.0,0.1],
                                  [0.0,0.9,0.1],
                                  [0.1,0.1,0.8],
                                 ])]
labeller_indices = [[2],[2]]


# In[52]:


assert find_gold_standard_for_case(labeller_indices, labeller_error_rates)[0] == np.array([2])


# In[96]:


labeller_error_rates = [np.array([[0.9,0.0,0.1],
                                  [0.0,0.9,0.1],
                                  [0.1,0.1,0.8],
                                 ]),
                        np.array([[0.9,0.0,0.1],
                                  [0.0,0.9,0.1],
                                  [0.1,0.1,0.8],
                                 ])]
labeller_indices = [[1],[0,1]] 
#Labeller 1 indicates that a single line of type 1 is present. Labeller 2 indicates that two lines, one of type 0 and one of type 1 are present


# In[97]:


find_gold_standard_for_case(labeller_indices, labeller_error_rates)[0] 
#since both labeller 1 and labeller 2 are both pretty good at type 0 and type 1 lines, but can sometimes miss them with a 0.1 chance, algo thinks most likely labeller 1 simply missed a type 0 line 


# In[102]:


labeller_error_rates = [np.array([[0.9,0.1,0.0],
                                  [0.1,0.9,0.0],
                                  [0.1,0.1,0.8],
                                 ]),
                        np.array([[0.8,0.2,0.0],
                                  [0.2,0.8,0.0],
                                  [0.1,0.1,0.8],
                                 ])]
find_gold_standard_for_case(labeller_indices, labeller_error_rates)[0] 
# now since labeller 2 is much weaker at identifying what type of line it is, and labeller 1 and 2 don't miss lines, the most likely explanation for the observed is that there are x2 type 1 lines. 
# labeller 1 observed x2 type 1 lines, but since we only annotate a binary yes/no
# this means that labeller 1 would still indicate this as [1] 


# In[31]:


def score_labeller(labeller, labeller_error_rates):
    confusion_matrix = labeller_error_rates[user_to_index[labeller]]
    return np.sum(np.eye(confusion_matrix.shape[0]) * confusion_matrix) / (confusion_matrix.shape[0])


# In[107]:


from tqdm import tqdm_notebook as tqdm
import pprint
def assess_annotation(target_annotations, predetermined_gold):
    cases_more_than_one = np.where((target_annotations.sum(axis=2) > 0).sum(axis=1) > 2)[0]
    annotations = target_annotations[cases_more_than_one]
    # initialise labeller error rates
    labeller_error_rates = np.array([np.eye(annotations.shape[-1]) for i in range(annotations.shape[1])])
    max_iterations = 100
    old_log_prob = 0
    for num_iterations in tqdm(range(max_iterations)):
        # Get gold standard
        gold_standards = []
        guessed_selections = []
        labeller_indices_per_case = []
        logprobs = []
        for index, case in enumerate(annotations):
            labeller_indices = np.where(case.sum(axis=1) > 0)[0]
            labeller_selections = [list(np.where(case[labeller_index])[0]) for labeller_index in labeller_indices]
            gold_standard, probability, guessed_selection = find_gold_standard_for_case(labeller_selections, 
                                                                labeller_error_rates[labeller_indices], 
                                                                preselected_gold_standard=(predetermined_gold[cases_more_than_one[index]] if cases_more_than_one[index] in predetermined_gold else None))
            
            gold_standards.append(gold_standard)
            labeller_indices_per_case.append(labeller_indices)
            guessed_selections.append(guessed_selection)   
            logprobs.append(np.log(probability))
        logprob = np.mean(logprobs)
        print (np.mean(logprobs))
        if np.abs(logprob - old_log_prob) < 1E-5:
            break
        old_log_prob = logprob
        

        # zero labeller error rates
        for i, error_rate in enumerate(labeller_error_rates):
            labeller_error_rates[i] = np.zeros((annotations.shape[-1], annotations.shape[-1]))
        # Estimate labeller error rates
        for i, gold_standard in enumerate(gold_standards):
            for j, labeller_index in enumerate(labeller_indices_per_case[i]):
                for k, gold_standard_label in enumerate(gold_standard):                
                    labeller_error_rates[labeller_index][gold_standard_label, guessed_selections[i][j][k]] += 1
        # normalize labeller error rates
        for i, error_rate in enumerate(labeller_error_rates):
            error_rate += np.ones(error_rate.shape[0]) * 1E-5
            error_rate += np.eye(error_rate.shape[0]) * 1E-4
            labeller_error_rates[i] = (error_rate) / (error_rate.sum(axis=0, keepdims=True) + 1E-10) 
    
    # initialise zero labeller error rates    
    final_labeller_error_rates = np.array([np.zeros((annotations.shape[-1], annotations.shape[-1])) for i in range(annotations.shape[1])])
    
    # Estimate labeller error rates
    for i, gold_standard in enumerate(gold_standards):
        for j, labeller_index in enumerate(labeller_indices_per_case[i]):
            for k, gold_standard_label in enumerate(gold_standard):                
                final_labeller_error_rates[labeller_index][gold_standard_label, guessed_selections[i][j][k]] += 1

            
    annotations = target_annotations
    gold_standards = []
    for index, case in enumerate(annotations):
        labeller_indices = np.where(case.sum(axis=1) > 0)[0]
        labeller_selections = [list(np.where(case[labeller_index])[0]) for labeller_index in labeller_indices]
        
        gold_standard, _, _ = find_gold_standard_for_case(labeller_selections, labeller_error_rates[labeller_indices], 
                                                          preselected_gold_standard=(predetermined_gold[index] if index in predetermined_gold else None))
            
            
        gold_standards.append(gold_standard)
    return [(index_to_exam_num[k], v) for k,v in enumerate(gold_standards)], final_labeller_error_rates, labeller_error_rates, sorted([(user,score_labeller(user, final_labeller_error_rates)) for user in user_to_index], key = lambda x:x[1], reverse=True)


# In[108]:


cvc_gold, cvc_error_rates, cvc_error_rates_2, cvc_scores = assess_annotation(cvc_annotations, predetermined_gold=gold_annotations_dict['cvc'])


# In[109]:


ett_gold, ett_error_rates, ett_error_rates_2, ett_scores = assess_annotation(ett_annotations, predetermined_gold=gold_annotations_dict['ett'])


# In[110]:


ngt_gold, ngt_error_rates, ngt_error_rates_2, ngt_scores = assess_annotation(ngt_annotations, predetermined_gold=gold_annotations_dict['ngt'])


# In[111]:


swan_gold, swan_error_rates, swan_error_rates_2,swan_scores = assess_annotation(swan_annotations, predetermined_gold=gold_annotations_dict['swan'])


# In[38]:


cvc_gold[exam_num_to_index[4968]]


# In[39]:


ett_gold[exam_num_to_index[4958]]


# In[40]:


ngt_gold[exam_num_to_index[4964]]


# In[42]:


tubetype_to_label_name


# In[43]:


# export
order = ['ett', 'ngt', 'cvc', 'swan']
headers = ['StudyInstanceUID'] + [d for x in order for d in tubetype_to_label_name[x]]

df = pd.DataFrame(columns=headers)
df.set_index(['StudyInstanceUID'], inplace=True)

for tubetype, gold in zip(order, [ett_gold, ngt_gold, cvc_gold, swan_gold]):
    gold = dict(gold)
    for exam_num in tqdm(range(1,45000)):
        for i in gold[exam_num]:
            if i < len(tubetype_to_label_name[tubetype]):
                df.loc[examnum_to_studyuid[exam_num],tubetype_to_label_name[tubetype][i]] = 1
df = df.fillna(value=0)


# In[ ]:


df.to_csv('labels.csv.gz')

