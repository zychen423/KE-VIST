import json
with open('vist_scored_terms_6_path.json') as jsonfile:
  output = json.load(jsonfile)
  
new_data = []
for key in output.keys():
    tmp_list = []
    for item in output[key]:
        tmp_dict = {'text':'',
                    'predicted_term_seq':item,
                    'story_id':key
                   }
        tmp_list.append(tmp_dict)
    new_data.append(tmp_list)
   
with open('vist_scored_terms_6_path_for_stage3.json', 'w') as jsonfile:
  json.dump(new_data, jsonfile, indent=4)
