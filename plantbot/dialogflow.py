import dialogflow_v2 as dialogflow
from flask import request
import json
from google.protobuf import struct_pb2

project_id = 'plant-bot-jnoxbj'
language_code = 'en'

with open('response.txt') as json_file:
    responses = json.load(json_file)
    
with open('data.txt') as json_file:
    data = json.load(json_file)

def detect_intent_texts(plant, session_id, text):
    """Returns the result of detect intent with texts as inputs.

    Using the same `session_id` between requests allows continuation
    of the conversation."""
    
    session_client = dialogflow.SessionsClient()
    session = session_client.session_path(project_id, session_id)
    # print('Session path: {}\n'.format(session))

    payload = struct_pb2.Struct()
    payload['plant_index'] = plant
    query_params = dialogflow.types.QueryParameters(payload=payload )

    text_input = dialogflow.types.TextInput(
        text=text, language_code=language_code)

    query_input = dialogflow.types.QueryInput(text=text_input)

    response = session_client.detect_intent(
        session=session, query_input=query_input, query_params=query_params)

    print('=' * 20)
    # print(response)
    # print('Query text: {}'.format(response.query_result.query_text))
    # print('Detected intent: {} (confidence: {})\n'.format(
    #     response.query_result.intent.display_name,
    #     response.query_result.intent_detection_confidence))
    # print('Fulfillment text: {}\n'.format(
    #     response.query_result.fulfillment_text))
    return (response.query_result.fulfillment_text)

def get_response():
    # build a request object
    req = request.get_json(force=True)
    print('req: ', req)
    plant_index = int(req.get('originalDetectIntentRequest').get('payload')['plant_index'])
    
    
        
    action = req.get('queryResult').get('action')
   
    
    if action == 'get_preference':
        response = get_preference(data['plants'][plant_index], req)
        
        
    elif  action == 'get_description':
        response = get_Description(data['plants'][plant_index], req)
        
    elif action == 'get_bloom_time':
        response = get_bloom_time(data['plants'][plant_index], req)
        
    elif action =='edible':
        response = edible(data['plants'][plant_index],req)    
        
    elif action =='get_features':
        response = get_features(data['plants'][plant_index],req)
        
    elif action =='get_life_span':
        response = get_life_span(data['plants'][plant_index],req)
        
    elif action =='get_toxicity':
        response = get_toxicity(data['plants'][plant_index],req)
        
    elif action =='get_price':
        response = get_price(data['plants'][plant_index],req)
        
        
    return {'fulfillmentText': data['plants'][plant_index]['plant name']+' '+ response}
        


def get_preference(plant_data, req):
    
    preferences = req.get('queryResult').get('parameters').get('plant_Preferences')
    res = ''
    if len(preferences) == 0:
        preferences = ['Sun Requirements','Water Preferences']
    for i, pref in enumerate(preferences):
        
        plant_pref = plant_data[pref]
        if i>0:
            res +=" and "
        res += responses[plant_pref]['response']
    return res

def get_Description(plant_data, req):
    
    descriptions = req.get('queryResult').get('parameters').get('plant_Description')
    parts = req.get('queryResult').get('parameters').get('plant_part')
    res = ''
        
    for i, desc in enumerate(descriptions):
        if i>0:
            res +=" and "
        print('in pref===============')  
        print(desc)
        if desc == 'height':
            res += 'height is ' + plant_data[desc]
            
        if desc == 'color':
            if plant_data['Color'] != "":
                res += 'flower color is '+plant_data['Color']
                
        if desc == 'spread':
            res += 'spread '+plant_data['Spread']
            
        if desc == 'leaves':
            res += 'is {} plant'.format(plant_data['Leaves'])
            
    return res

def get_bloom_time(plant_data, req):
    
    res = 'bloom time is {}'.format(plant_data['Flower Time'])
    return res

def edible(plant_date, req):
    
    part = req.get('queryResult').get('parameters').get('plant_part')
    res =''
    if plant_date['Edible Parts'] !='':
        if part == plant_date['Edible Parts']:
            res += '{} are edible'.format(plant_date['Edible Parts'])
        else:
            res += 'has no {} but the leaves are edible'.format(part)
        
            
    else:
        res += 'is not an edible plant'
    
    return res

def get_features(plant_data, req):
    
    features = req.get('queryResult').get('parameters').get('plant_features')
    res =''
    
    if 'Houseplant' in features:
        if plant_data['Suitable Locations'] == 'Houseplant':
            res += 'can growing indoors and is {}'.format(plant_data['Containers'])
        else:
            res += 'is not suitable For growing indoors'
        return res
    
    
    if 'container' in features:
        if plant_data['Containers'] !='':
            res += 'is {}'.format(plant_data['Containers'])
            
        else:
            res +=' is not suitable to pots'
        return res
            
def get_life_span(plant_date, req):
    
    res =''
    res ='is a {} plant'.format(plant_date['Life cycle'])
    return res

def get_toxicity(plant_date, req):
    
    if plant_date['Toxicity'] != '':
        return 'is toxic {}'.format(plant_date['Toxicity'])
    
    res =''
    res += 'is not toxic'
    if plant_date['Edible Parts'] !='':
        res += 'and the {} are edible'.format(plant_date['Edible Parts'])
    else:
        res += ' but is not an edible plant'
    return res
        
      
def get_price(plant_date, req):
    
    
    quantity = req.get('queryResult').get('parameters').get('number')
    if quantity == '':
        quantity =1
    res =''
    price = plant_date['price']
    res +='cost {} for {} plant'.format(quantity*price, quantity)
    return res


    