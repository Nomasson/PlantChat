import dialogflow_v2 as dialogflow
from flask import request
import json
from google.protobuf import struct_pb2
from .model import get_plant, get_plants

project_id = 'plant-bot-jnoxbj'
language_code = 'en'



def detect_intent_texts(plant, session_id, text):
    """Returns the result of detect intent with texts as inputs.

    Using the same `session_id` between requests allows continuation
    of the conversation."""
    
    session_client = dialogflow.SessionsClient()
    session = session_client.session_path(project_id, session_id)

    payload = struct_pb2.Struct()
    payload['plant_index'] = plant
    query_params = dialogflow.types.QueryParameters(payload=payload )

    text_input = dialogflow.types.TextInput(
        text=text, language_code=language_code)

    query_input = dialogflow.types.QueryInput(text=text_input)

    response = session_client.detect_intent(
        session=session, query_input=query_input, query_params=query_params)

    return (response.query_result.fulfillment_text)

def get_response():
    # build a request object
    req = request.get_json(force=True)
    plant_index = int(req.get('originalDetectIntentRequest').get('payload')['plant_index'])
    
    plant = get_plant(plant_index)
        
    action = req.get('queryResult').get('action')
    
    if action == 'get_preference':
        response = get_preference(plant, req)
        
        
    elif  action == 'get_description':
        response = get_description(plant, req)
        
    elif action == 'get_bloom_time':
        response = get_bloom_time(plant, req)
        
    elif action =='edible':
        response = edible(plant,req)    
        
    elif action =='get_features':
        response = get_features(plant,req)
        
    elif action =='get_life_span':
        response = get_life_span(plant,req)
        
    elif action =='get_toxicity':
        response = get_toxicity(plant,req)
        
    elif action =='get_price':
        response = get_price(plant,req)
        
        
    return {'fulfillmentText': plant['plant name']+' '+ response}
        


def get_preference(plant_data, req):
    
    
    with open('response.txt') as json_file:
        responses = json.load(json_file)
        
    preferences = req.get('queryResult').get('parameters').get('plant_Preferences')
    res = ''
    if len(preferences) == 0:
        preferences = ['sun requirements','water preferences']
    for i, pref in enumerate(preferences):
        
        plant_pref = plant_data[pref]
        if i>0:
            res +=" and "
        res += responses[plant_pref]['response']
    return res

def get_description(plant_data, req):
    
    descriptions = req.get('queryResult').get('parameters').get('plant_Description')
    res = ''
    if 'size' in descriptions:
        descriptions.append('height')
        descriptions.append('spread')
    for desc in descriptions:
        if res !='':
            res +=" and "
        if desc == 'height':
            res += 'height is ' + plant_data[desc]
            
        if desc == 'color':
            if 'color' in plant_data:
                res += 'flower color is '+plant_data['color']
            else:
                res += 'has no flowers'
                
        if desc == 'spread':
            if 'spread' in plant_data:
                res += 'spread '+plant_data['spread']
            else:
                if 'size' in descriptions:
                    break
                res += 'size {}'.format(plant_data['height'])
            
        if desc == 'leaves':
            res += 'is {} plant'.format(plant_data['leaves'])
            
    return res

def get_bloom_time(plant_data, req):
    res =''
    descriptions = req.get('queryResult').get('parameters').get('plant_Description')
    
    if 'color' in plant_data:
        if 'size' in descriptions:
            res += 'bloom time is {} and the size of the flower is {}'.format(plant_data['flower time'], plant_data['flower size'])
        else:
            res += 'bloom time is {}'.format(plant_data['flower time'])
    else:
        res += 'has no flowers'
    
    return res

def edible(plant_date, req):
    
    part = req.get('queryResult').get('parameters').get('plant_part')
    res =''
    if 'edible parts' in plant_date and part !='':
        if part == plant_date['Edible Parts']:
            res += '{} are edible'.format(plant_date['edible parts'])
        else:
            res += 'has no {} but the {} are edible'.format(part, plant_date['Edible Parts'])
        
    elif 'edible parts' in plant_date:
        res += '{} are edible'.format(plant_date['edible parts'])          
    else:
        res += 'is not an edible plant'
    
    return res

def get_features(plant_data, req):
    
    features = req.get('queryResult').get('parameters').get('plant_features')
    res =''
    
    if 'houseplant' in features:
        if 'suitable locations' in plant_data:
            res += 'can growing indoors and is {}'.format(plant_data['containers'])
        else:
            res += 'is not suitable For growing indoors'
        return res
    
    
    if 'container' in features:
        if 'containers' in plant_data:
            res += 'is {}'.format(plant_data['containers'])
            
        else:
            res +=' is not suitable to pots'
        return res
            
def get_life_span(plant_date, req):
    
    res =''
    res ='is a {} plant'.format(plant_date['life cycle'])
    return res

def get_toxicity(plant_date, req):
    
    if 'toxicity' in plant_date:
        return 'is toxic {}'.format(plant_date['toxicity'])
    
    res =''
    res += 'is not toxic'
    if 'edible parts' in plant_date:
        res += 'and the {} are edible'.format(plant_date['edible parts'])
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


    