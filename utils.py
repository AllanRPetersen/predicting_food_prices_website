### api's functions to retrieve nutrients ###
import pandas as pd
import numpy as np
import requests

def nutrients_redux(food):

    product = food

    API_KEY = 'zYKJIFsOxODivrK9dGg8Y2UUWxkr2j2HMNYXLLwf'

    params = {
        'api_key': API_KEY,
        'query': product,
        'pageSize': 1,
        'numberOfResultsPerPage': 1,
        #'dataType':"Foundation"
    }
    response = requests.get(' https://api.nal.usda.gov/fdc/v1/foods/search',
                            params=params).json()

    protein = 'not found'
    fat = 'not found'
    carb = 'not found'
    sugar = 'not found'
    sodium = 'not found'
    cholesterol = 'not found'
    calcium = 'not found'
    energy = 'not found'
    size = 100
    unit = 'g'
    category = 'not found'

    if response['totalHits'] != 0:
        try:

            for element in response['foods'][0]['foodNutrients']:
                if element['nutrientName'] == 'Protein':
                    protein = element['value']
                if element['nutrientName'] == 'Total lipid (fat)':
                    fat = element['value']
                if element['nutrientName'] == 'Carbohydrate, by difference':
                    carb = element['value']
                if element[
                        'nutrientName'] == 'Sugars, Total NLEA' or 'Sugars, total including NLEA':
                    sugar = element['value']
                if element['nutrientName'] == 'Sodium, Na':
                    sodium = element['value']
                if element['nutrientName'] == 'Cholesterol':
                    cholesterol = element['value']
                if element['nutrientName'] == 'Calcium, Ca':
                    calcium = element['value']
                if element['nutrientName'] == 'Energy':
                    energy = element['value']
            size = response['foods'][0]['servingSize']
            unit = response['foods'][0]['servingSizeUnit']
        except:
            print(f'missing branded info for {food}')

        result = {
            'protein': protein,
            'fat': fat,
            'carb': carb,
            'sugar': sugar,
            'sodium': sodium,
            'cholesterol': cholesterol,
            'calcium': calcium,
            'kcal': energy,
            'size': size,
            'unit': unit,
            'category': response['foods'][0]['foodCategory']
        }
        return result

    result = {
        'protein': protein,
        'fat': fat,
        'carb': carb,
        'sugar': sugar,
        'sodium': sodium,
        'cholesterol': cholesterol,
        'calcium': calcium,
        'kcal': energy,
        'size': 'not found',
        'unit': 'not found',
        'category': category
    }
    print('process failed')

    return result

#### nutrient redux 2 ###

def nutrients_redux_2(food):

    product = food

    API_KEY = 'zYKJIFsOxODivrK9dGg8Y2UUWxkr2j2HMNYXLLwf'

    params = {
        'api_key': API_KEY,
        'query': product,
        'pageSize': 1,
        'numberOfResultsPerPage': 1,
        'dataType': "Foundation"
    }
    response = requests.get(' https://api.nal.usda.gov/fdc/v1/foods/search',
                            params=params).json()

    protein = 'not found'
    fat = 'not found'
    carb = 'not found'
    sugar = 'not found'
    sodium = 'not found'
    cholesterol = 'not found'
    calcium = 'not found'
    energy = 'not found'
    size = 100
    unit = 'g'
    category = 'not found'

    if response['totalHits'] != 0:
        try:

            for element in response['foods'][0]['foodNutrients']:
                if element['nutrientName'] == 'Protein':
                    protein = element['value']
                if element['nutrientName'] == 'Total lipid (fat)':
                    fat = element['value']
                if element['nutrientName'] == 'Carbohydrate, by difference':
                    carb = element['value']
                if element[
                        'nutrientName'] == 'Sugars, Total NLEA' or 'Sugars, total including NLEA':
                    sugar = element['value']
                if element['nutrientName'] == 'Sodium, Na':
                    sodium = element['value']
                if element['nutrientName'] == 'Cholesterol':
                    cholesterol = element['value']
                if element['nutrientName'] == 'Calcium, Ca':
                    calcium = element['value']
                if element['nutrientName'] == 'Energy':
                    energy = element['value']
            size = response['foods'][0]['servingSize']
            unit = response['foods'][0]['servingSizeUnit']
        except:
            print(f'missing foundation info for {food}')

        result = {
            'protein': protein,
            'fat': fat,
            'carb': carb,
            'sugar': sugar,
            'sodium': sodium,
            'cholesterol': cholesterol,
            'calcium': calcium,
            'kcal': energy,
            'size': size,
            'unit': unit,
            'category': response['foods'][0]['foodCategory']
        }
        return result

    result = {
        'protein': protein,
        'fat': fat,
        'carb': carb,
        'sugar': sugar,
        'sodium': sodium,
        'cholesterol': cholesterol,
        'calcium': calcium,
        'kcal': energy,
        'size': 'not found',
        'unit': 'not found',
        'category': category
    }

    return result

###### Nutrients super ####

def nutrients_super(food):

    try:
        product = food

        result = nutrients_redux_2(product)

        if sum(x == 'not found' for x in result.values()) >= 3:
            result = nutrients_redux(product)
            print('switched to branded')
        else:
            print('foundation')
    except:
        print('crashed, forced nutrients_redux')
        result = nutrients_redux(product)
    return result
