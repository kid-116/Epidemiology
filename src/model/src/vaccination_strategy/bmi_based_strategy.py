import random

def vaccination_strategy(model):
    vaccination_count = int(0.2 * model.metadata.shape[0])
    priority_people = [int(person) for idx, person in enumerate(model.metadata['id']) 
    if int(model.metadata['bmi'][idx]) < 18 and int(model.metadata['bmi'][idx]) > 25]
    return random.sample(priority_people, k=min(vaccination_count, len(priority_people)))
