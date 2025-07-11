from typing import TypedDict

class person(TypedDict):
    name:str
    age:int

new_person: person = {
    'name': 'Garvit',
    'age':35
}

print(new_person)