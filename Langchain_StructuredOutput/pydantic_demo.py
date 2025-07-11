from pydantic import BaseModel,EmailStr
from typing import Optional

class Student (BaseModel):
    name:str = 'Garvit'
    age: Optional[int] = None
    email: EmailStr

new_student = {'email':'abc'} # raises error coz abc is not in email format

std=Student(**new_student)

print(std)