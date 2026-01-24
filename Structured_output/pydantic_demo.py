from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class student(BaseModel):
    name : str 
    age : Optional[int] = None
    email : EmailStr
    cgpa : float = Field(gt= 0 , lt=10, default=5 , description='decimal value representing the cgpa of a student') # cgpa should be between 0 to 10

new_student = {
    'name' : 'John Doe', #student ka naam humesa string hi hoga
    'age' : 21 , # if age not set then it will be None
    'email' : 'aditya@gmail.com',
    'cgpa' : 8.5
}

student_obj = student(**new_student)
print(student_obj)